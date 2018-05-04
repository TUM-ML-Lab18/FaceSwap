#
# This file includes the main functionality of the FaceAnonymizer module
# Author: Alexander Becker
#

import torch
from torch.nn import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from fastai.dataloader import DataLoader # todo decide which dataloader we want to use
from torch.optim import Adam
from pathlib import Path

from FaceAnonymizer.models.Decoder import Decoder
from FaceAnonymizer.models.Encoder import Encoder
from FaceAnonymizer.models.Autoencoder import AutoEncoder


class DeepFakeOriginal:
    def __init__(self, data_loader, input_dim=(2, 64, 64), latent_dim=1024, encoder=Encoder,
                 decoder=Decoder,
                 auto_encoder=AutoEncoder, loss_function=torch.nn.L1Loss(size_average=True), scheduler_arguments=None,
                 optimizer_arguments=None):
        """
        Initialize a new DeepFakeOriginal.

        Inputs:
        - data1: dataset of pictures of first face
        - data2: dataset of pictures of second face
        - batch_size: batch size
        - epochs: number of training epochs
        - learning_rate: learning rate
        """
        self.data_loader = data_loader
        self.encoder = encoder(input_dim=input_dim, latent_dim=1024).cuda()
        self.decoder1 = decoder(latent_dim=latent_dim // 2).cuda()
        self.decoder2 = decoder(latent_dim=latent_dim // 2).cuda()

        self.autoencoder1 = auto_encoder(self.encoder, self.decoder1).cuda()
        self.autoencoder2 = auto_encoder(self.encoder, self.decoder2).cuda()

        # use multiple gpus
        if torch.cuda.device_count() > 1:
            self.autoencoder1 = DataParallel(self.autoencoder1)
            self.autoencoder2 = DataParallel(self.autoencoder2)

        self.lossfn = loss_function.cuda()

        self.optimizer1 = Adam(self.autoencoder1.parameters(), **optimizer_arguments)
        self.scheduler1 = ReduceLROnPlateau(self.optimizer1, **scheduler_arguments)
        self.optimizer2 = Adam(self.autoencoder2.parameters(), **optimizer_arguments)
        self.scheduler2 = ReduceLROnPlateau(self.optimizer2, **scheduler_arguments)

    def train(self, current_epoch):
        loss1_mean, loss2_mean = 0, 0
        face1 = None
        face1_warped = None
        output1 = None
        face2 = None,
        face2_warped = None
        output2 = None
        iterations = 0

        for (face1_warped, face1), (face2_warped, face2) in self.data_loader:
            # face1 and face2 contain a batch of images of the first and second face, respectively
            face1, face2 = face1.cuda(), face2.cuda()
            face1_warped, face2_warped = face1_warped.cuda(), face2_warped.cuda()

            self.optimizer1.zero_grad()
            output1 = self.autoencoder1(face1_warped)
            loss1 = self.lossfn(output1, face1)
            loss1.backward()
            self.optimizer1.step()

            self.optimizer2.zero_grad()
            output2 = self.autoencoder2(face2_warped)
            loss2 = self.lossfn(output2, face2)
            loss2.backward()
            self.optimizer2.step()

            loss1_mean += loss1
            loss2_mean += loss2
            iterations += 1

        loss1_mean /= iterations
        loss2_mean /= iterations
        loss1_mean = loss1_mean.cpu().data.numpy()
        loss2_mean = loss2_mean.cpu().data.numpy()
        self.scheduler1.step(loss1_mean, current_epoch)
        self.scheduler2.step(loss2_mean, current_epoch)

        return loss1_mean, loss2_mean, [face1_warped, output1, face1, face2_warped, output2, face2]

    def anonymize(self, x):
        return self.autoencoder2(x)

    def anonymize_2(self, x):
        return self.autoencoder1(x)

    # TODO: Use save & load functions from models -> memory independent (RAM vs GPU)
    def save_model(self, path):
        # Create subfolder for models
        path = Path(path)
        subfolder = "model"  # "#datetime.now().strftime('model__%Y%m%d_%H%M%S')
        path = path / subfolder
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save(path / 'encoder.model')
        self.decoder1.save(path / 'decoder1.model')
        self.decoder2.save(path / 'decoder2.model')

    def load_model(self, path):
        path = Path(path)
        self.encoder.load(path / 'encoder.model')
        self.decoder1.load(path / 'decoder1.model')
        self.decoder2.load(path / 'decoder2.model')
