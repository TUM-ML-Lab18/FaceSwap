#
# This file includes the main functionality of the FaceAnonymizer module
# Author: Alexander Becker
#

import torch
from torch.autograd import Variable
from torch.nn import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.optim import Adam
from datetime import datetime
from pathlib import Path

from FaceAnonymizer.models.Decoder import Decoder
from FaceAnonymizer.models.Encoder import Encoder
from FaceAnonymizer.models.Autoencoder import AutoEncoder
from Logging.LoggingUtils import Logger


class Anonymizer:

    def __init__(self, data1, data2, batch_size=64, epochs=50000, learning_rate=1e-4):
        """
        Initialize a new Anonymizer.

        Inputs:
        - data1: dataset of pictures of first face
        - data2: dataset of pictures of second face
        - batch_size: batch size
        - epochs: number of training epochs
        - learning_rate: learning rate
        """
        self.learning_rate = learning_rate

        self.encoder = Encoder((3, 64, 64), 1024).cuda()
        self.decoder1 = Decoder(512).cuda()
        self.decoder2 = Decoder(512).cuda()

        self.autoencoder1 = AutoEncoder(self.encoder, self.decoder1).cuda()
        self.autoencoder2 = AutoEncoder(self.encoder, self.decoder2).cuda()

        # use multiple gpus
        if torch.cuda.device_count() > 1:
            self.autoencoder1 = DataParallel(self.autoencoder1)
            self.autoencoder2 = DataParallel(self.autoencoder2)
            batch_size *= torch.cuda.device_count()

        self.lossfn = torch.nn.L1Loss(size_average=True).cuda()
        self.dataLoader1 = DataLoader(data1, batch_size, shuffle=True, num_workers=4, drop_last=True)
        self.dataLoader2 = DataLoader(data2, batch_size, shuffle=True, num_workers=4, drop_last=True)
        self.epochs = epochs

    def train(self):
        logger = Logger()

        optimizer1 = Adam(self.autoencoder1.parameters(), lr=self.learning_rate)
        scheduler1 = ReduceLROnPlateau(optimizer1, 'min', verbose=True, threshold=1e-2, patience=10)
        optimizer2 = Adam(self.autoencoder2.parameters(), lr=self.learning_rate)
        scheduler2 = ReduceLROnPlateau(optimizer2, 'min', verbose=True, threshold=1e-2, patience=10)

        for i_epoch in range(self.epochs):
            loss1_mean, loss2_mean = 0, 0
            face1 = None
            face1_warped = None
            output1 = None
            face2 = None,
            face2_warped = None
            output2 = None
            iterations = 0
            for (face1_warped, face1), (face2_warped, face2) in zip(self.dataLoader1, self.dataLoader2):
                # face1 and face2 contain a batch of images of the first and second face, respectively
                face1, face2 = Variable(face1).cuda(), Variable(face2).cuda()
                face1_warped, face2_warped = Variable(face1_warped).cuda(), Variable(face2_warped).cuda()

                optimizer1.zero_grad()
                output1 = self.autoencoder1(face1_warped)
                loss1 = self.lossfn(output1, face1)
                loss1.backward()
                optimizer1.step()

                optimizer2.zero_grad()
                output2 = self.autoencoder2(face2_warped)
                loss2 = self.lossfn(output2, face2)
                loss2.backward()
                optimizer2.step()

                loss1_mean += loss1
                loss2_mean += loss2
                iterations += 1

            loss1_mean /= iterations
            loss2_mean /= iterations

            # scheduler1.step(loss1_mean.cpu().data.numpy(), i_epoch)
            # scheduler2.step(loss2_mean.cpu().data.numpy(), i_epoch)
            logger.log(i_epoch, loss1_mean, loss2_mean, self.autoencoder1,
                       images=[face1_warped, output1, face1, face2_warped, output2, face2])

    def anonymize(self, x):
        return self.autoencoder2(x)

    # TODO: Use save & load functions from models -> memory independent (RAM vs GPU)
    def save_model(self, path):
        # Create subfolder for models
        path = Path(path)
        subfolder = datetime.now().strftime('model__%Y%m%d_%H%M%S')
        path = path / subfolder
        path.mkdir(parents=True)
        self.encoder.save(path/'encoder.model')
        self.decoder1.save(path/'decoder1.model')
        self.decoder2.save(path/'decoder2.model')


    def load_model(self, path):
        path = Path(path)
        self.encoder.load(path/'encoder.model')
        self.decoder1.load(path/'decoder1.model')
        self.decoder2.load(path/'decoder2.model')
