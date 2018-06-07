import random

import torch
from PIL.Image import BICUBIC
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import ToTensor

from Models.DeepFake.Autoencoder import AutoEncoder
from Models.ModelUtils.ModelUtils import CombinedModel


class DeepFakeOriginal(CombinedModel):
    def __init__(self, encoder, decoder, auto_encoder=AutoEncoder, **kwargs):
        """
        Initialize a new DeepFakeOriginal.
        """
        self.encoder = encoder(self.img_size).cuda()
        self.decoder1 = decoder().cuda()
        self.decoder2 = decoder().cuda()

        self.select_ae = kwargs.get('select_autoencoder', 1)
        self.autoencoder1 = auto_encoder(self.encoder, self.decoder1).cuda()
        self.autoencoder2 = auto_encoder(self.encoder, self.decoder2).cuda()

        self.lossfn = torch.nn.L1Loss(size_average=True).cuda()

        self.optimizer1 = Adam(self.autoencoder1.parameters(), lr=1e-4)
        self.scheduler1 = ReduceLROnPlateau(self.optimizer1, patience=100, cooldown=50)
        self.optimizer2 = Adam(self.autoencoder2.parameters(), lr=1e-4)
        self.scheduler2 = ReduceLROnPlateau(self.optimizer2, patience=100, cooldown=50)

    def get_models(self):
        return [self.encoder, self.decoder1, self.decoder2]

    def get_model_names(self):
        return ['encoder', 'decoder1', 'decoder2']

    def get_remaining_modules(self):
        return [self.autoencoder1, self.autoencoder2, self.lossfn, self.optimizer1, self.optimizer2, self.scheduler1,
                self.scheduler2]

    def train(self, train_data_loader, batch_size, validate, **kwargs):
        current_epoch = kwargs.get('current_epoch', -1)

        loss1_mean, loss2_mean = 0, 0
        face1 = None
        face1_warped = None
        output1 = None
        face2 = None,
        face2_warped = None
        output2 = None
        iterations = 0

        for (face1_warped, face1), (face2_warped, face2) in train_data_loader:
            # face1 and face2 contain a batch of images of the first and second face, respectively
            face1, face2 = face1.cuda(), face2.cuda()
            face1_warped, face2_warped = face1_warped.cuda(), face2_warped.cuda()

            if not validate:
                self.optimizer1.zero_grad()

            output1 = self.autoencoder1(face1_warped)
            loss1 = self.lossfn(output1, face1)

            if not validate:
                loss1.backward()
                self.optimizer1.step()

            if not validate:
                self.optimizer2.zero_grad()

            output2 = self.autoencoder2(face2_warped)
            loss2 = self.lossfn(output2, face2)

            if not validate:
                loss2.backward()
                self.optimizer2.step()

            loss1_mean += loss1
            loss2_mean += loss2
            iterations += 1

        loss1_mean /= iterations
        loss2_mean /= iterations
        loss1_mean = loss1_mean.cpu().data.numpy()
        loss2_mean = loss2_mean.cpu().data.numpy()
        if not validate:
            self.scheduler1.step(loss1_mean, current_epoch)
            self.scheduler2.step(loss2_mean, current_epoch)

        if not validate:
            log_info = {'lossA': float(loss1_mean), 'lossB': float(loss2_mean)}
        else:
            log_info = {'lossA_val': float(loss1_mean), 'lossB_val': float(loss2_mean)}

        return log_info, [face1_warped, output1, face1, face2_warped, output2, face2]

    def anonymize(self, extracted_face, **kwargs):
        extracted_face = ToTensor()(extracted_face.resize((128, 128), resample=BICUBIC)).unsqueeze(0).cuda()
        if self.select_ae == 1:
            output = self.autoencoder1(extracted_face)
        else:
            output = self.autoencoder2(extracted_face)

        return output

    def log_images(self, logger, epoch, images, validation=True):
        examples = int(len(images[0]))
        example_indices = random.sample(range(0, examples - 1), 5)

        anonymized_images_trump = self.anonymize(images[2][example_indices])
        anonymized_images_cage = self.anonymize_2(images[5][example_indices])
        A = []
        B = []
        for idx, i in enumerate(example_indices):
            for j in range(3):
                A.append(images[j].cpu()[i])
                B.append(images[3 + j].cpu()[i])
            A.append(anonymized_images_trump.cpu()[idx])
            B.append(anonymized_images_cage.cpu()[idx])
            tag = 'validation_output' if validation else 'training_output'
        logger.log_images(epoch, A, f"{tag}/A", 4)
        logger.log_images(epoch, B, f"{tag}/B", 4)
