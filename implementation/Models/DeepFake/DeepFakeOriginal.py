import random
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL.Image import BICUBIC
from torch.nn import DataParallel
from pathlib import Path

from torchvision.transforms import ToTensor
from Models.DeepFake.Autoencoder import AutoEncoder


class DeepFakeOriginal:
    def __init__(self, encoder, decoder, auto_encoder=AutoEncoder):
        """
        Initialize a new DeepFakeOriginal.
        """
        self.encoder = encoder().cuda()
        self.decoder1 = decoder().cuda()
        self.decoder2 = decoder().cuda()

        self.autoencoder1 = auto_encoder(self.encoder, self.decoder1).cuda()
        self.autoencoder2 = auto_encoder(self.encoder, self.decoder2).cuda()

        # use multiple gpus
        if torch.cuda.device_count() > 1:
            self.autoencoder1 = DataParallel(self.autoencoder1)
            self.autoencoder2 = DataParallel(self.autoencoder2)

        self.lossfn = torch.nn.L1Loss(size_average=True).cuda()

        self.optimizer1 = Adam(self.autoencoder1.parameters(), lr=1e-4)
        self.scheduler1 = ReduceLROnPlateau(self.optimizer1, patience=100, cooldown=50)
        self.optimizer2 = Adam(self.autoencoder2.parameters(), lr=1e-4)
        self.scheduler2 = ReduceLROnPlateau(self.optimizer2, patience=100, cooldown=50)

    def train(self, current_epoch, batches):
        loss1_mean, loss2_mean = 0, 0
        face1 = None
        face1_warped = None
        output1 = None
        face2 = None,
        face2_warped = None
        output2 = None
        iterations = 0

        for (face1_warped, face1), (face2_warped, face2) in batches:
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

    def validate(self, batches):
        loss1_valid_mean, loss2_valid_mean = 0, 0
        iterations = 0

        for (face1_warped, face1), (face2_warped, face2) in batches:
            with torch.no_grad():
                face1, face2 = face1.cuda(), face2.cuda()
                face1_warped, face2_warped = face1_warped.cuda(), face2_warped.cuda()

                output1 = self.autoencoder1(face1_warped)
                loss1_valid_mean += self.lossfn(output1, face1)

                output2 = self.autoencoder2(face2_warped)
                loss2_valid_mean += self.lossfn(output2, face2)

                iterations += 1

        loss1_valid_mean /= iterations
        loss2_valid_mean /= iterations
        loss1_valid_mean = loss1_valid_mean.cpu().data.numpy()
        loss2_valid_mean = loss2_valid_mean.cpu().data.numpy()

        return loss1_valid_mean, loss2_valid_mean

    def anonymize(self, x):
        return self.autoencoder2(x)

    def anonymize_2(self, x):
        return self.autoencoder1(x)

    def log(self, logger, epoch, loss1, loss2, images, log_images=False):
        """
        use logger to log current loss etc...
        :param logger: logger used to log
        :param epoch: current epoch
        """
        logger.log_loss(epoch=epoch, loss={'lossA': float(loss1), 'lossB': float(loss2)})
        logger.log_fps(epoch=epoch)

        # log images
        if log_images:
            examples = int(len(images[0]))
            example_indices = random.sample(range(0, examples - 1), 5)

            anonymized_images_trump = self.anonymize(images[2][example_indices])
            anonymized_images_cage = self.anonymize_2(images[5][example_indices])
            A = []
            B = []
            for idx, i in enumerate(example_indices):
                for j in range(3):
                    A.append(images[j].cpu()[i] * 255.00)
                    B.append(images[3 + j].cpu()[i] * 255.00)
                A.append(anonymized_images_trump.cpu()[idx] * 255.00)
                B.append(anonymized_images_cage.cpu()[idx] * 255.00)
            logger.log_images(epoch, A, "sample_input/A", 4)
            logger.log_images(epoch, B, "sample_input/B", 4)
        logger.save_model(epoch)

    def log_validate(self, logger, epoch, loss1, loss2):
        logger.log_loss(epoch=epoch, loss={'lossA_val': float(loss1), 'lossB_val': float(loss2)})

    def img2latent_bridge(self, extracted_face, extracted_information, img_size):
        return ToTensor()(extracted_face.resize(img_size, resample=BICUBIC)).unsqueeze(0).cuda()
