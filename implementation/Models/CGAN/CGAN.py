import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from Configuration.config_general import ARRAY_CELEBA_LANDMARKS_MEAN, ARRAY_CELEBA_LANDMARKS_COV
from Models.CGAN.Discriminator import Discriminator
from Models.CGAN.Generator import Generator
from Models.ModelUtils.ModelUtils import CombinedModels


class CGAN(CombinedModels):
    def anonymize(self, x):
        z = torch.ones((x.shape[0], self.z_dim, 1, 1)).cuda() * 0.5
        # z = torch.randn((x.shape[0], self.z_dim, 1, 1)).cuda()
        return self.G(z, x)

    def img2latent_bridge(self, extracted_face, extracted_information):
        landmarks_normalized_flat = np.reshape(
            (np.array(extracted_information.landmarks) / extracted_information.size_fine).tolist(), -1).astype(
            np.float32)
        return torch.from_numpy(landmarks_normalized_flat).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).cuda()

    def get_models(self):
        return [self.G, self.D]

    def get_model_names(self):
        return ['generator', 'discriminator']

    def __str__(self):
        string = super().__str__()
        string += str(self.G_optimizer) + '\n'
        string += str(self.D_optimizer) + '\n'
        string += str(self.BCE_loss) + '\n'
        string = string.replace('\n', '\n\n')
        return string

    def __init__(self, **kwargs):
        self.z_dim = kwargs.get('z_dim', 100)
        self.y_dim = kwargs.get('y_dim', 10)
        self.img_dim = kwargs.get('img_dim', (64, 64, 3))
        path_to_y_mean = kwargs.get('y_mean', ARRAY_CELEBA_LANDMARKS_MEAN)
        path_to_y_cov = kwargs.get('y_cov', ARRAY_CELEBA_LANDMARKS_COV)
        lrG = kwargs.get('lrG', 0.0002)
        lrD = kwargs.get('lrD', 0.0002)

        self.G = Generator(input_dim=(self.z_dim, self.y_dim), output_dim=self.img_dim, ngf=32)
        self.D = Discriminator(y_dim=self.y_dim, input_dim=self.img_dim, ndf=32)

        beta1, beta2 = 0.5, 0.999
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=lrG, betas=(beta1, beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=lrD, betas=(beta1, beta2))

        self.BCE_loss = nn.BCELoss()

        # gaussian distribution of our landmarks
        self.y_mean = np.load(path_to_y_mean)
        self.y_cov = np.load(path_to_y_cov)

        if torch.cuda.is_available():
            self.cuda = True
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss.cuda()

    def train(self, train_data_loader, batch_size, **kwargs):

        G_loss_mean, D_loss_mean = 0, 0
        iterations = 0
        # Label vectors for loss function
        y_real, y_fake = (torch.ones(batch_size, 1), torch.zeros(batch_size, 1))
        if torch.cuda.is_available():
            y_real, y_fake = y_real.cuda(), y_fake.cuda()

        for face, landmarks in train_data_loader:
            z = torch.randn((batch_size, self.z_dim, 1, 1))
            # landmarks_gen = np.random.multivariate_normal(self.y_mean, self.y_cov,
            #                                      size=(batch_size))[:, :, None, None]
            # landmarks_gen = torch.from_numpy(landmarks_gen).type(torch.float32)
            if self.cuda:
                face, landmarks, z = face.cuda(), landmarks.cuda(), z.cuda()

            # ========== Training discriminator
            self.D_optimizer.zero_grad()

            # Train on real example with real features
            D_real = self.D(face, landmarks)
            D_real_loss = self.BCE_loss(D_real, y_real)  # real corresponds to log(D_real)

            # Train on real example with fake features
            D_fake_feature = self.D(face, landmarks)
            D_fake_feature_loss = self.BCE_loss(D_fake_feature, y_fake)

            # Train on fake example from generator
            x_fake = self.G(z, landmarks)  # landmarks_gen)
            D_fake = self.D(x_fake, landmarks)  # landmarks_gen)
            D_fake_loss = self.BCE_loss(D_fake, y_fake)  # face corresponds to log(1-D_fake)

            D_loss = D_real_loss + (D_fake_loss + D_fake_feature_loss) / 2

            D_loss.backward()
            self.D_optimizer.step()

            # ========== Training generator
            self.G_optimizer.zero_grad()
            # todo fix 2nd path
            # https://pytorch.org/docs/stable/autograd.html?highlight=backward#torch.Tensor.backward
            x_fake = self.G(z, landmarks)  # landmarks_gen)
            D_fake = self.D(x_fake, landmarks)  # landmarks_gen)
            G_loss = self.BCE_loss(D_fake, y_real)

            G_loss.backward()
            self.G_optimizer.step()

            # losses
            G_loss_mean += G_loss
            D_loss_mean += D_loss
            iterations += 1

        G_loss_mean /= iterations
        D_loss_mean /= iterations

        return G_loss_mean.cpu().data.numpy(), D_loss_mean.cpu().data.numpy(), x_fake

    def validate(self, validation_data_loader, batch_size, **kwargs):
        G_loss_mean, D_loss_mean = 0, 0
        iterations = 0

        # Label vectors for loss function
        y_real, y_fake = (torch.ones(batch_size, 1), torch.zeros(batch_size, 1))
        if torch.cuda.is_available():
            y_real, y_fake = y_real.cuda(), y_fake.cuda()

        for x, y in validation_data_loader:
            z = torch.randn((batch_size, self.z_dim, 1, 1))
            # y_gen = np.random.multivariate_normal(self.y_mean, self.y_cov,
            #                                      size=batch_size).astype('float32')
            # y_gen = torch.from_numpy(y_gen[:, :, None, None])
            if self.cuda:
                x, y, z = x.cuda(), y.cuda(), z.cuda()
                # y_gen = y_gen.cuda()

            # ========== Training discriminator
            # Train on real example from dataset
            D_real = self.D(x, y)
            D_real_loss = self.BCE_loss(D_real, y_real)

            x_fake = self.G(z, y)  # y_gen)
            D_fake = self.D(x_fake, y)  # y_gen)
            D_fake_loss = self.BCE_loss(D_fake, y_fake)

            D_loss = D_real_loss + D_fake_loss

            # ========== Training generator
            x_fake = self.G(z, y)  # y_gen)
            D_fake = self.D(x_fake, y)  # y_gen)
            G_loss = self.BCE_loss(D_fake, y_real)

            # losses
            G_loss_mean += G_loss
            D_loss_mean += D_loss
            iterations += 1

        G_loss_mean /= iterations
        D_loss_mean /= iterations

        return G_loss_mean.cpu().data.numpy(), D_loss_mean.cpu().data.numpy(), x_fake

    def log(self, logger, epoch, lossG, lossD, images, log_images=False):  # last parameter is not needed anymore
        """
        use logger to log current loss etc...
        :param logger: logger used to log
        :param epoch: current epoch
        """
        logger.log_loss(epoch=epoch, loss={'lossG': float(lossG), 'lossD': float(lossD)})
        logger.log_fps(epoch=epoch)
        logger.save_model(epoch)

        if log_images:
            self.log_images(logger, epoch, images, validation=False)

    def log_validation(self, logger, epoch, lossG, lossD, images):
        logger.log_loss(epoch=epoch, loss={'lossG_val': float(lossG), 'lossD_val': float(lossD)})

        self.log_images(logger, epoch, images, validation=True)

    def log_images(self, logger, epoch, images, validation=True):
        images = images.cpu()
        examples = int(len(images))
        example_indices = random.sample(range(0, examples - 1), 4 * 4)
        A = []
        for idx, i in enumerate(example_indices):
            A.append(images[i] * 255.00)
        tag = 'validation_output' if validation else 'training_output'
        logger.log_images(epoch, A, tag, 4)
