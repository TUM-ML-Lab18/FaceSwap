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
    def get_models(self):
        return [self.G, self.D]

    def get_model_names(self):
        return ['generator', 'discriminator']

    def __str__(self):
        string = super().__str__()
        string += str(self.G_optimizer) + '\n'
        string += str(self.D_optimizer) + '\n'
        string += str(self.BCE_loss) + '\n'
        return string

    def __init__(self, **kwargs):
        self.z_dim = kwargs.get('z_dim', 100)
        self.y_dim = kwargs.get('y_dim', 10)
        path_to_y_mean = kwargs.get('y_mean', ARRAY_CELEBA_LANDMARKS_MEAN)
        path_to_y_cov = kwargs.get('y_cov', ARRAY_CELEBA_LANDMARKS_COV)
        lrG = kwargs.get('lrG', 0.0002)
        lrD = kwargs.get('lrD', 0.0002)

        self.G = Generator((self.z_dim, self.y_dim))
        self.D = Discriminator(self.y_dim)

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

        for x, y in train_data_loader:
            z = torch.randn((batch_size, self.z_dim, 1, 1))
            y_gen = np.random.multivariate_normal(self.y_mean, self.y_cov,
                                                  size=(batch_size))[:, :, None, None]
            y_gen = torch.from_numpy(y_gen).type(torch.float32)
            if self.cuda:
                x, y, y_gen, z = x.cuda(), y.cuda(), y_gen.cuda(), z.cuda()

            # ========== Training discriminator
            self.D_optimizer.zero_grad()

            # Train on real example with real features
            D_real = self.D(x, y)
            D_real_loss = self.BCE_loss(D_real, y_real)

            # Train on real example with fake features
            D_fake_feature = self.D(x, y_gen)
            D_fake_feature_loss = self.BCE_loss(D_fake_feature, y_fake)

            # Train on fake example from generator
            x_fake = self.G(z, y_gen)
            D_fake = self.D(x_fake, y_gen)
            D_fake_loss = self.BCE_loss(D_fake, y_fake)

            D_loss = D_real_loss + D_fake_loss + D_fake_feature_loss

            D_loss.backward()
            self.D_optimizer.step()

            # ========== Training generator
            self.G_optimizer.zero_grad()

            # TODO: Try 'retrain_variables=True' in D_loss.backward()
            x_fake = self.G(z, y_gen)
            D_fake = self.D(x_fake, y_gen)
            G_loss = self.BCE_loss(D_fake, y_real)

            G_loss.backward()
            self.G_optimizer.step()

            # losses
            G_loss_mean += G_loss
            D_loss_mean += D_loss
            iterations += 1

        G_loss_mean /= iterations
        D_loss_mean /= iterations

        return G_loss_mean.cpu().data.numpy(), D_loss_mean.cpu().data.numpy()

    def validate(self, validation_data_loader, batch_size, **kwargs):
        G_loss_mean, D_loss_mean = 0, 0
        iterations = 0

        # Label vectors for loss function
        y_real, y_fake = (torch.ones(batch_size, 1), torch.zeros(batch_size, 1))
        if torch.cuda.is_available():
            y_real, y_fake = y_real.cuda(), y_fake.cuda()

        for x, y in validation_data_loader:
            z = torch.randn((batch_size, self.z_dim, 1, 1))
            y_gen = np.random.multivariate_normal(self.y_mean, self.y_cov,
                                                  size=(batch_size))
            y_gen = torch.from_numpy(y_gen[:, :, None, None]).type(torch.float)
            if self.cuda:
                x, y, y_gen, z = x.cuda(), y.cuda(), y_gen.cuda(), z.cuda()

            # ========== Training discriminator
            # Train on real example from dataset
            D_real = self.D(x, y)
            D_real_loss = self.BCE_loss(D_real, y_real)

            x_fake = self.G(z, y_gen)
            D_fake = self.D(x_fake, y_gen)
            D_fake_loss = self.BCE_loss(D_fake, y_fake)

            D_loss = D_real_loss + D_fake_loss

            # ========== Training generator
            x_fake = self.G(z, y_gen)
            D_fake = self.D(x_fake, y_gen)
            G_loss = self.BCE_loss(D_fake, y_real)

            # losses
            G_loss_mean += G_loss
            D_loss_mean += D_loss
            iterations += 1

        G_loss_mean /= iterations
        D_loss_mean /= iterations

        return G_loss_mean.cpu().data.numpy(), D_loss_mean.cpu().data.numpy(), x_fake

    def log(self, logger, epoch, lossG, lossD, log_images=None):  # last parameter is not needed anymore
        """
        use logger to log current loss etc...
        :param logger: logger used to log
        :param epoch: current epoch
        """
        logger.log_loss(epoch=epoch, loss={'lossG': float(lossG), 'lossD': float(lossD)})
        logger.log_fps(epoch=epoch)
        logger.save_model(epoch)

    def log_validation(self, logger, epoch, lossG, lossD, images):
        logger.log_loss(epoch=epoch, loss={'lossG_val': float(lossG), 'lossD_val': float(lossD)})

        images = images.cpu()
        examples = int(len(images))
        example_indices = random.sample(range(0, examples - 1), 4 * 4)
        A = []
        for idx, i in enumerate(example_indices):
            A.append(images[i] * 255.00)
        logger.log_images(epoch, A, "sample_output", 4)
