import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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
        string += self.G_optimizer + '\n'
        string += self.D_optimizer + '\n'
        string += self.BCE_loss + '\n'
        return string

    def __init__(self, batch_size=64, y_dim=10, z_dim=100):

        self.z_dim = z_dim  # 5 landmarks (x,y)
        self.y_dim = y_dim  # random vector
        self.batch_size = batch_size
        if torch.cuda.device_count() > 1:
            self.batch_size *= torch.cuda.device_count()
        seed = 547
        np.random.seed(seed)
        self.cuda = False

        self.G = Generator((self.z_dim, self.y_dim))
        self.D = Discriminator(self.y_dim)

        beta1, beta2 = 0.5, 0.999
        lrG, lrD = 0.0002, 0.0001
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=lrG, betas=(beta1, beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=lrD, betas=(beta1, beta2))

        self.BCE_loss = nn.BCELoss()

        # gaussian distribution of our landmarks
        # todo fix this shit
        # self.y_mean = StaticLandmarks32x32Dataset.y_mean.copy()
        # self.y_cov = StaticLandmarks32x32Dataset.y_cov.copy()

        # Label vectors for loss function
        self.y_real, self.y_fake = (torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1))

        if torch.cuda.is_available():
            self.cuda = True
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss.cuda()
            self.y_real, self.y_fake = self.y_real.cuda(), self.y_fake.cuda()

    def train(self, current_epoch, batches):
        G_loss_mean, D_loss_mean = 0, 0
        iterations = 0
        for x, y in batches:
            z = torch.randn((self.batch_size, self.z_dim, 1, 1))
            # y_gen = np.random.multivariate_normal(self.y_mean, self.y_cov,
            #                                      size=(self.batch_size))
            # y_gen = torch.from_numpy(y_gen[:, :, None, None]).type(torch.FloatTensor)
            if self.cuda:
                x, y, z = x.cuda(), y.cuda(), z.cuda()

            # ========== Training discriminator
            self.D_optimizer.zero_grad()

            # Train on real example from dataset
            D_real = self.D(x, y)
            D_real_loss = self.BCE_loss(D_real, self.y_real)

            # Train on fake example from generator
            # TODO: UserWarning: Using a target size (torch.Size([64, 1])) that is different
            # to the input size (torch.Size([64, 1, 1, 1])) is deprecated. Please ensure they have the same size.
            #   "Please ensure they have the same size.".format(target.size(), input.size()))
            x_fake = self.G(z, y)  # y_gen)
            D_fake = self.D(x_fake, y)  # y_gen)
            D_fake_loss = self.BCE_loss(D_fake, self.y_fake)

            D_loss = D_real_loss + D_fake_loss

            D_loss.backward()
            self.D_optimizer.step()

            # ========== Training generator
            self.G_optimizer.zero_grad()

            # TODO: Check if reusable from generator training
            x_fake = self.G(z, y)  # y_gen)
            D_fake = self.D(x_fake, y)  # y_gen)
            G_loss = self.BCE_loss(D_fake, self.y_real)

            G_loss.backward()
            self.G_optimizer.step()

            # losses
            G_loss_mean += G_loss
            D_loss_mean += D_loss
            iterations += 1

        G_loss_mean /= iterations
        D_loss_mean /= iterations

        return G_loss_mean.cpu().data.numpy(), D_loss_mean.cpu().data.numpy()

    def validate(self, batches):
        G_loss_mean, D_loss_mean = 0, 0
        iterations = 0
        for x, y in batches:
            z = torch.randn((self.batch_size, self.z_dim, 1, 1))
            # y_gen = np.random.multivariate_normal(self.y_mean, self.y_cov,
            #                                      size=(self.batch_size))
            # y_gen = torch.from_numpy(y_gen[:, :, None, None]).type(torch.FloatTensor)
            if self.cuda:
                x, y, z = x.cuda(), y.cuda(), z.cuda()

            # ========== Training discriminator
            # Train on real example from dataset
            D_real = self.D(x, y)
            D_real_loss = self.BCE_loss(D_real, self.y_real)

            x_fake = self.G(z, y)  # y_gen)
            D_fake = self.D(x_fake, y)  # y_gen)
            D_fake_loss = self.BCE_loss(D_fake, self.y_fake)

            D_loss = D_real_loss + D_fake_loss

            # ========== Training generator
            x_fake = self.G(z, y)  # y_gen)
            D_fake = self.D(x_fake, y)  # y_gen)
            G_loss = self.BCE_loss(D_fake, self.y_real)

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
