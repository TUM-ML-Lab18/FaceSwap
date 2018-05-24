import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from FaceAnonymizer.models.ModelUtils import weights_init
from Preprocessor.ImageDataset import StaticLandmarks32x32Dataset


class Generator(nn.Module):
    def __init__(self, input_dim=(100, 10), output_dim=(64, 64, 3), ngf=32):
        """
        Initializer for a Generator object
        :param input_dim: Size of the input vectors (latent space)
                          Tuple of integers - (N_random, N_feature)
        :param output_dim: Size of the output image
                           Tuple of integers - (W, H, C)
        :param ngf: Number of generator filters in the last conv layer
                           TODO: Investigate relation to image size => Maybe ngf==W ?
        """
        super(Generator, self).__init__()

        self.z_dim, self.y_dim = input_dim
        self.W_out, self.H_out, self.C_out = output_dim
        self.input_dim = self.z_dim + self.y_dim
        self.ngf = ngf
        self.ngpu = torch.cuda.device_count()

        # TODO: Maybe create Conv-Blocks
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.input_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, self.C_out, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

        self.apply(weights_init)

    def forward(self, z, y):
        """
        Calculates forward pass
        :param z: Random vector
        :param y: Feature vector
        :return: Tensor Image
        """
        x = torch.cat([z, y], 1)

        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            output = self.main(x)

        return output

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Load model with its parameters from the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Loading model... %s' % path)
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))


class Discriminator(nn.Module):
    def __init__(self, y_dim=10, input_dim=(64, 64, 3), ndf=32):
        """
        Initializer for a Discriminator object
        :param y_dim: Dimensionality of feature vector
        :param input_dim: Size of the input vectors
                          Tuple of integers - (W, H, C)
        :param ngf: Number of generator filters in the last conv layer
                           TODO: Investigate relation to image size => Maybe ngf==W ?
        """
        super(Discriminator, self).__init__()

        self.W_in, self.H_in, self.C_in = input_dim
        self.y_dim = y_dim
        self.input_dim = self.C_in + self.y_dim
        self.ndf = ndf
        self.ngpu = torch.cuda.device_count()

        self.conv = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.C_in, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # ========== Input feature vector TODO: More elegant solution
        self.main = nn.Sequential(
            # state size. (ndf + y_dim) x 32 x 32
            nn.Conv2d(ndf + self.y_dim, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.apply(weights_init)

    def forward(self, x, y):
        """
        Calculates forward pass
        :param x: Tensor image
        :param y: Feature vector
        :return: Scalar
        """
        # TODO: Make y_fill dynamic
        y_fill = y.repeat((1, 1, 32, 32))
        if x.is_cuda and self.ngpu > 1:
            x = nn.parallel.data_parallel(self.conv, x, range(self.ngpu))
            x = torch.cat([x, y_fill], 1)
            x = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            x = self.conv(x)
            x = torch.cat([x, y_fill], 1)
            x = self.main(x)

        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Load model with its parameters from the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Loading model... %s' % path)
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))


class CGAN(object):
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
        self.y_mean = StaticLandmarks32x32Dataset.y_mean.copy()
        self.y_cov = StaticLandmarks32x32Dataset.y_cov.copy()

        # Label vectors for loss function
        self.y_real, self.y_fake = (torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1))

        if torch.cuda.is_available():
            self.cuda = True
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss.cuda()
            self.y_real, self.y_fake = self.y_real.cuda(), self.y_fake.cuda()

    def set_train_mode(self, mode):
        if mode:
            self.G.train()
            self.D.train()
            torch.set_grad_enabled(True)
        else:
            self.G.eval()
            self.D.eval()
            torch.set_grad_enabled(False)

    def train(self, current_epoch, batches):
        G_loss_mean, D_loss_mean = 0, 0
        iterations = 0
        for x, y in batches:
            z = torch.randn((self.batch_size, self.z_dim, 1, 1))
            y_gen = np.random.multivariate_normal(self.y_mean, self.y_cov,
                                                  size=(self.batch_size))
            y_gen = torch.from_numpy(y_gen[:, :, None, None]).type(torch.FloatTensor)
            if self.cuda:
                x, y, y_gen, z = x.cuda(), y.cuda(), y_gen.cuda(), z.cuda()

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
            y_gen = np.random.multivariate_normal(self.y_mean, self.y_cov,
                                                  size=(self.batch_size))
            y_gen = torch.from_numpy(y_gen[:, :, None, None]).type(torch.FloatTensor)
            if self.cuda:
                x, y, y_gen, z = x.cuda(), y.cuda(), y_gen.cuda(), z.cuda()

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

    def log(self, logger, epoch, lossG, lossD, log_images=None): # last parameter is not needed anymore
        """
        use logger to log current loss etc...
        :param logger: logger used to log
        :param epoch: current epoch
        """
        logger.log_loss(epoch=epoch, loss={'lossG': float(lossG), 'lossD': float(lossD)})
        logger.log_fps(epoch=epoch)
        logger.save_model(epoch)

    def log_validate(self, logger, epoch, lossG, lossD, images):
        logger.log_loss(epoch=epoch, loss={'lossG_val': float(lossG), 'lossD_val': float(lossD)})

        images = images.cpu()
        examples = int(len(images))
        example_indices = random.sample(range(0, examples - 1), 4*4)
        A = []
        for idx, i in enumerate(example_indices):
            A.append(images[i] * 255.00)
        logger.log_images(epoch, A, "sample_output", 4)

    def save_model(self, path):
        # Create subfolder for models
        path = Path(path)
        subfolder = "model"
        path = path / subfolder
        path.mkdir(parents=True, exist_ok=True)
        self.G.save(path / 'generator.model')
        self.D.save(path / 'discriminator.model')

    def load_model(self, path):
        path = Path(path)
        self.G.load(path / 'generator.model')
        self.D.load(path / 'discriminator.model')
