import torch
from torch import nn as nn

from Models.ModelUtils.ModelUtils import CustomModule, View, UpscaleBlockBlock


class Generator(CustomModule):
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

        self.apply(self.weights_init)

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


class LatentDecoderGAN(CustomModule):
    def __init__(self, input_dim):
        super().__init__()
        self.ngpu = torch.cuda.device_count()
        self.sequ = nn.Sequential(nn.Linear(input_dim,
                                            1024),
                                  nn.Linear(1024, 4 * 4 * 1024),
                                  View(-1, 1024, 4, 4),
                                  UpscaleBlockBlock(1024, 512, 4),
                                  nn.Conv2d(64, 3, kernel_size=5, padding=2),
                                  nn.Sigmoid())

    def forward(self, z, y):
        x = torch.cat([z, y], 1).squeeze(-1).squeeze(-1)
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.sequ, x, range(self.ngpu))
        else:
            output = self.sequ(x)
        return output