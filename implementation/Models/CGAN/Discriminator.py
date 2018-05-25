import torch
from torch import nn as nn

from Models.ModelUtils.ModelUtils import weights_init, CustomModule


class Discriminator(CustomModule):
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