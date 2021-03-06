import torch
from torch import nn as nn

from Models.ModelUtils.ModelUtils import CustomModule


class Discriminator(CustomModule):
    def __init__(self, y_dim=10, input_dim=(64, 64, 3), ndf=64):
        """
        Initializer for a Discriminator object
        :param y_dim: Dimensionality of feature vector
        :param input_dim: Size of the input vectors
                          Tuple of integers - (W, H, C)
        :param ndf: Number of generator filters in the last conv layer
        """
        super(Discriminator, self).__init__()

        self.W_in, self.H_in, self.C_in = input_dim
        self.y_dim = y_dim
        self.input_dim = self.C_in + self.y_dim
        self.ndf = ndf
        self.ngpu = torch.cuda.device_count()
        self.input_layer_dim = 32

        # ========== Input image
        self.conv = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.C_in, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # ========== Input self.conv output concatenated with feature vector
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

        self.apply(self.weights_init)

    def forward(self, x, y):
        """
        Calculates forward pass
        :param x: Tensor image
        :param y: Feature vector
        :return: Scalar
        """
        bs = x.shape[0]
        y_fill = y.view((bs, -1, 1, 1)).repeat((1, 1, self.input_layer_dim, self.input_layer_dim))
        if x.is_cuda and self.ngpu > 1:
            x = nn.parallel.data_parallel(self.conv, x, range(self.ngpu))
            x = torch.cat([x, y_fill], 1)
            x = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            x = self.conv(x)
            x = torch.cat([x, y_fill], 1)
            x = self.main(x)
        return x
