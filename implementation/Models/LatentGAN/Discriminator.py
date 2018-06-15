from torch import nn as nn
from Models.CGAN.Discriminator import Discriminator as CGANDiscriminator


class Discriminator(CGANDiscriminator):

    def __init__(self, input_dim, ndf):
        super().__init__(y_dim=0, input_dim=input_dim, ndf=ndf)
        self.conv = nn.Sequential(
            nn.Conv2d(self.C_in, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # ========== Input feature vector
        self.main = nn.Sequential(
            # state size. (ndf + y_dim) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
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

    def forward(self, x):
        """
        Calculates forward pass
        :param x: Tensor image
        :param y: Feature vector
        :return: Scalar
        """
        if x.is_cuda and self.ngpu > 1:
            x = nn.parallel.data_parallel(self.conv, x, range(self.ngpu))
            x = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            x = self.conv(x)
            x = self.main(x)

        return x
