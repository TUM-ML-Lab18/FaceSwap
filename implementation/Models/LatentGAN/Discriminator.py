from torch import nn as nn
import Models.CGAN.Discriminator


class Discriminator(Models.CGAN.Discriminator):

    def __init__(self, input_dim=(64, 64, 3), ndf=64):
        super(Discriminator, self).__init__(y_dim=0, input_dim=input_dim, ndf=ndf)

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