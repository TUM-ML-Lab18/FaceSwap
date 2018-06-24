import torch
from torch import nn as nn

from Models.ModelUtils.ModelUtils import View, UpscaleBlockBlock, CustomModule


class LatentDecoder(CustomModule):
    """
    It's the same decoder used in the deepfakes repository
    """
    def __init__(self, input_dim):
        super().__init__()
        self.ngpu = torch.cuda.device_count()
        self.sequ = nn.Sequential(nn.Linear(input_dim, 1024),
                                  nn.Linear(1024, 4 * 4 * 1024),
                                  View(-1, 1024, 4, 4),
                                  UpscaleBlockBlock(1024, 512, 5),
                                  nn.Conv2d(32, 3, kernel_size=5, padding=2),
                                  nn.Tanh())

    def forward(self, x):
        """
        Forward pass of the encoder network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        if x.is_cuda and self.ngpu > 1:
            x = nn.parallel.data_parallel(self.sequ, x, range(self.ngpu))
        else:
            x = self.sequ(x)
        return x


class LatentReducedDecoder(CustomModule):
    def __init__(self, input_dim):
        super().__init__()
        self.sequ = nn.Sequential(  # nn.Dropout(p=0.5),
            nn.Linear(input_dim, 512),
            nn.Linear(512, 2 * 2 * 512),
            View(-1, 512, 2, 2),
            UpscaleBlockBlock(512, 256, 5),
            nn.Conv2d(16, 3, kernel_size=5, padding=2),
            nn.Sigmoid())

    def forward(self, x):
        """
        Forward pass of the encoder network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        return self.sequ(x)
