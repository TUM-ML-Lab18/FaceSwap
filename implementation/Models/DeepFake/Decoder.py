import torch
import torch.nn as nn

from Models.ModelUtils.ModelUtils import UpscaleBlockBlock, CustomModule


class Decoder(CustomModule):
    def __init__(self, input_dim, num_convblocks=3):
        """
        Initialize a new decoder network.

        Inputs:
        - latent_dim: dimension of the latent space.
        """
        super(Decoder, self).__init__()
        self.upscale = UpscaleBlockBlock(input_dim, 256, num_convblocks)
        resulting_channels = 256 // (2 ** (num_convblocks - 1))
        self.conv = nn.Conv2d(resulting_channels, 3, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the encoder network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        x = self.upscale(x)
        x = self.conv(x)
        x = self.sigmoid(x)

        return x
