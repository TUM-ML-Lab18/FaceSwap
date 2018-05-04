import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from FaceAnonymizer.models.ModelUtils import ConvBlock, UpscaleBlock, Flatten, View, ConvBlockBlock


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_convblocks=4):
        """
        Initialize a new encoder network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - latent_dim: Int giving the size of the latent space.
        """
        super(Encoder, self).__init__()
        C, H, W = input_dim
        self.conv = ConvBlockBlock(C, 128, num_convblocks)
        self.flat = Flatten()
        self.fc_1 = nn.Linear(self._get_conv_out(input_dim),
                              out_features=latent_dim)
        self.fc_2 = nn.Linear(in_features=latent_dim,
                              out_features=4 * 4 * 1024)
        self.view = View(-1, 1024, 4, 4)
        self.upscale = UpscaleBlock(1024, 512)

    def _get_conv_out(self, shape):
        """
        Calculate output size of the conv-layers

        Inputs:
        - shape: Shape of sample input for the network.
        """
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        Forward pass of the encoder network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        x = self.conv(x)
        x = self.flat(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.view(x)
        x = self.upscale(x)

        return x

    # TODO: Maybe a superclass only with those common methods
    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

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


class SebisEncoder(Encoder):
    def __init__(self, input_dim, latent_dim):
        super(SebisEncoder, self).__init__(input_dim, latent_dim)
