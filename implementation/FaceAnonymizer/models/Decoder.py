import torch
import torch.nn as nn

from FaceAnonymizer.models.ModelUtils import UpscaleBlock


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        """
        Initialize a new decoder network.

        Inputs:
        - latent_dim: dimension of the latent space.
        """
        super(Decoder, self).__init__()
        self.upscale_1 = UpscaleBlock(latent_dim, 256)
        self.upscale_2 = UpscaleBlock(256, 128)
        self.upscale_3 = UpscaleBlock(128, 64)
        self.conv = nn.Conv2d(64, 3, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the encoder network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        x = self.upscale_1(x)
        x = self.upscale_2(x)
        x = self.upscale_3(x)
        x = self.conv(x)
        x = self.sigmoid(x)

        return x

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
