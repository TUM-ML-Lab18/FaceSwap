from abc import abstractmethod
from pathlib import Path

import torch.nn as nn
import torch


class CustomModule(nn.Module):
    @abstractmethod
    def forward(self, *input):
        pass

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

    @staticmethod
    def weights_init(m):
        """
        TODO
        :return:
        """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        pass


class CombinedModels:
    @abstractmethod
    def get_models(self):
        pass

    @abstractmethod
    def get_model_names(self):
        pass

    @abstractmethod
    def train(self, current_epoch, train_data_loader):
        pass

    @abstractmethod
    def validate(self, validation_data_loader):
        pass

    @abstractmethod
    def log(self, *info):
        pass

    @abstractmethod
    def log_validation(self, *info):
        pass

    @abstractmethod
    def __str__(self):
        """
        TODO
        :return:
        """
        # TODO return models
        s = str()
        for model in self.get_models():
            s += str(model) + '\n'
        return s

    def set_train_mode(self, mode):
        """
        TODO
        :param mode:
        :return:
        """
        for model in self.get_models():
            model.train(mode)
        torch.set_grad_enabled(mode)

    def save_model(self, path):
        """
        TODO
        :param path:
        :return:
        """
        path = Path(path)
        path = path / 'model'
        path.mkdir(parents=True, exist_ok=True)
        for name, model in zip(self.get_model_names(), self.get_models()):
            model.save(path / (name + '.model'))

    def load_model(self, path):
        """
        TODO
        :param path:
        :return:
        """
        path = Path(path)
        for name, model in zip(self.get_model_names(), self.get_models()):
            model.load(path / (name + '.model'))


class ConvBlock(nn.Module):
    """Convolution followed by a LeakyReLU"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 stride=2):
        """
        Initialize a ConvBlock.

        Inputs:
        - in_channels: Number of channels of the input
        - out_channels: Number of filters
        - kernel_size: Size of a convolution filter
        - stride: Stride of the convolutions
        """
        super(ConvBlock, self).__init__()
        # spatial size preserving padding: Padding = (Filter-1)/2
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(kernel_size - 1) // 2)
        self.leaky = nn.LeakyReLU(negative_slope=0.1,
                                  inplace=True)

    def forward(self, x):
        """
        Forward pass of the ConvBlock. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        x = self.conv(x)
        x = self.leaky(x)

        return x


class UpscaleBlock(nn.Module):
    """Scales image up"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1):
        """
        Initialize a UpscaleBlock.

        Inputs:
        - in_channels: Number of channels of the input
        - out_channels: Number of filters
        - kernel_size: Size of a convolution filter
        - stride: Stride of the convolutions
        """
        super(UpscaleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels * 4,  # compensate PixelShuffle dimensionality reduction
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(kernel_size - 1) // 2)
        self.leaky = nn.LeakyReLU(negative_slope=0.1,
                                  inplace=True)
        # TODO: Compare pixelshuffle from FaceSwap to the one from PyTorch
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        """
        Forward pass of the UpscaleBlock. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        x = self.conv(x)
        x = self.leaky(x)
        x = self.pixel_shuffle(x)
        return x


class Flatten(nn.Module):
    """Flatten images"""

    def forward(self, input):
        return input.view(input.size(0), -1)


class View(nn.Module):
    """
    Reshape tensor
    https://discuss.pytorch.org/t/equivalent-of-np-reshape-in-pytorch/144/5
    """

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)


class ConvBlockBlock(nn.Sequential):
    def __init__(self, channels_in, num_channels_first_layer=128, depth=4):
        block_list = [ConvBlock(channels_in, num_channels_first_layer)]
        for i in range(1, depth):
            block_list.append(
                ConvBlock(num_channels_first_layer * (2 ** (i - 1)), num_channels_first_layer * (2 ** i)))
        super().__init__(*block_list)


class UpscaleBlockBlock(nn.Sequential):
    def __init__(self, channels_in, num_channels_first_layer=256, depth=3):
        block_list = [UpscaleBlock(channels_in, num_channels_first_layer)]
        for i in range(1, depth):
            block_list.append(
                UpscaleBlock(num_channels_first_layer // (2 ** (i - 1)), num_channels_first_layer // (2 ** i)))
        super().__init__(*block_list)
