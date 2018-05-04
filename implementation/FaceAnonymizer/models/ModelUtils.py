import torch.nn as nn


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
