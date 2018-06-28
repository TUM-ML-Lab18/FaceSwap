import torch

from torch.nn import Module


class AutoEncoder(Module):
    """
    This class is just a wrapper around a encoder and decoder
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ngpu = torch.cuda.device_count()

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            latent = torch.nn.parallel.data_parallel(self.encoder, x, range(self.ngpu))
            decoded = torch.nn.parallel.data_parallel(self.decoder, latent, range(self.ngpu))
        else:
            latent = self.encoder(x)
            decoded = self.decoder(latent)
        return decoded
