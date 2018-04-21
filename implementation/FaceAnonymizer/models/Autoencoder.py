from torch.nn import Module


class Autoencoder(Module):

    def __init__(self, encoder, decoder1, decoder2):
        super().__init__()
        self.encoder = encoder
        self.decoder1 = decoder1
        self.decoder2 = decoder2

    def forward(self, x):
        latent = self.encoder.forward(x)
        if use_decoder1: #TODO: distinguish
            return self.decoder1.forward(latent)
        else:
            return self.decoder2.forward(latent)