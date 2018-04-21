from torch.nn import Module


class Encoder(Module):

    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class Decoder(Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self):
        latent_space = self.encoder()
        pass


class Autoencoder():
    def __init__(self, Encoder, Decoder):
        self.decoder = Decoder()
        self.encoder1 = Encoder(self.decoder)
        self.encoder2 = Encoder(self.decoder)
