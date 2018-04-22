#
# This file includes the main functionality of the FaceAnonymizer module
# Author: Alexander Becker
#

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam
from .models import Autoencoder, Encoder, Decoder


class Anonymizer:

    def __init__(self, data1, data2, batch_size=50, epochs=500, learning_rate=1e-4):
        self.autoencoder = Autoencoder(Encoder(), Decoder(), Decoder())
        self.lossfn = torch.nn.MSELoss(size_average=False)
        self.dataLoader1 = DataLoader(data1, batch_size, shuffle=True, num_workers=4)
        self.dataLoader2 = DataLoader(data2, batch_size, shuffle=True, num_workers=4)
        self.epochs = epochs

    def train(self):
        optimizer = Adam(self.autoencoder.parameters())

        for i_epoch in range(self.epochs):
            for face1, face2 in zip(self.dataLoader1, self.dataLoader2):
                # face1 and face2 contain a batch of images of the first and second face, respectively

                face1, face2 = Variable(face1), Variable(face2)

                output = self.autoencoder(face1)
                loss = self.lossfn(output, face1)
                loss.backward()
                optimizer.step()

                #TODO: Do this for face2 and distinguish them in Autoencoder



    def optimize(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass
