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
        """
        Initialize a new Anonymizer.

        Inputs:
        - data1: dataset of pictures of first face
        - data2: dataset of pictures of second face
        - batch_size: batch size
        - epochs: number of training epochs
        - learning_rate: learning rate
        """
        self.encoder = Encoder()
        self.decoder1 = Decoder()
        self.decoder2 = Decoder()

        self.autoencoder1 = Autoencoder(self.encoder, self.decoder1)
        self.autoencoder2 = Autoencoder(self.encoder, self.decoder2)

        self.lossfn = torch.nn.MSELoss(size_average=False)
        self.dataLoader1 = DataLoader(data1, batch_size, shuffle=True, num_workers=4)
        self.dataLoader2 = DataLoader(data2, batch_size, shuffle=True, num_workers=4)
        self.epochs = epochs

    def train(self):
        optimizer1 = Adam(self.autoencoder1.parameters())
        optimizer2 = Adam(self.autoencoder2.parameters())

        for i_epoch in range(self.epochs):
            loss1, loss2 = 0, 0
            for face1, face2 in zip(self.dataLoader1, self.dataLoader2):
                # face1 and face2 contain a batch of images of the first and second face, respectively
                face1, face2 = Variable(face1), Variable(face2)
                optimizer1.zero_grad()
                optimizer2.zero_grad()

                output1 = self.autoencoder1(face1)
                loss1 = self.lossfn(output1, face1)
                loss1.backward()
                optimizer1.step()

                output2 = self.autoencoder2(face2)
                loss2 = self.lossfn(output2, face2)
                loss2.backward()
                optimizer2.step()

            print("[Epoch {0}] loss1: {2:.5f}, loss2: {3:.5f}".format(i_epoch, loss1, loss2), end='\n')

    def anonymize(self, x):
        return self.autoencoder2(x)

    def save_model(self, path):
        data = {'ae1': self.autoencoder1.state_dict(), 'ae2': self.autoencoder2.state_dict()}
        torch.save(data, path)

    def load_model(self, path):
        data = torch.load(path)
        self.autoencoder1.load_state_dict(data['ae1'])
        self.autoencoder2.load_state_dict(data['ae2'])
