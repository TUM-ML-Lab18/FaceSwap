#
# This file includes the main functionality of the FaceAnonymizer module
# Author: Alexander Becker
#
import datetime

import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam

from FaceAnonymizer.models.Decoder import Decoder
from FaceAnonymizer.models.Encoder import Encoder
from FaceAnonymizer.models.Autoencoder import AutoEncoder


class Anonymizer:

    def __init__(self, data1, data2, batch_size=64, epochs=500, learning_rate=1e-4):
        """
        Initialize a new Anonymizer.

        Inputs:
        - data1: dataset of pictures of first face
        - data2: dataset of pictures of second face
        - batch_size: batch size
        - epochs: number of training epochs
        - learning_rate: learning rate
        """
        self.encoder = Encoder((3, 64, 64), 1024).cuda()
        self.decoder1 = Decoder(512).cuda()
        self.decoder2 = Decoder(512).cuda()

        self.autoencoder1 = AutoEncoder(self.encoder, self.decoder1).cuda()
        self.autoencoder2 = AutoEncoder(self.encoder, self.decoder2).cuda()

        self.lossfn = torch.nn.L1Loss(size_average=True).cuda()
        self.dataLoader1 = DataLoader(data1, batch_size, shuffle=True, num_workers=4, drop_last=True)
        self.dataLoader2 = DataLoader(data2, batch_size, shuffle=True, num_workers=4, drop_last=True)
        self.epochs = epochs

    def train(self):
        writer = SummaryWriter("./logs/" + str(datetime.datetime.now()))

        optimizer1 = Adam(self.autoencoder1.parameters())
        optimizer2 = Adam(self.autoencoder2.parameters())

        for i_epoch in range(self.epochs):
            loss1, loss2 = 0, 0
            for face1, face2 in zip(self.dataLoader1, self.dataLoader2):
                # face1 and face2 contain a batch of images of the first and second face, respectively
                face1, face2 = Variable(face1).cuda(), Variable(face2).cuda()
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
            writer.add_scalar("loss/A", loss1, i_epoch)
            writer.add_scalar("loss/B", loss2, i_epoch)
            log_images_histograms(self.autoencoder1, writer, i_epoch)
            print(f"[Epoch {i_epoch}] loss1: {loss1}, loss2: {loss2}", end='\n')

    def anonymize(self, x):
        return self.autoencoder2(x)

    # TODO: Use save & load functions from models -> memory independent (RAM vs GPU)
    def save_model(self, path):
        data = {'ae1': self.autoencoder1.state_dict(), 'ae2': self.autoencoder2.state_dict()}
        torch.save(data, path)

    def load_model(self, path):
        data = torch.load(path)
        self.autoencoder1.load_state_dict(data['ae1'])
        self.autoencoder2.load_state_dict(data['ae2'])


def log_images_histograms(net, writer, frame_idx):
    q = vutils.make_grid(torch.cat(torch.split(next(net.parameters()).data.cpu(), 1, 1), 0), normalize=True,
                         scale_each=True)
    writer.add_image('conv_layers/encoder', q, frame_idx)
    for name, param in net.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), frame_idx)
