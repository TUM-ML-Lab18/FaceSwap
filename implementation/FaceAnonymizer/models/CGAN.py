import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from FaceAnonymizer.models.ModelUtils import initialize_weights

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.input_height = 32
        self.input_width = 32
        self.input_dim = 62 + 10 # 62 random + 5 landmarks (x,y)
        self.output_dim = 3 # channel RGB

        # TODO: Create LinearBlock
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )

        initialize_weights(self)

    def forward(self, input, label):
        # Batchsize of 1 works not
        x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
        x = self.deconv(x)

        return x

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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.input_height = 32
        self.input_width = 32
        self.input_dim = 3 + 10 # RGB channel + 5 landmarks (x,y)
        self.output_dim = 1 # not always 1 => TRUE/FALSE?

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )

        initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.conv(x)
        x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        x = self.fc(x)

        return x

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


class CGAN(object):
    def __init__(self):
        seed = 547
        np.random.seed(seed)
        self.cuda = False
        self.batch_size = 64
        self.y_dim = 10 # 5 landmarks (x,y)
        self.z_dim = 62 # random vector
        self.input_height = 32
        self.input_width = 32

        self.G = Generator()
        self.D = Discriminator()

        beta1, beta2 = 0.5, 0.999
        lrG, lrD = 0.0002, 0.0002
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=lrG, betas=(beta1, beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=lrD, betas=(beta1, beta2))

        self.BCE_loss = nn.BCELoss()

        # load dataset
        self.data_X = np.load('/nfs/students/summer-term-2018/project_2/data/CelebA_new/data32.npy')
        self.data_Y = np.load('/nfs/students/summer-term-2018/project_2/data/CelebA_new/labels5.npy')
        self.data_X = torch.from_numpy(self.data_X).type(torch.FloatTensor)
        self.data_Y = torch.from_numpy(self.data_Y).type(torch.FloatTensor)
        np.random.shuffle(self.data_X)
        np.random.shuffle(self.data_Y)

        if torch.cuda.is_available():
            self.cuda = True
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss.cuda()

    def train(self):
        epochs = 50
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.G.train()
        self.D.train()

        # Label vectors for loss function
        y_real, y_fake = (torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1))
        if self.cuda:
            y_real, y_fake = y_real.cuda(), y_fake.cuda()

        for epoch in range(epochs):
            print('Epoch', epoch)
            for i_batch in range(len(self.data_X) // self.batch_size):
                x = self.data_X[i_batch*self.batch_size:(i_batch+1)*self.batch_size]
                y = self.data_Y[i_batch*self.batch_size:(i_batch+1)*self.batch_size]
                # TODO: Check H / W order or use just quadratic
                y_fill = y[:,:,None,None].repeat((1,1,self.input_width,self.input_height))
                z = torch.rand((self.batch_size, self.z_dim))
                # TODO: Check functionality without variable -> Pytorch 0.4?
                if self.cuda:
                    x ,y ,y_fill, z = x.cuda(), y.cuda(), y_fill.cuda(), z.cuda()

                # ========== Training discriminator
                self.D_optimizer.zero_grad()

                # Train on real example from dataset
                D_real = self.D(x, y_fill)
                D_real_loss = self.BCE_loss(D_real, y_real)

                # TODO: sample imaginary label vectors
                # Train on fake example from generator
                x_fake = self.G(z, y)
                D_fake = self.D(x_fake, y_fill)
                D_fake_loss = self.BCE_loss(D_fake, y_fake)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.data[0]) #TODO: deprecated index access

                D_loss.backward()
                self.D_optimizer.step()

                # ========== Training generator
                self.G_optimizer.zero_grad()

                # TODO: Check if reusable from generator training
                x_fake = self.G(z, y)
                D_fake = self.D(x_fake, y_fill)
                G_loss = self.BCE_loss(D_fake, y_real)
                self.train_hist['G_loss'].append(G_loss.data[0])

                G_loss.backward()
                self.G_optimizer.step()

                if ((i_batch + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (i_batch + 1), len(self.data_X) // self.batch_size, D_loss.data[0], G_loss.data[0]))

        self.G.save('/home/stromaxi/G.model')
        self.D.save('/home/stromaxi/D.model')


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train()