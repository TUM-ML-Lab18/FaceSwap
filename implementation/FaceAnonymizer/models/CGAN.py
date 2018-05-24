import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from implementation.FaceAnonymizer.models.ModelUtils import weights_init, save_model_dict, load_model_dict

class Generator(nn.Module):
    def __init__(self, input_dim=(100,10), output_dim=(64,64,3), ngf=32, ngpu=1):
        """
        Initializer for a Generator object
        :param input_dim: Size of the input vectors (latent space)
                          Tuple of integers - (N_random, N_feature)
        :param output_dim: Size of the output image
                           Tuple of integers - (W, H, C)
        :param ngf: Number of generator filters in the last conv layer
                           TODO: Investigate relation to image size => Maybe ngf==W ?
        :param ngpu: Number of GPUs to use
        """
        super(Generator, self).__init__()

        self.z_dim, self.y_dim = input_dim
        self.W_out, self.H_out, self.C_out = output_dim
        self.input_dim = self.z_dim + self.y_dim
        self.ngf = ngf
        self.ngpu = ngpu

        # TODO: Maybe create Conv-Blocks
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.input_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, self.C_out, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

        self.apply(weights_init)

    def forward(self, z, y):
        """
        Calculates forward pass
        :param z: Random vector
        :param y: Feature vector
        :return: Tensor Image
        """
        x = torch.cat([z, y], 1)

        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            output = self.main(x)

        return output


class Discriminator(nn.Module):
    def __init__(self, y_dim=10, input_dim=(64,64,3), ndf=32, ngpu=1):
        """
        Initializer for a Discriminator object
        :param y_dim: Dimensionality of feature vector
        :param input_dim: Size of the input vectors
                          Tuple of integers - (W, H, C)
        :param ngf: Number of generator filters in the last conv layer
                           TODO: Investigate relation to image size => Maybe ngf==W ?
        :param ngpu: Number of GPUs to use
        """
        super(Discriminator, self).__init__()

        self.W_in, self.H_in, self.C_in = input_dim
        self.y_dim = y_dim
        self.input_dim = self.C_in  + self.y_dim
        self.ndf = ndf
        self.ngpu = ngpu

        self.conv = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.C_in, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # ========== Input feature vector TODO: More elegant solution
        self.main = nn.Sequential(
            # state size. (ndf + y_dim) x 32 x 32
            nn.Conv2d(ndf + self.y_dim, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.apply(weights_init)


    def forward(self, x, y):
        """
        Calculates forward pass
        :param x: Tensor image
        :param y: Feature vector
        :return: Scalar
        """
        # TODO: Make y_fill dynamic
        y_fill = y.repeat((1, 1, 32, 32))
        if x.is_cuda and self.ngpu > 1:
            x = nn.parallel.data_parallel(self.conv, x, range(self.ngpu))
            x = torch.cat([x, y_fill], 1)
            x = nn.parallel.data_parallel(self.conv, x, range(self.ngpu))
        else:
            x = self.conv(x)
            x = torch.cat([x, y_fill], 1)
            x = self.main(x)

        return x


class CGAN(object):
    def __init__(self):
        seed = 547
        np.random.seed(seed)
        self.cuda = False
        self.batch_size = 64
        self.y_dim = 10 # 5 landmarks (x,y)
        self.z_dim = 100 # random vector
        self.input_height = 64
        self.input_width = 64

        self.G = Generator((self.z_dim, self.y_dim), ngpu=4)
        self.D = Discriminator(self.y_dim)

        beta1, beta2 = 0.5, 0.999
        lrG, lrD = 0.0002, 0.0001
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=lrG, betas=(beta1, beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=lrD, betas=(beta1, beta2))

        self.BCE_loss = nn.BCELoss()

        # load dataset
        self.data_X = np.load('/nfs/students/summer-term-2018/project_2/data/CelebA/data64.npy')
        self.data_Y = np.load('/nfs/students/summer-term-2018/project_2/data/CelebA/landmarks5.npy')
        # TODO: different shuffles?
        #np.random.shuffle(self.data_X)
        #np.random.shuffle(self.data_Y)

        print('loaded data...')
        # generate model for sampling landmarks
        self.y_mean = np.mean(self.data_Y, axis=0)
        self.y_cov = np.cov(self.data_Y, rowvar=0)

        # Transform data in tensors
        self.data_X = torch.from_numpy(self.data_X).type(torch.FloatTensor)
        self.data_Y = torch.from_numpy(self.data_Y).type(torch.FloatTensor)[:,:,None,None]

        if torch.cuda.is_available():
            self.cuda = True
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss.cuda()

    def train(self):
        epochs = 25
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
        print('started training...')
        for epoch in range(epochs):
            print('Epoch', epoch)
            for i_batch in range(len(self.data_X) // self.batch_size):
                x = self.data_X[i_batch*self.batch_size:(i_batch+1)*self.batch_size]
                y = self.data_Y[i_batch*self.batch_size:(i_batch+1)*self.batch_size]
                z = torch.randn((self.batch_size, self.z_dim,1,1))
                y_gen = np.random.multivariate_normal(self.y_mean, self.y_cov,
                                                      size=(self.batch_size))
                y_gen = torch.from_numpy(y_gen[:,:,None,None]).type(torch.FloatTensor)
                if self.cuda:
                    x, y, y_gen, z = x.cuda(), y.cuda(), y_gen.cuda(), z.cuda()

                # ========== Training discriminator
                self.D_optimizer.zero_grad()

                # Train on real example from dataset
                D_real = self.D(x, y)
                D_real_loss = self.BCE_loss(D_real, y_real)

                # Train on fake example from generator
                # TODO: UserWarning: Using a target size (torch.Size([64, 1])) that is different
                # to the input size (torch.Size([64, 1, 1, 1])) is deprecated. Please ensure they have the same size.
                #   "Please ensure they have the same size.".format(target.size(), input.size()))
                x_fake = self.G(z, y)# y_gen)
                D_fake = self.D(x_fake, y)# y_gen)
                D_fake_loss = self.BCE_loss(D_fake, y_fake)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.data[0]) #TODO: deprecated index access

                D_loss.backward()
                self.D_optimizer.step()

                # ========== Training generator
                self.G_optimizer.zero_grad()

                # TODO: Check if reusable from generator training
                x_fake = self.G(z, y)#y_gen)
                D_fake = self.D(x_fake, y)#y_gen)
                G_loss = self.BCE_loss(D_fake, y_real)
                self.train_hist['G_loss'].append(G_loss.data[0])

                G_loss.backward()
                self.G_optimizer.step()

                if ((i_batch + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (i_batch + 1), len(self.data_X) // self.batch_size, D_loss.data[0], G_loss.data[0]))

        save_model_dict(self.G, '/home/stromaxi/G.model')
        save_model_dict(self.D, '/home/stromaxi/D.model')


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train()