import torch
from torch import optim, nn

from Models.CGAN import CGAN
from Models.ModelUtils.ModelUtils import CombinedModel


class DCGAN(CombinedModel):
    def __init__(self, **kwargs):
        self.image_size = (64, 64, 3)
        self.nz = 100
        self.ngf = 64
        self.ndf = 64

        # self.g = Generator(nc=self.image_size[2], nz=self.nz, ngf=self.ngf)
        self.g = CGAN.Generator(input_dim=(self.nz, 10), output_dim=self.image_size, ngf=self.ngf)
        # self.d = Discriminator(nc=self.image_size[2], ndf=self.ndf)
        self.d = CGAN.Discriminator(y_dim=10, input_dim=self.image_size, ndf=self.ndf)

        self.BCE_loss = nn.BCELoss()

        if torch.cuda.is_available():
            self.cuda = True
            self.g.cuda()
            self.d.cuda()
            self.BCE_loss.cuda()

        lrG = 0.0002
        lrD = 0.0002

        beta1, beta2 = 0.5, 0.999
        self.G_optimizer = optim.Adam(self.g.parameters(), lr=lrG, betas=(beta1, beta2))
        self.D_optimizer = optim.Adam(self.d.parameters(), lr=lrD, betas=(beta1, beta2))

        self.static_noise = torch.randn(64 * 4, self.nz)

    def train(self, data_loader, batch_size, validate, **kwargs):
        # Label vectors for loss function
        label_real, label_fake = (torch.ones(batch_size, 1, 1, 1), torch.zeros(batch_size, 1, 1, 1))

        if self.cuda:
            label_real, label_fake = label_real.cuda(), label_fake.cuda()

        # sum the loss for logging
        g_loss_summed, d_loss_summed = 0, 0
        iterations = 0

        for data, features in data_loader:
            if validate:
                noise = self.static_noise
            else:
                noise = torch.randn(batch_size, self.nz)
            if self.cuda:
                data = data.cuda()
                noise = noise.cuda()
                features = features.cuda()
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            if not validate:
                self.d.zero_grad()

            output = self.d(data, features)
            errD_real = self.BCE_loss(output, label_real)

            if not validate:
                errD_real.backward()

            # train with fake
            fake = self.g(noise, features)
            output = self.d(fake.detach(), features)
            errD_fake = self.BCE_loss(output, label_fake)
            if not validate:
                errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            if not validate:
                self.D_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            if not validate:
                self.g.zero_grad()
            output = self.d(fake, features)
            errG = self.BCE_loss(output, label_real)
            if not validate:
                errG.backward()
            D_G_z2 = output.mean().item()
            if not validate:
                self.G_optimizer.step()

            # losses
            g_loss_summed += float(errG)
            d_loss_summed += float(errD)
            iterations += 1

            if validate:
                break

        g_loss_summed /= iterations
        d_loss_summed /= iterations

        if not validate:
            log_info = {'loss': {'lossG': g_loss_summed,
                                 'lossD': d_loss_summed,
                                 'meanG': float(D_G_z1), 'meanD': float(D_G_z2)}}
        else:
            log_info = {'loss': {'lossG_val': g_loss_summed,
                                 'lossD_val': d_loss_summed,
                                 'meanG_val': float(D_G_z1), 'meanD_val': float(D_G_z2)}}

        return log_info, fake

    def get_models(self):
        return [self.g, self.d]

    def get_model_names(self):
        return ['generator', 'discriminator']

    def get_remaining_modules(self):
        return [self.G_optimizer, self.D_optimizer, self.BCE_loss]

    def anonymize(self, x):
        raise NotImplementedError

    def log_images(self, logger, epoch, images, validation=True):
        tag = 'validation_output' if validation else 'training_output'
        logger.log_images(epoch, images, tag, 8)
