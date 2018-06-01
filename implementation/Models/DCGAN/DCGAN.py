import random

import torch
from torch import optim, nn

from Models.DCGAN.Discriminator import Discriminator
from Models.DCGAN.Generator import Generator
from Models.ModelUtils.ModelUtils import CombinedModel


class DCGAN(CombinedModel):
    def __init__(self, **kwargs):
        self.batch_size = 64
        self.image_size = (64, 64, 3)
        self.nz = 100
        self.ngf = 64
        self.ndf = 64

        self.g = Generator(nc=self.image_size[2], nz=self.nz, ngf=self.ngf)
        self.d = Discriminator(nc=self.image_size[2], ndf=self.ndf)

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

        self.real_label = 1
        self.fake_label = 0

    def __str__(self):
        string = super().__str__()
        string += str(self.G_optimizer) + '\n'
        string += str(self.D_optimizer) + '\n'
        string += str(self.BCE_loss) + '\n'
        string = string.replace('\n', '\n\n')
        return string

    def get_models(self):
        return [self.g, self.d]

    def get_model_names(self):
        return ['generator', 'discriminator']

    def _train(self, data_loader, batch_size, **kwargs):
        # indicates if the graph should get updated
        validate = kwargs.get('validate', False)

        # sum the loss for logging
        g_loss_summed, d_loss_summed = 0, 0
        iterations = 0

        for i, data in enumerate(data_loader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            if not validate:
                self.d.zero_grad()
            real_cpu = data[0].cuda()
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), self.real_label).cuda()

            output = self.d(real_cpu)
            errD_real = self.BCE_loss(output, label)

            if not validate:
                errD_real.backward()

            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, self.nz, 1, 1).cuda()
            fake = self.g(noise)
            label.fill_(self.fake_label)
            output = self.d(fake.detach())
            errD_fake = self.BCE_loss(output, label)
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
            label.fill_(self.real_label)  # fake labels are real for generator cost
            output = self.d(fake)
            errG = self.BCE_loss(output, label)
            if not validate:
                errG.backward()
            D_G_z2 = output.mean().item()
            if not validate:
                self.G_optimizer.step()

            # losses
            g_loss_summed += errG
            d_loss_summed += errD
            iterations += 1

        g_loss_summed /= iterations
        d_loss_summed /= iterations

        return g_loss_summed.cpu().data.numpy(), d_loss_summed.cpu().data.numpy(), D_G_z1, D_G_z2, fake

    def train(self, train_data_loader, batch_size, **kwargs):
        g_loss, d_loss, g_mean, d_mean, generated_images = self._train(train_data_loader, batch_size, validate=False,
                                                                       **kwargs)
        return g_loss, d_loss, g_mean, d_mean, generated_images

    def validate(self, validation_data_loader, batch_size, **kwargs):
        g_loss, d_loss, g_mean, d_mean, generated_images = self._train(validation_data_loader, batch_size,
                                                                       validate=True, **kwargs)
        return g_loss, d_loss, g_mean, d_mean, generated_images

    def log(self, logger, epoch, lossG, lossD, g_mean, d_mean, images, log_images=False):
        """
        use logger to log current loss etc...
        :param logger: logger used to log
        :param epoch: current epoch
        """
        logger.log_loss(epoch=epoch, loss={'lossG': float(lossG), 'lossD': float(lossD), 'meanG': float(g_mean),
                                           'meanD': float(d_mean)})
        logger.log_fps(epoch=epoch)
        logger.save_model(epoch)

        if log_images:
            self.log_images(logger, epoch, images, validation=False)

    def log_images(self, logger, epoch, images, validation=True):
        images = images.cpu()
        examples = int(len(images))
        example_indices = random.sample(range(0, examples - 1), 4 * 4)
        A = []
        for idx, i in enumerate(example_indices):
            A.append(images[i])
        tag = 'validation_output' if validation else 'training_output'
        logger.log_images(epoch, A, tag, 4)

    def log_validation(self, logger, epoch, lossG, lossD, g_mean, d_mean, images):
        logger.log_loss(epoch=epoch,
                        loss={'lossG_val': float(lossG), 'lossD_val': float(lossD), 'meanG_val': float(g_mean),
                              'meanD_val': float(d_mean)})
        self.log_images(logger, epoch, images, validation=True)

    def img2latent_bridge(self, extracted_face, extracted_information):
        pass

    def anonymize(self, x):
        pass
