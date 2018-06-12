import math

from torch import nn, optim

from Models.ModelUtils.ModelUtils import CombinedModel, RandomNoiseGenerator
from Models.PGGAN_NEW.model import Generator, Discriminator, torch


class PGGAN(CombinedModel):
    def __init__(self, **kwargs):
        # get values from args
        self.dataset = kwargs.get('dataset', None)
        if self.dataset is None:
            raise FileNotFoundError()
        self.data_loader = kwargs.get('data_loader', None)
        if self.data_loader is None:
            raise FileNotFoundError()
        self.initial_batch_size = kwargs.get('initial_batch_size', None)
        if self.initial_batch_size is None or not math.log(self.initial_batch_size, 2).is_integer():
            raise AttributeError(
                f"This module needs the variable batch_size. It also has to be to the power of 2. Got {self.initial_batch_size}")
        self.target_resolution = kwargs.get('target_resolution', 32)
        self.latent_size = kwargs.get('latent_size', 512)

        # Modules with parameters
        self.G = Generator(num_channels=3, latent_size=self.latent_size, resolution=self.target_resolution,
                           fmap_max=self.latent_size, fmap_base=8192, tanh_at_end=True)
        # self.g = DataParallel(self.g)
        self.D = Discriminator(num_channels=3, mbstat_avg='all', resolution=self.target_resolution,
                               fmap_max=self.latent_size, fmap_base=8192, sigmoid_at_end=True)
        # self.d = DataParallel(self.d)

        # loss function
        self.BCE_loss = nn.BCELoss()

        # move to gpu if available
        if torch.cuda.is_available():
            self.cuda = True
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss.cuda()

        # optimizers
        lrG = kwargs.get('lrG', 0.0002)
        lrD = kwargs.get('lrD', 0.0002)
        beta1 = kwargs.get('beta1', 0.5)
        beta2 = kwargs.get('beta2', 0.999)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=lrG, betas=(beta1, beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=lrD, betas=(beta1, beta2))

        # noise generation and static noise for logging
        self.noise = RandomNoiseGenerator(self.latent_size, 'gaussian')
        self.static_noise = self.noise(self.initial_batch_size)

        # variables for growing the network
        self.TICK_dic = {1: 6, 2: 12, 3: 18, 4: 24, 5: 30}  # 2^5 = 32
        self.batch_size_dic = {1: self.initial_batch_size / (2 ** 0), 2: self.initial_batch_size / (2 ** 1),
                               3: self.initial_batch_size / (2 ** 2), 4: self.initial_batch_size / (2 ** 3),
                               5: self.initial_batch_size / (2 ** 4)}

        self.level = 1

    def get_models(self):
        return [self.G, self.D]

    def get_model_names(self):
        return ['generator', 'discriminator']

    def get_remaining_modules(self):
        return [self.noise]

    def train(self, train_data_loader, batch_size, validate, **kwargs):
        current_epoch = kwargs.get('current_epoch', 99999)
        batch_size = self.initial_batch_size
        # Label vectors for loss function
        label_real, label_fake = (torch.ones(batch_size, 1, 1, 1), torch.zeros(batch_size, 1, 1, 1))

        if self.cuda:
            label_real, label_fake = label_real.cuda(), label_fake.cuda()

        # sum the loss for logging
        g_loss_summed, d_loss_summed = 0, 0
        iterations = 0

        for images, features in train_data_loader:
            if validate:
                noise = self.static_noise
            else:
                noise = self.noise(batch_size)

            if self.cuda:
                images = images.cuda()
                noise = noise.cuda()
                features = features.cuda()

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                if not validate:
                    self.D_optimizer.zero_grad()

                # Train on real example with real features
                real_predictions = self.D(images, cur_level=self.level)
                d_real_predictions_loss = self.BCE_loss(real_predictions,
                                                        label_real)  # real corresponds to log(D_real)

                if not validate:
                    # make backward instantly
                    d_real_predictions_loss.backward()

                # Train on fake example from generator
                generated_images = self.G(noise, cur_level=self.level)
                fake_images_predictions = self.D(
                    generated_images.detach(),
                    cur_level=self.level)  # todo what happens if we detach the output of the Discriminator
                d_fake_images_loss = self.BCE_loss(fake_images_predictions,
                                                   label_fake)  # fake corresponds to log(1-D_fake)

                if not validate:
                    # make backward instantly
                    d_fake_images_loss.backward()

                d_loss = d_real_predictions_loss + d_fake_images_loss

                if not validate:
                    self.D_optimizer.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                if not validate:
                    self.G_optimizer.zero_grad()

                # Train on fooling the Discriminator
                fake_images_predictions = self.D(generated_images, cur_level=self.level)
                g_loss = self.BCE_loss(fake_images_predictions, label_real)

                if not validate:
                    g_loss.backward()
                    self.G_optimizer.step()

                # losses
                g_loss_summed += g_loss
                d_loss_summed += d_loss
                iterations += 1

            g_loss_summed /= iterations
            d_loss_summed /= iterations

        if not validate:
            log_info = {'lossG': float(g_loss_summed.cpu().data.numpy()),
                        'lossD': float(d_loss_summed.cpu().data.numpy())}
            log_img = generated_images

            self.schedule_resolution(current_epoch)
        else:
            log_info = {'lossG_val': float(g_loss_summed.cpu().data.numpy()),
                        'lossD_val': float(d_loss_summed.cpu().data.numpy())}
            log_img = generated_images

        return log_info, log_img

    def anonymize(self, extracted_face, extracted_information):
        raise NotImplementedError

    def log_images(self, logger, epoch, images, validation):
        tag = 'validation_output' if validation else 'training_output'
        logger.log_images(epoch, images[:64], tag, 8)

    def schedule_resolution(self, current_epoch):
        if self.TICK_dic[self.level] <= current_epoch:
            self.level += 1
            self.initial_batch_size = int(self.batch_size_dic[self.level])
            self.data_loader.adjusted_batch_size_and_increase_resolution(self.initial_batch_size)
            self.static_noise = self.static_noise[:self.initial_batch_size]
