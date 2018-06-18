import math

from torch import optim

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
        self.batch_size = kwargs.get('batch_size', None)
        if self.batch_size is None or not math.log(self.batch_size, 2).is_integer():
            raise AttributeError(
                f"This module needs the variable batch_size. It also has to be to the power of 2. Got {self.batch_size}")
        self.target_resolution = kwargs.get('target_resolution', 32)
        self.latent_size = kwargs.get('latent_size', 512)

        # Modules with parameters
        self.G = Generator(num_channels=3, latent_size=self.latent_size, resolution=self.target_resolution,
                           fmap_max=self.latent_size, fmap_base=8192, tanh_at_end=True, ngpu=1).cuda()
        # self.G = nn.DataParallel(self.G)
        self.D = Discriminator(num_channels=3, mbstat_avg='all', resolution=self.target_resolution,
                               fmap_max=self.latent_size, fmap_base=8192, sigmoid_at_end=False, ngpu=1).cuda()
        # self.D = nn.DataParallel(self.D)

        # loss function
        # self.BCE_loss = nn.BCELoss()

        # move to gpu if available
        if torch.cuda.is_available():
            self.cuda = True
            self.G.cuda()
            self.D.cuda()
            # self.BCE_loss.cuda()

        # optimizers
        lrG = kwargs.get('lrG', 0.0002)
        lrD = kwargs.get('lrD', 0.0002)
        beta1 = kwargs.get('beta1', 0)
        beta2 = kwargs.get('beta2', 0.99)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=lrG, betas=(beta1, beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=lrD, betas=(beta1, beta2))

        # noise generation and static noise for logging
        self.noise = RandomNoiseGenerator(self.latent_size, 'gaussian')
        self.static_noise = self.noise(32)  # smallest batch size

        # variables for growing the network
        self.epochs_fade = 4
        self.images_per_fading = len(self.dataset) * self.epochs_fade  # CELEBA size
        self.epochs_stab = 4
        self.epochs_stage = self.epochs_fade + self.epochs_stab
        self.epochs_in_stage = self.epochs_fade  # Trick for stage 1
        self.level = 1
        self.imgs_faded_in = 0
        self.stabilization_phase = True
        self.level_with_multiple_gpus = 4

        self.batch_size_schedule = {1: 64, 2: 64, 3: 64, 4: 64, 5: 32, 6: 24}

    def get_models(self):
        return [self.G, self.D]

    def get_model_names(self):
        return ['generator', 'discriminator']

    def get_remaining_modules(self):
        return [self.noise]

    def train(self, train_data_loader, batch_size, validate, **kwargs):

        if not validate:
            self.schedule_resolution()
            train_data_loader = self.data_loader.get_train_data_loader()

        # # Classic GAN
        # # Label vectors for loss function
        # label_real, label_fake = (torch.ones(batch_size, 1, 1, 1), torch.zeros(batch_size, 1, 1, 1))
        # # Move to GPU
        # if self.cuda:
        #     label_real, label_fake = label_real.cuda(), label_fake.cuda()

        # Gradient weights WGAN-GP TODO: What are they for?
        one, mone = torch.FloatTensor([1]), torch.FloatTensor([-1])
        if self.cuda:
            one, mone = one.cuda(), mone.cuda()

        # sum the loss for logging
        g_loss_summed, d_loss_summed = 0, 0
        iterations = 0

        for images in train_data_loader:
            if self.stabilization_phase:
                fade_in_factor = 0
                cur_level = self.level
            else:
                fade_in_factor = self.imgs_faded_in / self.images_per_fading
                cur_level = self.level - 1 + (
                    fade_in_factor if fade_in_factor != 0 else 1e-10)  # FuckUp implementation...

            if validate:
                noise = self.static_noise
            else:
                noise = self.noise(self.batch_size)

            # Move to GPU
            if self.cuda:
                images = images.cuda()
                noise = noise.cuda()

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            if not validate:
                # Avoid computations in Generator training
                self.D.train(True)
                self.D_optimizer.zero_grad()

            # Train on real example with real features
            D_real = self.D(images, cur_level=cur_level)

            # Wasserstein Loss
            D_real = D_real.mean()
            if not validate:
                D_real.backward(mone)

            # # Classic loss
            # D_real_loss = self.BCE_loss(D_real, label_real)  # real corresponds to log(D_real)
            # if not validate:
            #     D_real_loss.backward()

            # Train on fake example from generator
            G_fake = self.G(noise, cur_level=cur_level)
            if validate:
                # Validate only generated image
                break
            # todo what happens if we detach the output of the Discriminator
            D_fake = self.D(G_fake.detach(), cur_level=cur_level)

            # Wasserstein Loss
            D_fake = D_fake.mean()
            if not validate:
                D_fake.backward(one)

            # # Classic loss
            # D_fake_loss = self.BCE_loss(D_fake, label_fake)  # fake corresponds to log(1-D_fake)
            # if not validate:
            #     D_fake_loss.backward()

            # # Classic loss
            # D_loss = D_real_loss + D_fake_loss

            # Wasserstein loss
            # train with gradient penalty
            gp = self.calculate_gradient_penalty(images, G_fake.detach(), cur_level)
            if not validate:
                gp.backward()

            # Wasserstein loss
            D_loss = float(D_fake - D_real + gp)
            Wasserstein_D = float(D_real - D_fake)

            if not validate:
                self.D_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            if not validate:
                self.D.train(False)
                self.G_optimizer.zero_grad()

            # Train on fooling the Discriminator
            D_fake = self.D(G_fake, cur_level=cur_level)

            # Wasserstein Loss
            D_fake = D_fake.mean()
            if not validate:
                D_fake.backward(mone)
                self.G_optimizer.step()
            G_loss = float(-D_fake)

            # # Classic loss
            # G_loss = self.BCE_loss(D_fake, label_real)
            # if not validate:
            #     G_loss.backward()
            #     self.G_optimizer.step()

            # losses
            g_loss_summed += G_loss
            d_loss_summed += D_loss
            iterations += 1

            if not self.stabilization_phase and not validate:
                # Count only images during training
                self.imgs_faded_in += self.batch_size

        if not validate:
            g_loss_summed /= iterations
            d_loss_summed /= iterations
            log_info = {'loss': {'lossG': g_loss_summed,
                                 'lossD': d_loss_summed},
                        'info/WassersteinDistance': Wasserstein_D,
                        'info/FadeInFactor': fade_in_factor,
                        'info/Level': self.level}
            log_img = G_fake
        else:
            log_info = {}
            log_img = G_fake

        return log_info, log_img

    def anonymize(self, extracted_face, extracted_information):
        raise NotImplementedError

    def log_images(self, logger, epoch, images, validation):
        tag = 'validation_output' if validation else 'training_output'
        logger.log_images(epoch, images[:64], tag, 8)

    def schedule_resolution(self):
        if self.epochs_in_stage >= self.epochs_stage:
            # Enter new stage
            self.level += 1
            self.batch_size = int(self.batch_size_schedule[self.level])
            self.data_loader.adjusted_batch_size_and_increase_resolution(self.batch_size)
            self.static_noise = self.static_noise[:self.batch_size]
            self.epochs_in_stage = 0
            print('Scheduling... level update, level:', self.level,
                  'epochs in stage:', self.epochs_in_stage,
                  'batch size:', self.batch_size)

        if self.epochs_in_stage < self.epochs_fade:
            # Fade in
            print('Scheduling... Fade-in phase, level:', self.level,
                  'epochs in stage:', self.epochs_in_stage,
                  'batch size:', self.batch_size)
            self.stabilization_phase = False
        else:
            # Stabilization phase
            print('Scheduling... Stabilization phase, level:', self.level,
                  'epochs in stage:', self.epochs_in_stage,
                  'batch size:', self.batch_size)
            self.stabilization_phase = True
            self.imgs_faded_in = 0

        if self.level == self.level_with_multiple_gpus:
            self.G.ngpu = self.D.ngpu = torch.cuda.device_count()

        self.epochs_in_stage += 1

    def calculate_gradient_penalty(self, real_data, fake_data, cur_level):
        """
        https://github.com/caogang/wgan-gp/
        """
        # Interpolation between real & fake data
        alpha = torch.rand(self.batch_size, 1, 1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda() if self.cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.cuda() if self.cuda else interpolates
        interpolates.requires_grad_()

        D_interpolate = self.D(interpolates, cur_level=cur_level)

        grad = torch.autograd.grad(outputs=D_interpolate,
                                   inputs=interpolates,
                                   grad_outputs=torch.ones(D_interpolate.size()).cuda() if self.cuda else
                                   torch.ones(D_interpolate.size()),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]

        _lambda = 10  # CelebA TF Code
        gradient_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean() * _lambda

        return gradient_penalty
