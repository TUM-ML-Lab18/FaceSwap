from torch import optim

from Models.ModelUtils.ModelUtils import CombinedModel, RandomNoiseGenerator
from Models.PGGAN.model import Generator, Discriminator, torch


class PGGAN(CombinedModel):
    def __init__(self, **kwargs):
        # the data loader is used to change resolution of input images
        self.data_loader = kwargs.get('data_loader', None)
        if self.data_loader is None:
            raise FileNotFoundError()

        # the maximum resolution we will train
        self.target_resolution = kwargs.get('target_resolution', 64)

        # Features size in classic PGGAN always 0 -> Ensure compatibility with C-PGGAN
        self.feature_size = kwargs.get('feature_size', 0)
        # this latent_size is used as channels for generator and discriminator
        self.latent_size = kwargs.get('latent_size', 512)

        # Modules with parameters
        self.G = Generator(num_channels=3, latent_size=self.latent_size, resolution=self.target_resolution,
                           fmap_max=self.latent_size, fmap_base=8192, tanh_at_end=True, ngpu=1)
        self.D = Discriminator(num_channels=3 + self.feature_size, mbstat_avg='all', resolution=self.target_resolution,
                               fmap_max=self.latent_size, fmap_base=8192, sigmoid_at_end=False, ngpu=1)

        # move to gpu if available
        if torch.cuda.is_available():
            self.cuda = True
            self.G.cuda()
            self.D.cuda()

        # optimizers
        lrG = kwargs.get('lrG', 0.0002)
        lrD = kwargs.get('lrD', 0.0002)
        beta1 = kwargs.get('beta1', 0)
        beta2 = kwargs.get('beta2', 0.99)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=lrG, betas=(beta1, beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=lrD, betas=(beta1, beta2))

        ############################
        # variables for growing the network
        ###########################

        # fading phase: interpolation between two resolutions: (increasing the resolution bit by bit)
        self.fading_epochs = kwargs.get('epochs_fade', None)
        # stabilizing phase: just normal training with one constant resolution (after fading)
        self.stabilizing_epochs = kwargs.get('epochs_stab', None)
        # how many epochs correspond to one stage?
        self.epochs_per_stage = self.fading_epochs + self.stabilizing_epochs
        # how many epochs are trained allready in the current stage?
        # by setting it initially to fading_epochs we ensure that we don't fade in but only "stabilize"
        self.epochs_in_current_stage = self.fading_epochs  # Trick for stage 1

        # the level of the current resolution, we start with 4x4 | res = 2**(1+resolution_level)
        # 1     2       3       4       5       6
        # 4x4   8x8     16x16   32x32   64x64   128x128
        self.resolution_level = 1

        # the number of images shown to the network until completion of the fading phase
        self.images_per_fading_phase = len(self.data_loader.get_train_data_loader()) * self.fading_epochs
        # current number of images shown to the network during fading phase
        self.images_faded_in = 0
        # indicates the current phase
        self.stabilization_phase = True

        # we start training with only one gpu because the broadcasting operations consume on the lower resolution a lot
        # of time; but when reaching this level, switch to training with all available gpus
        self.level_with_multiple_gpus = kwargs.get('level_with_multiple_gpus', 4)

        # batch sizes for each level | be careful with changing them independent from the learning rate | original prams
        # from the paper
        # todo: what happens with the batchsize if using multiple gpus? maybe we should fix this
        self.batch_size_schedule = kwargs.get('batch_size_schedule',
                                              {1: 64, 2: 64, 3: 64, 4: 64, 5: 16, 6: 16})
        # current batch size is just the current levels batch size
        self.batch_size = self.batch_size_schedule[self.resolution_level]

        # noise generation and static noise for logging
        self.noise = RandomNoiseGenerator(self.latent_size - self.feature_size, 'gaussian')
        self.static_noise = self.noise(self.batch_size)

    def get_models(self):
        return [self.G, self.D]

    def get_model_names(self):
        return ['generator', 'discriminator']

    def get_remaining_modules(self):
        return [self.G_optimizer, self.D_optimizer, self.noise]

    def train(self, train_data_loader, batch_size, validate, **kwargs):

        # during training we adjust our current level if needed -> higher resolution -> we need to reload the
        # data_loader
        if not validate:
            # todo write with return statement -> we get a data loader from it
            self.schedule_resolution()
            train_data_loader = self.data_loader.get_train_data_loader()

        # sum the loss for logging
        g_loss_summed, d_loss_summed, wasserstein_d_summed, eps_summed = 0, 0, 0, 0
        iterations = 0

        for images in train_data_loader:
            # todo move to self.schedule_resolution?
            # set the fade_in_factor:
            # during stabilizing phase we don't interpolate but use the current level=resolution
            # after increasing the resolution we interpolate between de lower and higher resolution
            # i.e.: self.level = 3 (4*1*2*2x4*1*2*2: 16x16)
            # this means we were in level 2 but interpolate to level 3:
            # cur_level = self.level - 1 = 2
            # now we add the fade_in_factor which is between 1e-10 and 1
            # as a result the cur_level is a value between the last level and the current value, but it has to be
            # greater then the last level (2) otherwise the forward does not work

            if self.stabilization_phase:
                fade_in_factor = 0
                cur_level = self.resolution_level
            else:
                fade_in_factor = self.images_faded_in / self.images_per_fading_phase
                cur_level = self.resolution_level - 1 + (
                    fade_in_factor if fade_in_factor != 0 else 1e-10)  # FuckUp implementation...

            # differentiate between validation and training
            if validate:
                # for validation use always the same noise -> you can see how the face gets better in the tensorboard
                noise = self.static_noise[:self.batch_size]
            else:
                # generate new noise
                noise = self.noise(self.batch_size)

            # Move to GPU
            if self.cuda:
                images = images.cuda()
                noise = noise.cuda()

            ############################
            # (1) Update D network: minimize -D(x) + D(G(z)) + penalty instead of clipping
            ###########################
            if not validate:
                self.D.train(True)
                self.D_optimizer.zero_grad()

            # Train on real example with real features
            D_real = self.D(images, cur_level=cur_level)

            # Epsilon loss => 4th loss term from Nvidia paper
            eps_loss = D_real ** 2
            eps_loss = 0.001 * eps_loss.mean()

            # Wasserstein Loss
            D_real = -D_real.mean() + eps_loss
            if not validate:
                D_real.backward()

            # Train on fake example from generator
            G_fake = self.G(noise, cur_level=cur_level)
            if validate:
                # Validate only generated image
                break
            D_fake = self.D(G_fake.detach(), cur_level=cur_level)

            # Wasserstein Loss
            D_fake = D_fake.mean()
            if not validate:
                D_fake.backward()

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
            # (2) Update G network: minimize -D(G(z)) (is same to maximise D(G(z)) / Discriminator makes an error)
            ###########################
            if not validate:
                # Avoid computations in Generator training
                self.D.train(False)
                self.G_optimizer.zero_grad()

            # Train on fooling the Discriminator
            D_fake = self.D(G_fake, cur_level=cur_level)

            # Wasserstein Loss
            D_fake = -D_fake.mean()
            if not validate:
                D_fake.backward()
                self.G_optimizer.step()
            G_loss = float(D_fake)

            # losses
            g_loss_summed += G_loss
            d_loss_summed += D_loss
            wasserstein_d_summed += Wasserstein_D
            eps_summed += float(eps_loss)
            iterations += 1

            if not self.stabilization_phase and not validate:
                # Count only images during training
                self.images_faded_in += self.batch_size

        if not validate:
            g_loss_summed /= iterations
            d_loss_summed /= iterations
            wasserstein_d_summed /= iterations
            eps_summed /= iterations
            log_info = {'loss': {'lossG': g_loss_summed,
                                 'lossD': d_loss_summed},
                        'info/WassersteinDistance': wasserstein_d_summed,
                        'info/eps': eps_summed,
                        'info/FadeInFactor': fade_in_factor,
                        'info/Level': self.resolution_level}
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
        if self.epochs_in_current_stage >= self.epochs_per_stage:
            # Enter new stage
            self.resolution_level += 1
            self.batch_size = int(self.batch_size_schedule[self.resolution_level])
            self.data_loader.adjusted_batch_size_and_increase_resolution(self.batch_size)
            self.static_noise = self.static_noise[:self.batch_size]
            self.epochs_in_current_stage = 0
            print('Scheduling... level update, level:', self.resolution_level,
                  'epochs in stage:', self.epochs_in_current_stage,
                  'batch size:', self.batch_size)

        if self.epochs_in_current_stage < self.fading_epochs:
            # Fade in
            print('Scheduling... Fade-in phase, level:', self.resolution_level,
                  'epochs in stage:', self.epochs_in_current_stage,
                  'batch size:', self.batch_size)
            self.stabilization_phase = False
        else:
            # Stabilization phase
            print('Scheduling... Stabilization phase, level:', self.resolution_level,
                  'epochs in stage:', self.epochs_in_current_stage,
                  'batch size:', self.batch_size)
            self.stabilization_phase = True
            self.images_faded_in = 0

        if self.resolution_level == self.level_with_multiple_gpus:
            self.G.ngpu = self.D.ngpu = torch.cuda.device_count()

        self.epochs_in_current_stage += 1

    def calculate_gradient_penalty(self, real_data, fake_data, cur_level):
        """
        https://github.com/caogang/wgan-gp/
        :param real_data: todo
        :param fake_data: todo
        :param cur_level: todo
        :return:
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

        _lambda = 10  # CelebA TF Code (NVIDIA PAPER)
        gradient_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean() * _lambda

        return gradient_penalty
