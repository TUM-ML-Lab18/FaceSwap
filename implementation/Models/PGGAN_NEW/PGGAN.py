import os

from Models.ModelUtils.ModelUtils import CombinedModel, RandomNoiseGenerator
from Models.PGGAN_NEW.model import Generator, Discriminator, torch


class PGGAN(CombinedModel):
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = "1, 2, 3"
        self.target_resol = 32
        self.z_dim = 4
        self.latent_size = 512
        self.g = Generator(num_channels=3, latent_size=self.latent_size, resolution=self.target_resol,
                           fmap_max=self.latent_size, fmap_base=8192, tanh_at_end=True).cuda()
        print(self.g)
        # self.g = DataParallel(self.g)
        self.d = Discriminator(num_channels=3, mbstat_avg='all', resolution=self.target_resol,
                               fmap_max=self.latent_size, fmap_base=8192, sigmoid_at_end=True).cuda()
        print(self.d)
        # self.d = DataParallel(self.d)
        self.noise = RandomNoiseGenerator(self.latent_size, 'gaussian')

    def get_models(self):
        return [self.g, self.d]

    def get_model_names(self):
        return ['generator', 'discriminator']

    def get_remaining_modules(self):
        return [self.noise]

    def train(self, train_data_loader, batch_size, validate, **kwargs):
        for images, features in train_data_loader:
            z = torch.from_numpy(self.noise(batch_size)).cuda()
            level = 1  # only produce 2x2
            img = self.g(z, cur_level=level)
            disc = self.d(img, cur_level=level)

    def anonymize(self, extracted_face, extracted_information):
        pass

    def log_images(self, logger, epoch, images, validation):
        pass
