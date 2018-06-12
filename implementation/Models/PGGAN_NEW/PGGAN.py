from Models.ModelUtils.ModelUtils import CombinedModel, RandomNoiseGenerator
from Models.PGGAN_NEW.model import Generator, Discriminator, torch


class PGGAN(CombinedModel):
    def __init__(self):
        self.target_resol = 32
        self.latent_size = 512
        self.g = Generator(num_channels=3, latent_size=self.latent_size, resolution=self.target_resol,
                           fmap_max=self.latent_size, fmap_base=8192, tanh_at_end=True).cuda()
        # self.g = DataParallel(self.g)
        self.d = Discriminator(num_channels=3, mbstat_avg='all', resolution=self.target_resol,
                               fmap_max=self.latent_size, fmap_base=8192, sigmoid_at_end=True).cuda()
        # self.d = DataParallel(self.d)
        self.noise = RandomNoiseGenerator(self.latent_size, 'gaussian')

        self.TICK = 1000
        self.TICK_dic = {1: 1000, 2: 1000, 3: 1000, 4: 1000, 5: 1000}  # 2^5 = 32

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
