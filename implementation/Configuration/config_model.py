from Configuration.config_general import *
from Models.CGAN.CGAN import CGAN
from Models.DCGAN.DCGAN import DCGAN
from Models.DeepFake.Autoencoder import AutoEncoder
from Models.DeepFake.Decoder import Decoder
from Models.DeepFake.DeepFakeOriginal import DeepFakeOriginal
from Models.DeepFake.Encoder import Encoder
from Models.LatentGAN.LatentGAN import LatentGAN
from Models.LatentModel.Decoder import LatentDecoder
from Models.LatentModel.LatentModel import LowResModel, RetrainLowResModel
from Models.PGGAN_NEW.PGGAN import PGGAN
from Utils.ImageDataset import *


class Config:
    batch_size = 64
    validation_size = 0.2
    model = None
    model_params = None

    @staticmethod
    def data_set():
        raise NotImplementedError()

    max_epochs = 101
    validate_index = 0
    validation_frequencies = [2, 5, 20]
    validation_periods = [0, 10, 20, max_epochs + 1]

    save_model_every_nth = 20


class Deep_Fakes_Config(Config):
    model = DeepFakeOriginal
    model_params = {'encoder': lambda: Encoder(input_dim=(3, 128, 128),
                                               latent_dim=1024,
                                               num_convblocks=5),
                    'decoder': lambda: Decoder(input_dim=512,
                                               num_convblocks=4),
                    'auto_encoder': AutoEncoder,
                    'select_autoencoder': 1}

    @staticmethod
    def data_set():
        return ImageDatesetCombined(Path(SIMONE_MERKEL), size_multiplicator=100,
                                    img_size=(128, 128))


class LowResConfig(Config):
    batch_size = 256
    model = LowResModel
    model_params = {'decoder': lambda: LatentDecoder(72 * 2 + 8 * 8 * 3)}

    @staticmethod
    def data_set():
        return ImageFeatureDataset(ARRAY_CELEBA_IMAGES_128,
                                   [ARRAY_CELEBA_LANDMARKS, ARRAY_CELEBA_LOWRES])


class RetrainConfig(LowResConfig):
    model = RetrainLowResModel
    LowResConfig.model_params['model_path'] = '/nfs/students/summer-term-2018/project_2/models/latent_model/model'
    dataset = lambda: ImageFeatureDataset(ARRAY_CAR_IMAGES_128,
                                          [ARRAY_CAR_LANDMARKS, ARRAY_CAR_LOWRES])


class GAN_CONFIG(Config):
    validation_size = 0.005
    validation_frequencies = [1, 1, 1]
    save_model_every_nth = 5


class CGAN_CONFIG(GAN_CONFIG):
    model = CGAN,
    model_params = {'y_dim': 56,
                    'z_dim': 44,
                    'ngf': 256,
                    'ndf': 256,
                    'lrG': 0.0002,
                    'lrD': 0.00005,
                    'lm_mean': ARRAY_CELEBA_LANDMARKS_28_MEAN,
                    'lm_cov': ARRAY_CELEBA_LANDMARKS_28_COV,
                    }

    @staticmethod
    def data_set():
        return ImageFeatureDataset(ARRAY_CELEBA_IMAGES_64, [ARRAY_CELEBA_LANDMARKS_28, ])


class LGAN_CONFIG(GAN_CONFIG):
    model = LatentGAN
    batch_size = 256
    model_params = {
        'input_dim': 72 * 2 + 8 * 8 * 3,
        'img_dim': (128, 128, 3),
        'z_dim': 44,
        'ndf': 256,
        'lrD': 0.00005,
        'alpha': 0.5
    }

    @staticmethod
    def data_set():
        return ImageFeatureDataset(ARRAY_CELEBA_IMAGES_128, [ARRAY_CELEBA_LANDMARKS, ARRAY_CELEBA_LOWRES])


class DCGAN_CONFIG(GAN_CONFIG):
    model = DCGAN

    @staticmethod
    def dataset():
        return ImageFeatureDataset(ARRAY_CELEBA_IMAGES_64, ARRAY_CELEBA_LANDMARKS_5)


class PGGAN_CONFIG(GAN_CONFIG):
    batch_size = 64
    model = PGGAN
    model_params = {'target_resolution': 32,
                    'latent_size': 512,
                    'lrG': 0.001,
                    'lrD': 0.001,
                    'batch_size': batch_size}

    @staticmethod
    def data_set():
        return ProgressiveFeatureDataset(ARRAY_CELEBA_LANDMARKS_5, initial_resolution=2)


current_config = PGGAN_CONFIG
