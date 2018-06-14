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
        ImageDatesetCombined(Path(SIMONE_MERKEL), size_multiplicator=100,
                             img_size=(128, 128))


# DeepFakes Original
deep_fakes_config = {'batch_size': 64,
                     'model': DeepFakeOriginal,
                     'model_params': {'encoder': lambda: Encoder(input_dim=(3, 128, 128),
                                                                 latent_dim=1024,
                                                                 num_convblocks=5),
                                      'decoder': lambda: Decoder(input_dim=512,
                                                                 num_convblocks=4),
                                      'auto_encoder': AutoEncoder,
                                      'select_autoencoder': 1},
                     'dataset': lambda: ImageDatesetCombined(Path(SIMONE_MERKEL), size_multiplicator=100,
                                                             img_size=(128, 128))
                     }


class LowResConfig(Config):
    def __init__(self):
        super().__init__()
        self.batch_size = 256
        self.model = LowResModel
        self.model_params = {'decoder': lambda: LatentDecoder(72 * 2 + 8 * 8 * 3)}
        self.dataset = lambda: ImageFeatureDataset(ARRAY_CELEBA_IMAGES_128,
                                                   [ARRAY_CELEBA_LANDMARKS, ARRAY_CELEBA_LOWRES])


# Latent model landmarks + LowRes
lowres_config = {'batch_size': 256,
                 'model': LowResModel,
                 'model_params': {'decoder': lambda: LatentDecoder(72 * 2 + 8 * 8 * 3)},
                 'dataset': lambda: ImageFeatureDataset(ARRAY_CELEBA_IMAGES_128,
                                                        [ARRAY_CELEBA_LANDMARKS, ARRAY_CELEBA_LOWRES])}


class RetrainConfig(LowResConfig):
    def __init__(self):
        super().__init__()
        self.model = RetrainLowResModel
        self.model_params['model_path'] = '/nfs/students/summer-term-2018/project_2/models/latent_model/model'
        self.dataset = lambda: ImageFeatureDataset(ARRAY_CAR_IMAGES_128,
                                                   [ARRAY_CAR_LANDMARKS, ARRAY_CAR_LOWRES])


retrain_lowres_config = lowres_config.copy()
retrain_lowres_config['model'] = RetrainLowResModel
retrain_lowres_config['model_params'][
    'model_path'] = '/nfs/students/summer-term-2018/project_2/models/latent_model/model'
retrain_lowres_config['dataset'] = lambda: ImageFeatureDataset(ARRAY_CAR_IMAGES_128,
                                                               [ARRAY_CAR_LANDMARKS, ARRAY_CAR_LOWRES])


class CGAN_CONFIG(Config):
    def __init__(self):
        super().__init__()
        self.model = CGAN,
        self.validation_size = 0.005
        self.model_params = {'y_dim': 56,
                             'z_dim': 44,
                             'ngf': 256,
                             'ndf': 256,
                             'lrG': 0.0002,
                             'lrD': 0.00005,
                             'lm_mean': ARRAY_CELEBA_LANDMARKS_28_MEAN,
                             'lm_cov': ARRAY_CELEBA_LANDMARKS_28_COV,
                             }
        self.dataset = lambda: ImageFeatureDataset(ARRAY_CELEBA_IMAGES_64, [ARRAY_CELEBA_LANDMARKS_28, ])


# CGAN
cgan_config = {'batch_size': 64,
               'model': CGAN,
               'model_params': {'y_dim': 56,
                                'z_dim': 100,
                                'ngf': 128,
                                'ndf': 128,
                                'lrG': 0.0002,
                                'lrD': 0.00005,
                                'lm_mean': ARRAY_CELEBA_LANDMARKS_28_MEAN,
                                'lm_cov': ARRAY_CELEBA_LANDMARKS_28_COV,
                                },
               'dataset': lambda: ImageFeatureDataset(ARRAY_CELEBA_IMAGES_64, [ARRAY_CELEBA_LANDMARKS_28, ])}


#LatentGAN
latent_gan_config = {
    'batch_size': 256,
    'model': LatentGAN,
    'model_params': {
        'input_dim': 72 * 2 + 8 * 8 * 3,
        'y_dim': 56,
        'z_dim': 44,
        'ndf': 256,
        'lrD': 0.00005
    },
    'dataset': lambda: ImageFeatureDataset(ARRAY_CELEBA_IMAGES_128, [ARRAY_CELEBA_LANDMARKS, ARRAY_CELEBA_LOWRES])
}


class DCGAN_CONFIG(Config):
    def __init__(self):
        super().__init__()
        self.model = DCGAN,
        self.validation_size = 0.005
        self.dataset = lambda: ImageFeatureDataset(ARRAY_CELEBA_IMAGES_64, ARRAY_CELEBA_LANDMARKS_5)


# DCGAN
dcgan_config = {'batch_size': 64,
                'model': DCGAN,
                'model_params': {},
                'dataset': lambda: ImageFeatureDataset(ARRAY_CELEBA_IMAGES_64, ARRAY_CELEBA_LANDMARKS_5)
                }


class PGGAN_CONFIG(Config):
    def __init__(self):
        super().__init__()
        self.batch_size = 128
        self.model = PGGAN,
        self.validation_size = 0.005
        self.model_params = {'target_resolution': 32,
                             'latent_size': 512,
                             'lrG': 0.0002,
                             'lrD': 0.00005,
                             'batch_size': 128}
        self.dataset = lambda: lambda: ProgressiveFeatureDataset(ARRAY_CELEBA_LANDMARKS_5, initial_resolution=2)


# PGGAN
pggan_config = {'batch_size': 64,
                'model': PGGAN,
                'model_params': {'target_resolution': 32,
                                 'latent_size': 512,
                                 'lrG': 0.001,
                                 'lrD': 0.001,
                                 'batch_size': 16},
                'dataset': lambda: ProgressiveFeatureDataset(ARRAY_CELEBA_LANDMARKS_5, initial_resolution=2)
                }

current_config = pggan_config
