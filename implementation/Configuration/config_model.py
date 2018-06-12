from Configuration.config_general import *
from Models.CGAN.CGAN import CGAN
from Models.DCGAN.DCGAN import DCGAN
from Models.DeepFake.Autoencoder import AutoEncoder
from Models.DeepFake.Decoder import Decoder
from Models.DeepFake.DeepFakeOriginal import DeepFakeOriginal
from Models.DeepFake.Encoder import Encoder
from Models.LatentModel.Decoder import LatentDecoder
from Models.LatentModel.LatentModel import LowResModel, RetrainLowResModel
from Models.PGGAN_NEW.PGGAN import PGGAN
from Utils.ImageDataset import *

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

# Latent model landmarks + LowRes
lowres_config = {'batch_size': 256,
                 'model': LowResModel,
                 'model_params': {'decoder': lambda: LatentDecoder(72 * 2 + 8 * 8 * 3)},
                 'dataset': lambda: ImageFeatureDataset(ARRAY_CELEBA_IMAGES_128,
                                                        [ARRAY_CELEBA_LANDMARKS, ARRAY_CELEBA_LOWRES])}

retrain_lowres_config = lowres_config.copy()
retrain_lowres_config['model'] = RetrainLowResModel
retrain_lowres_config['model_params'][
    'model_path'] = '/nfs/students/summer-term-2018/project_2/models/latent_model/model'
retrain_lowres_config['dataset'] = lambda: ImageFeatureDataset(ARRAY_CAR_IMAGES_128,
                                                               [ARRAY_CAR_LANDMARKS, ARRAY_CAR_LOWRES])

# CGAN
cgan_config = {'batch_size': 64,
               'model': CGAN,
               'model_params': {'y_dim': 56 + 8 * 8 * 3,
                                'z_dim': 100,
                                'ngf': 128,
                                'ndf': 128,
                                'lrG': 0.0002,
                                'lrD': 0.00005,
                                'y_mean': ARRAY_CELEBA_LANDMARKS_28_MEAN,
                                'y_cov': ARRAY_CELEBA_LANDMARKS_28_COV
                                },
               'dataset': lambda: ImageFeatureDataset(ARRAY_CELEBA_IMAGES_64,
                                                      [ARRAY_CELEBA_LANDMARKS_28, ARRAY_CELEBA_LOWRES])}

# DCGAN
dcgan_config = {'batch_size': 64,
                'model': DCGAN,
                'model_params': {},
                'dataset': lambda: ImageFeatureDataset(ARRAY_CELEBA_IMAGES_64, ARRAY_CELEBA_LANDMARKS_5)
                }

# DCGAN
pggan_config = {'batch_size': 64,
                'model': PGGAN,
                'model_params': {},
                'dataset': lambda: ImageFeatureDataset(ARRAY_CELEBA_IMAGES_64, ARRAY_CELEBA_LANDMARKS_5)
                }

current_config = pggan_config
