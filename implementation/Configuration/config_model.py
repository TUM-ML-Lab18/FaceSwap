from torchvision.transforms import ToTensor

from Configuration.config_general import *
from Models.CGAN.CGAN import CGAN
from Models.DCGAN.DCGAN import DCGAN
from Models.DeepFake.Autoencoder import AutoEncoder
from Models.DeepFake.Decoder import Decoder
from Models.DeepFake.DeepFakeOriginal import DeepFakeOriginal
from Models.DeepFake.Encoder import Encoder
from Models.LatentModel.Decoder import LatentDecoder
from Models.LatentModel.LatentModel import LowResModel
from Utils.ImageDataset import *

# TODO: Remove, if DeepFakes config works
standard_config = {'batch_size': 64,
                   'img_size': (128, 128),
                   'model': lambda img_size: DeepFakeOriginal(
                       encoder=lambda: Encoder(input_dim=(3,) + img_size,
                                               latent_dim=1024,
                                               num_convblocks=5),
                       decoder=lambda: Decoder(input_dim=512,
                                               num_convblocks=4),
                       auto_encoder=AutoEncoder),
                   'dataset': lambda root_folder, img_size: ImageDatesetCombined(root_folder, size_multiplicator=1,
                                                                                 img_size=img_size),
                   'img2latent_bridge:': lambda extracted_face, extracted_information, img_size:
                   ToTensor()(extracted_face.resize(img_size, resample=BICUBIC)).unsqueeze(0).cuda()
                   }

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
                     'dataset': lambda: ImageDatesetCombined(MEGA_MERKEL_TRUMP, size_multiplicator=1,
                                                             img_size=(128, 128))
                     }

# Latent model landmarks + LowRes
lowres_config = {'batch_size': 256,
                 'model': LowResModel,
                 'model_params': {'decoder': lambda: LatentDecoder(72 * 2 + 8 * 8 * 3)},
                 'dataset': lambda: ImageFeatureDataset(ARRAY_CELEBA_IMAGES_128,
                                                        [ARRAY_CELEBA_LANDMARKS, ARRAY_CELEBA_LOWRES])}

# CGAN
cgan_config = {'batch_size': 64,
               'model': CGAN,
               'model_params': {'y_dim': 10,
                                'z_dim': 100,
                                'lrG': 0.0002,
                                'lrD': 0.0002,
                                'y_mean': ARRAY_CELEBA_LANDMARKS_5_MEAN,
                                'y_cov': ARRAY_CELEBA_LANDMARKS_5_COV},
               'dataset': lambda: ImageFeatureDataset(ARRAY_CELEBA_IMAGES_64, ARRAY_CELEBA_LANDMARKS_5)}

# DCGAN
dcgan_config = {'batch_size': 64,
                'model': DCGAN,
                'model_params': {},
                'dataset': lambda: ImageFeatureDataset(ARRAY_CELEBA_IMAGES_64, ARRAY_CELEBA_LANDMARKS_5)
                }

current_config = lowres_config
