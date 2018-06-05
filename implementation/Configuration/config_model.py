from torchvision.transforms import ToTensor

from Configuration.config_general import *
from Models.CGAN.CGAN import CGAN
from Models.DCGAN.DCGAN import DCGAN
from Models.DeepFake.Autoencoder import AutoEncoder
from Models.DeepFake.Decoder import Decoder
from Models.DeepFake.DeepFakeOriginal import DeepFakeOriginal
from Models.DeepFake.Encoder import Encoder
from Models.LatentModel.Decoder import LatentDecoder
from Models.LatentModel.LatentModel import LatentModel, LowResAnnotationModel, HistAnnotationModel, HistModel, \
    LowResModel
from Utils.ImageDataset import *

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

####### config for using only landmarks as input
# todo move the input dimensionality of the network to somewhere als as parameter
# dim = 72*2
landmarks_config = {'batch_size': 64,
                    'img_size': (128, 128),
                    'model': LatentModel,
                    'model_params': {'decoder': lambda: LatentDecoder(72 * 2 + 8 * 8 * 3)},
                    'dataset': lambda: ImageFeatureDataset(ARRAY_CELEBA_IMAGES_64, [ARRAY_CELEBA_LANDMARKS])}

####### config for using landmarks as well as a low res image as input
# dim = 72+2+8+8+3
lm_lowres_config = landmarks_config.copy()
lm_lowres_config['dataset'] = lambda: ImageFeatureDataset(ARRAY_CAR_IMAGES_128,
                                                          [ARRAY_CAR_LANDMARKS, ARRAY_CAR_LOWRES])
lm_lowres_config['model'] = LowResModel

###### config for using landmarks as well as a histogram of the target as input
lm_hist_config = landmarks_config.copy()
lm_hist_config['dataset'] = lambda: ImageFeatureDataset(ARRAY_CELEBA_IMAGES_64,
                                                        [ARRAY_CELEBA_LANDMARKS, ARRAY_CELEBA_HISTO])
lm_hist_config['model'] = lambda img_size: HistModel(decoder=lambda: LatentDecoder(72 * 2 + 768))

###### config for using landmarks as well as a histogram as well as annotations of the target as input
lm_hist_annotations_config = lm_hist_config.copy()
lm_hist_annotations_config['dataset'] = lambda: ImageFeatureDataset(ARRAY_CELEBA_IMAGES_64,
                                                                    [ARRAY_CELEBA_LANDMARKS, ARRAY_CELEBA_HISTO,
                                                                     ARRAY_CELEBA_ATTRIBUTES])
lm_hist_annotations_config['model'] = lambda img_size: HistAnnotationModel(
    decoder=lambda: LatentDecoder(72 * 2 + 768 + 40))

###### config for using landmarks as well as a lowres as well as annotations of the target as input
lm_lowres_annotations_config = lm_lowres_config.copy()
lm_lowres_annotations_config['dataset'] = lambda: ImageFeatureDataset(ARRAY_CELEBA_IMAGES_64,
                                                                      [ARRAY_CELEBA_LANDMARKS, ARRAY_CELEBA_ATTRIBUTES,
                                                                       ARRAY_CELEBA_LOWRES])
lm_lowres_annotations_config['model'] = lambda img_size: LowResAnnotationModel(
    decoder=lambda: LatentDecoder(72 * 2 + 8 * 8 * 3 + 40))

cgan_config = {'batch_size': 64,
               'model': CGAN,
               'model_params': {'y_dim': 10,
                                'z_dim': 100,
                                'lrG': 0.0002,
                                'lrD': 0.0002,
                                'y_mean': ARRAY_CELEBA_LANDMARKS_5_MEAN,
                                'y_cov': ARRAY_CELEBA_LANDMARKS_5_COV},
               'dataset': lambda: ImageFeatureDataset(ARRAY_CELEBA_IMAGES_64, ARRAY_CELEBA_LANDMARKS_5)}

dcgan_config = {'batch_size': 64,
                'model': DCGAN,
                'model_params': {},
                # 'dataset2': lambda: CIFAR10(root='./Models/DCGAN/download', download=True,
                #                             transform=transforms.Compose([
                #                                 transforms.Resize(64),
                #                                 transforms.ToTensor(),
                #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                #                             ])),
                # 'dataset': lambda: ImageFolder(
                #     root='/nfs/students/summer-term-2018/project_2/data/CelebA/preprocessed64',
                #     transform=transforms.Compose([
                #         transforms.Resize(64),
                #         transforms.ToTensor(),
                #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                #     ]))
                'dataset': lambda: ImageFeatureDataset(ARRAY_CELEBA_IMAGES_64, ARRAY_CELEBA_LANDMARKS_5)
                }

current_config = cgan_config
