import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from FaceAnonymizer import TrainValidationLoader

from FaceAnonymizer.models.Autoencoder import AutoEncoder
from FaceAnonymizer.models.Decoder import Decoder, LatentDecoder
from FaceAnonymizer.models.DeepFakeOriginal import DeepFakeOriginal
from FaceAnonymizer.models.Encoder import Encoder
from FaceAnonymizer.models.LatentModel import LatentModel
from Preprocessor.FaceExtractor import FaceExtractor
from Preprocessor.ImageDataset import ImageDatesetCombined, LandmarkDataset
from Preprocessor.Preprocessor import Preprocessor

standart_config = {'batch_size': 64,
                'num_epoch': 5000,
                'img_size': (128, 128),
                'validation_freq': 20,
                'data_loader':lambda dataset, batch_size: TrainValidationLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           validation_size=0.2,
                                           shuffle=True,
                                           num_workers=12,
                                           pin_memory=True,
                                           drop_last=True),
                'model': lambda: DeepFakeOriginal(
                    encoder=lambda: Encoder(input_dim=(3, 128, 128),
                                            latent_dim=1024,
                                            num_convblocks=5),
                    decoder=lambda: Decoder(input_dim=512,
                                            num_convblocks=4),
                    auto_encoder=AutoEncoder,
                    loss_function=torch.nn.L1Loss(size_average=True),
                    optimizer=lambda params: Adam(params=params, lr=1e-4),
                    scheduler=lambda optimizer: ReduceLROnPlateau(optimizer=optimizer,
                                                                  verbose=True,
                                                                  patience=100,
                                                                  cooldown=50), ),
                'preprocessor': lambda root_folder: Preprocessor(root_folder=root_folder,
                                                                 face_extractor=lambda: FaceExtractor(margin=0.05,
                                                                                                      mask_type=np.bool,
                                                                                                      mask_factor=10),
                                                                 image_dataset=lambda path: ImageDatesetCombined(
                                                                     dataset=path,
                                                                     img_size=(128, 128)))
                }

alex_config = {'batch_size': 512,
                'num_epoch': 5000,
                'img_size': (128, 128),
                'validation_freq': 20,
                'data_loader':lambda dataset, batch_size: TrainValidationLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           validation_size=0.2,
                                           shuffle=True,
                                           num_workers=12,
                                           pin_memory=True,
                                           drop_last=True),
                   'model': lambda: DeepFakeOriginal(
                    encoder=lambda: Encoder(input_dim=(3, 128, 128),
                                            latent_dim=1024,
                                            num_convblocks=5),
                    decoder=lambda: Decoder(input_dim=512,
                                            num_convblocks=4),
                    auto_encoder=AutoEncoder,
                    loss_function=torch.nn.L1Loss(size_average=True),
                    optimizer=lambda params: Adam(params=params, lr=1e-4),
                    scheduler=lambda optimizer: ReduceLROnPlateau(optimizer=optimizer,
                                                                  verbose=True,
                                                                  patience=100,
                                                                  cooldown=50), ),
                   'preprocessor': lambda root_folder: Preprocessor(root_folder=root_folder,
                                                                 face_extractor=lambda: FaceExtractor(margin=0.05,
                                                                                                      mask_type=np.bool,
                                                                                                      mask_factor=10),
                                                                 image_dataset=lambda path: ImageDatesetCombined(
                                                                     dataset=path,
                                                                     img_size=(128, 128)))
                   }

landmarks_config = {'batch_size': 64,
                    'num_epoch': 5000,
                    'img_size': (128, 128),
                    'validation_freq': 20,
                    'data_loader': lambda dataset, batch_size: TrainValidationLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           validation_size=0.2,
                                           shuffle=True,
                                           num_workers=12,
                                           pin_memory=True,
                                           drop_last=True),
                    'model': lambda: LatentModel(
                        decoder=lambda: LatentDecoder(72*2),
                        loss_function=torch.nn.L1Loss(size_average=True),
                        optimizer=lambda params: Adam(params=params, lr=1e-4),
                        scheduler=lambda optimizer: ReduceLROnPlateau(optimizer=optimizer,
                                                                      verbose=True,
                                                                      patience=100,
                                                                      cooldown=50)),
                    'preprocessor': lambda root_folder: Preprocessor(root_folder=root_folder,
                                                                     face_extractor=lambda: FaceExtractor(margin=0.05,
                                                                                                          mask_type=np.bool,
                                                                                                          mask_factor=10),
                                                                     image_dataset=lambda path: LandmarkDataset(
                                                                         dataset=path,
                                                                         img_size=(128, 128)))}
current_config = standart_config
