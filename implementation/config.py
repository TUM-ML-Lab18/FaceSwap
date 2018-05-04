import torch

from FaceAnonymizer.models.Autoencoder import AutoEncoder
from FaceAnonymizer.models.DeepFakeOriginal import DeepFakeOriginal
from FaceAnonymizer.models.Encoder import Encoder
from FaceAnonymizer.models.Decoder import Decoder

TRUMP_CAGE_BASE = "/nfs/students/summer-term-2018/project_2/data/Trump_Cage"
TRUMP = "/trump"
CAGE = "/cage"
SMALL = "_small"

MERKEL_KLUM_BASE = "/nfs/students/summer-term-2018/project_2/data/Merkel_Klum"

CONVERTER_BASE = "/nfs/students/summer-term-2018/project_2/data/converter"
CONVERTER_INPUT = "/test_converter_images"
CONVERTER_OUTPUT = "/converter_output"

RAW = "raw"
A = "A"
B = "B"
PREPROCESSED = "preprocessed"

PROCESSED_IMAGES = TRUMP_CAGE_BASE + "/" + PREPROCESSED

IMAGE_DOWNLOADER = "/nfs/students/summer-term-2018/project_2/data/ImageDownloader"

SAMPLE_MODEL = "./model"

MOST_RECENT_MODEL = "."

deep_fake_config = {'model': DeepFakeOriginal,
                    'model_arguments': {'input_dim': (3, 64, 64),
                                        'latent_dim': 1024,
                                        'encoder': Encoder,
                                        'decoder': Decoder,
                                        'auto_encoder': AutoEncoder,
                                        'loss_function': torch.nn.L1Loss(size_average=True),
                                        'scheduler_arguments': {'threshold': 1e-6, 'verbose': True, 'patience': 100,
                                                                'cooldown': 50},
                                        'optimizer_arguments': {'lr': 1e-4}},
                    'batch_size': 64,
                    'num_epoch': 5000}
