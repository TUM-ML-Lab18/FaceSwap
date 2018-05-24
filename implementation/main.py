from pathlib import Path

import torch

from FaceAnonymizer.Trainer import Trainer

from Configuration.config_general import MEGA_MERKEL_TRUMP, MERKEL_TRUMP_NORMAL_BASE, MERKEL_TRUMP_LANDMARKS, \
    TRUMP_LANDMARKS, CELEBA
from Configuration.config_model import current_config

if __name__ == '__main__':
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    p = current_config['preprocessor']()
    # p(Path(CELEBA))
    trainer = Trainer(None, current_config)

    trainer.train()
