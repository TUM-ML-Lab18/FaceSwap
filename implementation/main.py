from pathlib import Path

import torch

from FaceAnonymizer.Trainer import Trainer

from configuration.general_config import MEGA_MERKEL_TRUMP, MERKEL_TRUMP_NORMAL_BASE, MERKEL_TRUMP_LANDMARKS
from configuration.run_config import current_config

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #p = current_config['preprocessor']()
    #p(Path(MERKEL_TRUMP_NORMAL_BASE)) #todo fix
    trainer = Trainer(Path(MERKEL_TRUMP_NORMAL_BASE), current_config)

    trainer.train()
