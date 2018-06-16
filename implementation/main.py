import torch

from Configuration.config_model import current_config
from Utils.Trainer import Trainer

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic = True

    trainer = Trainer(current_config)

    trainer.train()
