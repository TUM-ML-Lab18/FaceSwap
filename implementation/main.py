from FaceAnonymizer.Trainer import Trainer

from configuration.general_config import MEGA_MERKEL_TRUMP, MERKEL_TRUMP_NORMAL_BASE, MERKEL_TRUMP_LANDMARKS
from configuration.run_config import current_config

if __name__ == '__main__':
    p = current_config['preprocessor'](MERKEL_TRUMP_LANDMARKS)
    data = p.dataset
    trainer = Trainer(data, current_config)

    trainer.train()
