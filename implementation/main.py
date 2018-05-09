from FaceAnonymizer.Trainer import Trainer
from configuration.general_config import MERKEL_TRUMP_NORMAL_BASE
from configuration.run_config import current_config

if __name__ == '__main__':
    p = current_config['preprocessor'](MERKEL_TRUMP_NORMAL_BASE)
    data = p.dataset

    trainer = Trainer(data, current_config)

    trainer.train()
