from FaceAnonymizer.Trainer import Trainer
from configuration.gerneral_config import MERKEL_TRUMP_NORMAL_BASE, MERKEL_TRUMP_LANDMARKS, MEGA_MERKEL_TRUMP
from configuration.run_config import current_config

if __name__ == '__main__':
    p = current_config['preprocessor'](MEGA_MERKEL_TRUMP)
    data = p.dataset
    trainer = Trainer(data, current_config)

    #trainer.train()
