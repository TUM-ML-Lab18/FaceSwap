from FaceAnonymizer.Trainer import Trainer
from Preprocessor.Preprocessor import Preprocessor, deep_fake_config
from config import MERKEL_KLUM_BASE, sebis_config

if __name__ == '__main__':
    p = Preprocessor(MERKEL_KLUM_BASE, deep_fake_config)
    data = p.dataset

    trainer = Trainer(data, deep_fake_config)

    trainer.train()
