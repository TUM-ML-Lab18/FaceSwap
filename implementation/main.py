from FaceAnonymizer.Trainer import Trainer
from Preprocessor.Preprocessor import Preprocessor, deep_fake_config
from config import MERKEL_KLUM_BASE, BARDEM_MORGAN_BASE, sebis_config

if __name__ == '__main__':
    current_config = sebis_config

    p = Preprocessor(BARDEM_MORGAN_BASE, current_config)
    data = p.dataset

    trainer = Trainer(data, current_config)

    trainer.train()
