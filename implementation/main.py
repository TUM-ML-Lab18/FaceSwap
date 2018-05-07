from FaceAnonymizer.Trainer import Trainer
from Preprocessor.Preprocessor import Preprocessor, current_config
from config import MERKEL_KLUM_BASE, BARDEM_MORGAN_BASE, sebis_config, MERKEL_TRUMP_NO_MASK_BASE

if __name__ == '__main__':
    p = Preprocessor(MERKEL_TRUMP_NO_MASK_BASE, current_config)
    data = p.dataset

    trainer = Trainer(data, current_config)

    trainer.train()
