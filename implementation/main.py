from FaceAnonymizer.Trainer import Trainer
from Preprocessor.Preprocessor import Preprocessor, deep_fake_config
from config import MERKEL_KLUM_BASE

if __name__ == '__main__':
    p = Preprocessor(MERKEL_KLUM_BASE)
    data = p.dataset

    trainer = Trainer(data, deep_fake_config)

    trainer.train()
