from FaceAnonymizer.Trainer import Trainer
from Preprocessor.Preprocessor import Preprocessor
from config import MERKEL_KLUM_BASE

if __name__ == '__main__':
    p = Preprocessor(MERKEL_KLUM_BASE)
    data = p.dataset

    model = Trainer(data, batch_size=64)
    model.train()
