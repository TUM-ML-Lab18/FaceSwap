from FaceAnonymizer.Trainer import Trainer
from Preprocessor.Preprocessor import Preprocessor
from config import TRUMP_CAGE_BASE

if __name__ == '__main__':
    p = Preprocessor(TRUMP_CAGE_BASE)
    data = p.dataset

    model = Trainer(data)
    model.train()
