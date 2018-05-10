from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import DataParallel
from torchvision.transforms import ToPILImage

from Preprocessor.FaceExtractor import ExtractionInformation


class LatentModel:
    def __init__(self, optimizer, scheduler, decoder, loss_function):
        self.decoder = decoder().cuda()

        if torch.cuda.device_count() > 1:
            self.decoder = DataParallel(self.decoder)

        self.lossfn = loss_function.cuda()

        self.optimizer1 = optimizer(self.decoder.parameters())
        self.scheduler1 = scheduler(self.optimizer1)

    def train(self, current_epoch, batches):
        loss1_mean, loss2_mean = 0, 0
        face1 = None
        output1 = None
        face2 = None
        output2 = None
        iterations = 0

        for (face1_landmarks, face1), (face2_landmarks, face2) in batches:
            # face1 and face2 contain a batch of images of the first and second face, respectively
            face1, face2 = face1.cuda(), face2.cuda()
            face1_landmarks, face2_landmarks = face1_landmarks.cuda(), face2_landmarks.cuda()

            self.optimizer1.zero_grad()
            output1 = self.decoder(face1_landmarks)
            loss1 = self.lossfn(output1, face1)
            loss1.backward()

            # output2 = self.decoder(face2_landmarks)
            # loss2 = self.lossfn(output2, face2)
            # loss2.backward()

            self.optimizer1.step()

            loss1_mean += loss1
            iterations += 1

        loss1_mean /= iterations
        loss1_mean = loss1_mean.cpu().data.numpy()
        loss2_mean = 0
        self.scheduler1.step(loss1_mean, current_epoch)

        return loss1_mean, loss2_mean, [face1, output1]

    def validate(self, batches):
        loss1_valid_mean, loss2_valid_mean = 0, 0
        iterations = 0

        for (face1_warped, face1), (face2_warped, face2) in batches:
            pass

        loss1_valid_mean /= iterations
        loss2_valid_mean /= iterations
        loss1_valid_mean = loss1_valid_mean.cpu().data.numpy()
        loss2_valid_mean = loss2_valid_mean.cpu().data.numpy()

        return loss1_valid_mean, loss2_valid_mean

    def anonymize(self, x: Image, y: ExtractionInformation):
        return self.decoder(y.landmarks)

    def anonymize_2(self, x: Image, y: ExtractionInformation):
        return self.anonymize(x, y)

    # TODO: Use save & load functions from models -> memory independent (RAM vs GPU)
    def save_model(self, path):
        # Create subfolder for models
        path = Path(path)
        subfolder = "model"  # "#datetime.now().strftime('model__%Y%m%d_%H%M%S')
        path = path / subfolder
        path.mkdir(parents=True, exist_ok=True)
        self.decoder.save(path / 'decoder.model')

    def load_model(self, path):
        path = Path(path)
        self.decoder.load(path / 'decoder.model')
