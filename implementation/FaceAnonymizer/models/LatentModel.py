from pathlib import Path

import random
import torch
from PIL import Image
from torch.nn import DataParallel

from Preprocessor.FaceExtractor import ExtractionInformation


class LatentModel:
    def __init__(self, optimizer, scheduler, decoder, loss_function):
        self.decoder = decoder()

        if torch.cuda.device_count() > 1:
            self.decoder = DataParallel(self.decoder)
        self.decoder = self.decoder.cuda()

        self.lossfn = loss_function.cuda()

        self.optimizer1 = optimizer(self.decoder.parameters())
        self.scheduler1 = scheduler(self.optimizer1)

    def set_train_mode(self, mode):
        self.decoder.train(mode)

    def train(self, current_epoch, batches):
        loss1_mean = 0
        face1 = None
        output1 = None
        iterations = 0

        for (latent_information, face1) in batches:
            face1 = face1.cuda()
            latent_information = latent_information.cuda()

            self.optimizer1.zero_grad()
            output1 = self.decoder(latent_information)
            loss1 = self.lossfn(output1, face1)
            loss1.backward()

            self.optimizer1.step()

            loss1_mean += loss1
            iterations += 1

        loss1_mean /= iterations
        loss1_mean = loss1_mean.cpu().data.numpy()
        self.scheduler1.step(loss1_mean, current_epoch)

        return loss1_mean, [face1, output1]

    def validate(self, batches):
        loss1_valid_mean = 0
        iterations = 0

        for (latent_information, face1) in batches:
            with torch.no_grad():
                face1 = face1.cuda()
                latent_information = latent_information.cuda()

                output1 = self.decoder(latent_information)
                loss1_valid_mean += self.lossfn(output1, face1)

                iterations += 1

        loss1_valid_mean /= iterations
        loss1_valid_mean = loss1_valid_mean.cpu().data.numpy()

        return [loss1_valid_mean]

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
        if torch.cuda.device_count() > 1:
            self.decoder.module.save(path / 'decoder.model')
        else:
            self.decoder.save(path / 'decoder.model')

    def load_model(self, path):
        path = Path(path)
        self.decoder.load(path / 'decoder.model')

    def log(self, logger, epoch, loss1, images, log_images=False):
        """
        use logger to log current loss etc...
        :param logger: logger used to log
        :param epoch: current epoch
        """
        logger.log_loss(epoch=epoch, loss={'lossA': float(loss1)})
        logger.log_fps(epoch=epoch)

        # log images
        if log_images:
            examples = int(len(images[0]))
            example_indices = random.sample(range(0, examples - 1), 5)
            A = []
            for idx, i in enumerate(example_indices):
                A.append(images[0].cpu()[i] * 255.00)
                A.append(images[1].cpu()[i] * 255.00)
            logger.log_images(epoch, A, "sample_input/A", 2)
        logger.save_model(epoch)

    def log_validate(self, logger, epoch, loss1):
        logger.log_loss(epoch=epoch, loss={'lossA_val': float(loss1)})
