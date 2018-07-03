import random

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Models.ModelUtils.ModelUtils import CombinedModel
from Preprocessor.FaceExtractor import normalize_landmarks, extract_lowres


class LatentModel(CombinedModel):
    """
    This model is designed to take some latent information as input and feed it through a generator.
    The output of this generator is then compared with the corresponding image (that was used to extract that latent
    information) via a L1 loss. This L1 loss is then used to update the parameters.
    """

    def __init__(self, **kwargs):
        # the decoder aka generator
        self.decoder = kwargs.get('decoder')()
        lr = kwargs.get('lr', 1e-4)

        self.loss = torch.nn.L1Loss(size_average=True)

        if torch.cuda.is_available():
            self.cuda = True
            self.decoder.cuda()
            self.loss.cuda()

        self.optimizer = Adam(params=self.decoder.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=100, cooldown=50)

    def train(self, train_data_loader, batch_size, validate, **kwargs):
        loss_mean = 0
        face = None
        output = None
        iterations = 0

        current_epoch = kwargs.get('current_epoch', -1)

        for face, latent_information in train_data_loader:
            if self.cuda:
                face = face.cuda()
                latent_information = latent_information.cuda()

            if not validate:
                self.optimizer.zero_grad()

            # just feed in the latent information and calculate the loss
            output = self.decoder(latent_information)
            loss = self.loss(output, face)

            if not validate:
                loss.backward()
                self.optimizer.step()

            loss_mean += float(loss)
            iterations += 1

        loss_mean /= iterations

        if not validate:
            self.scheduler.step(loss_mean, current_epoch)

        if not validate:
            log_info = {'loss': {'loss': loss_mean}}
        else:
            log_info = {'loss': {'loss_val': loss_mean}}

        return log_info, [face, output]

    def get_modules(self):
        return [self.decoder]

    def get_model_names(self):
        return ['decoder']

    def get_remaining_modules(self):
        return [self.optimizer, self.scheduler, self.loss]

    def log_images(self, logger, epoch, images, validation=True):

        examples = int(len(images[0]))
        example_indices = random.sample(range(0, examples - 1), 5)
        A = []
        for idx, i in enumerate(example_indices):
            A.append(images[0].cpu()[i])
            A.append(images[1].cpu()[i])

        tag = 'validation_output' if validation else 'training_output'
        logger.log_images(epoch, A, tag, 2)


class LowResModel(LatentModel):
    def anonymize(self, extracted_face, extracted_information):
        # ===== Landmarks
        # Normalize landmarks
        landmarks = normalize_landmarks(extracted_information).reshape((1, -1))

        # ===== LowRes pixelmap
        lowres = extract_lowres(extracted_face, resolution=8)

        # ===== Merge latent features
        latent_vector = torch.from_numpy(np.hstack((landmarks, lowres))).type(torch.float32).reshape((1, -1))

        # ===== Zero centering
        latent_vector -= 0.5
        latent_vector *= 2.0

        if self.cuda:
            latent_vector = latent_vector.cuda()

        unnormalized = self.decoder(latent_vector)
        normalized = unnormalized / 2.0 + 0.5
        return normalized


class RetrainLowResModel(LowResModel):
    """
    Just a simple model to retrain a LowResModel
    """

    def __init__(self, decoder, model_path):
        super().__init__(decoder)
        self.load_model(model_path)
