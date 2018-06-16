import random

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Models.ModelUtils.ModelUtils import CombinedModel


class LatentModel(CombinedModel):
    def __init__(self, decoder):
        self.decoder = decoder().cuda()

        self.lossfn = torch.nn.L1Loss(size_average=True).cuda()

        self.optimizer = Adam(params=self.decoder.parameters(), lr=1e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=100, cooldown=50)

    def train(self, train_data_loader, batch_size, validate, **kwargs):
        loss_mean = 0
        face = None
        output = None
        iterations = 0

        current_epoch = kwargs.get('current_epoch', -1)

        for face, latent_information in train_data_loader:
            face = face.cuda()
            latent_information = latent_information.cuda()

            if not validate:
                self.optimizer.zero_grad()

            output = self.decoder(latent_information)
            loss = self.lossfn(output, face)

            if not validate:
                loss.backward()
                self.optimizer.step()

            loss_mean += float(loss)
            iterations += 1

        loss_mean /= iterations
        loss_mean = loss_mean.cpu().data.numpy()

        if not validate:
            self.scheduler.step(loss_mean, current_epoch)

        if not validate:
            log_info = {'loss': {'loss': loss_mean}}
        else:
            log_info = {'loss': {'loss_val': loss_mean}}

        return log_info, [face, output]

    def get_models(self):
        return [self.decoder]

    def get_model_names(self):
        return ['decoder']

    def get_remaining_modules(self):
        return [self.optimizer, self.scheduler, self.lossfn]

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
    STATIC_NOISE = np.random.randn(8 * 8 * 3) * 0.05

    def anonymize(self, extracted_face, extracted_information):
        resized_image_flat = np.array(extracted_face.resize((8, 8))).transpose((2, 0, 1))

        resized_image_flat = resized_image_flat.reshape((1, -1)) / 255.0  # + self.STATIC_NOISE

        landmarks_normalized_flat = np.reshape(
            (np.array(extracted_information.landmarks) / extracted_information.size_fine), (1, -1))

        latent_vector = np.hstack([landmarks_normalized_flat, resized_image_flat, ])
        latent_vector = torch.from_numpy(latent_vector).type(torch.float32)
        latent_vector -= 0.5
        latent_vector *= 2.0

        latent_vector = latent_vector.cuda()

        unnormalized = self.decoder(latent_vector)
        normalized = unnormalized / 2.0 + 0.5
        return normalized


class RetrainLowResModel(LowResModel):
    def __init__(self, decoder, model_path):
        super().__init__(decoder)
        self.load_model(model_path)
