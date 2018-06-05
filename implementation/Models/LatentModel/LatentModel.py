import random
from abc import abstractmethod

import numpy as np
import torch
from PIL.Image import BICUBIC
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Models.ModelUtils.ModelUtils import CombinedModel


class LatentModel(CombinedModel):
    def get_models(self):
        return [self.decoder]

    def get_model_names(self):
        return ['decoder']

    def __str__(self):
        string = super().__str__()
        string += str(self.optimizer) + '\n'
        string += str(self.scheduler) + '\n'
        string += str(self.lossfn) + '\n'
        string = string.replace('\n', '\n\n')
        return string

    def __init__(self, decoder):
        self.decoder = decoder().cuda()

        self.lossfn = torch.nn.L1Loss(size_average=True).cuda()

        self.optimizer = Adam(params=self.decoder.parameters(), lr=1e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=100, cooldown=50)

    def train(self, train_data_loader, batch_size, **kwargs):
        loss_mean = 0
        face = None
        output = None
        iterations = 0

        current_epoch = kwargs.get('current_epoch', -1)

        for face, latent_information in train_data_loader:
            face = face.cuda()
            latent_information = latent_information.cuda()

            self.optimizer.zero_grad()
            output = self.decoder(latent_information)
            loss = self.lossfn(output, face)
            loss.backward()

            self.optimizer.step()

            loss_mean += loss
            iterations += 1

        loss_mean /= iterations
        loss_mean = loss_mean.cpu().data.numpy()
        self.scheduler.step(loss_mean, current_epoch)

        return loss_mean, [face, output]

    def validate(self, validation_data_loader, batch_size, **kwargs):
        loss_valid_mean = 0
        iterations = 0
        face = None
        output = None
        for face, latent_information in validation_data_loader:
            with torch.no_grad():
                face = face.cuda()
                latent_information = latent_information.cuda()

                output = self.decoder(latent_information)
                loss_valid_mean += self.lossfn(output, face)

                iterations += 1

        loss_valid_mean /= iterations
        loss_valid_mean = loss_valid_mean.cpu().data.numpy()

        return loss_valid_mean, [face, output]

    def anonymize(self, x):

        unnormalized = self.decoder(x)
        normalized = unnormalized / 2.0 + 0.5
        return normalized

    def anonymize_2(self, x):
        return self.anonymize(x)

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
            self.log_images(logger, epoch, images, validation=False)
        logger.save_model(epoch)

    def log_validation(self, logger, epoch, loss1, images):
        logger.log_loss(epoch=epoch, loss={'lossA_val': float(loss1)})
        self.log_images(logger, epoch, images, validation=True)

    @abstractmethod
    def img2latent_bridge(self, extracted_face, extracted_information):
        pass

    def log_images(self, logger, epoch, images, validation=True):

        examples = int(len(images[0]))
        example_indices = random.sample(range(0, examples - 1), 5)
        A = []
        for idx, i in enumerate(example_indices):
            A.append(images[0].cpu()[i])
            A.append(images[1].cpu()[i])

        tag = 'validation_output' if validation else 'training_output'
        logger.log_images(epoch, A, tag, 2)


class LowResAnnotationModel(LatentModel):
    def img2latent_bridge(self, extracted_face, extracted_information):
        resized_image_flat = np.array(extracted_face.resize((8, 8)))
        resized_image_flat = resized_image_flat.reshape((len(resized_image_flat), -1)) / 255.0

        landmarks_normalized_flat = np.reshape(
            (np.array(extracted_information.landmarks) / extracted_information.size_fine).tolist(), -1)

        latent_vector = np.hstack([landmarks_normalized_flat, resized_image_flat, ])
        latent_vector = torch.from_numpy(latent_vector).type(torch.float32)
        latent_vector -= 0.5
        latent_vector *= 2.0
        return latent_vector


class HistAnnotationModel(LatentModel):
    # this is deprecated
    def img2latent_bridge(self, extracted_face, extracted_information, img_size):
        hist_flat = np.array(extracted_face.resize(img_size, BICUBIC).histogram()).flatten() / (
                img_size[0] * img_size[1] * 3)
        landmarks_normalized_flat = np.reshape(
            (np.array(extracted_information.landmarks) / extracted_information.size_fine).tolist(), -1)
        latent_vector = np.append(landmarks_normalized_flat, hist_flat)

        # annotations of pic 000001.jpg
        annotations = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1,
                       1, 0, 1, 0, 1, 0, 0, 1]
        latent_vector = np.append(latent_vector, annotations)

        latent_vector = latent_vector.astype(np.float32)
        return torch.from_numpy(latent_vector).unsqueeze(0).cuda()


class HistModel(LatentModel):
    last_hist = None

    # this is deprecated
    def img2latent_bridge(self, extracted_face, extracted_information, img_size):
        hist_flat = np.array(extracted_face.resize(img_size, BICUBIC).histogram()).flatten() / (
                img_size[0] * img_size[1] * 3)
        if HistModel.last_hist is None:
            HistModel.last_hist = hist_flat
        else:
            hist_flat = HistModel.last_hist

        landmarks_normalized_flat = np.reshape(
            (np.array(extracted_information.landmarks) / extracted_information.size_fine).tolist(), -1)

        # print(landmarks_normalized_flat)
        latent_vector = np.append(landmarks_normalized_flat, hist_flat)

        latent_vector = latent_vector.astype(np.float32)
        return torch.from_numpy(latent_vector).unsqueeze(0).cuda()


class LowResModel(LatentModel):
    # only this one
    def img2latent_bridge(self, extracted_face, extracted_information):
        resized_image_flat = np.array(extracted_face.resize((8, 8))).transpose((2, 0, 1))
        resized_image_flat = resized_image_flat.reshape((1, -1)) / 255.0

        landmarks_normalized_flat = np.reshape(
            (np.array(extracted_information.landmarks) / extracted_information.size_fine), (1, -1))

        latent_vector = np.hstack([landmarks_normalized_flat, resized_image_flat, ])
        latent_vector = torch.from_numpy(latent_vector).type(torch.float32)
        latent_vector -= 0.5
        latent_vector *= 2.0
        return latent_vector.cuda()


class HistReducedModel(LatentModel):
    def img2latent_bridge(self, extracted_face, extracted_information, img_size):
        img = np.array(extracted_face.resize(img_size, BICUBIC))
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        r = np.histogram(r, bins=100, range=(0, 255), density=True)[0]
        g = np.histogram(g, bins=100, range=(0, 255), density=True)[0]
        b = np.histogram(b, bins=100, range=(0, 255), density=True)[0]
        hist_flat = np.concatenate((r, g, b))

        landmarks_normalized_flat = np.reshape(
            (np.array(extracted_information.landmarks) / extracted_information.size_fine).tolist(), -1)

        # print(landmarks_normalized_flat)
        latent_vector = np.append(landmarks_normalized_flat, hist_flat)

        latent_vector = latent_vector.astype(np.float32)
        return torch.from_numpy(latent_vector).unsqueeze(0).cuda()
