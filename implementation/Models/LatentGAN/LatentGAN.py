import random

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Models.LatentModel.Decoder import LatentDecoder
from Models.ModelUtils.ModelUtils import CombinedModel
from .Discriminator import Discriminator


class LatentGAN(CombinedModel):

    def __init__(self, **kwargs):
        self.input_dim = kwargs.get('input_dim', 100)
        self.alpha = kwargs['alpha']
        self.z_dim = kwargs['z_dim']
        self.img_dim = kwargs['img_dim']
        self.ndf = kwargs['ndf']
        self.lrD = kwargs['lrD']

        self.decoder = LatentDecoder(self.input_dim)
        self.discriminator = Discriminator(input_dim=self.img_dim, ndf=self.ndf)

        self.l1_loss = torch.nn.L1Loss(size_average=True).cuda()
        self.bce_loss = torch.nn.BCELoss()

        self.dec_optimizer = Adam(params=self.decoder.parameters(), lr=1e-4)
        self.dec_scheduler = ReduceLROnPlateau(self.dec_optimizer, patience=100, cooldown=50)
        self.disc_optimizer = Adam(params=self.discriminator.parameters(), lr=self.lrD)

        if torch.cuda.is_available():
            self.cuda = True
            self.decoder.cuda()
            self.discriminator.cuda()
            self.l1_loss.cuda()
            self.bce_loss.cuda()

    def train(self, train_data_loader, batch_size, validate, **kwargs):
        g_l1_loss_mean = 0
        d_loss_mean = 0
        faces = None
        output = None
        iterations = 0

        current_epoch = kwargs.get('current_epoch', -1)

        for faces, latent_information in train_data_loader:
            label_real, label_fake = (torch.ones(batch_size, 1, 1, 1), torch.zeros(batch_size, 1, 1, 1))
            if self.cuda:
                label_real, label_fake = label_real.cuda(), label_fake.cuda()
                faces = faces.cuda()
                latent_information = latent_information.cuda()

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            if not validate:
                self.disc_optimizer.zero_grad()

            # Train on real examples
            real_predictions = self.discriminator(faces)
            d_real_predictions_loss = self.bce_loss(real_predictions, label_real)

            if not validate:
                d_real_predictions_loss.backward()

            # Train on fake examples from encoder
            generated = self.decoder(latent_information)
            fake_predictions = self.discriminator(generated.detach())
            d_fake_predictions_loss = self.bce_loss(fake_predictions, label_fake)

            if not validate:
                d_fake_predictions_loss.backward()

            d_overall_loss = d_real_predictions_loss + d_fake_predictions_loss

            ############################
            # (2) Update encoder network
            ###########################
            if not validate:
                self.dec_optimizer.zero_grad()

            # L1 loss for image reconstruction
            output = self.decoder(latent_information)
            g_l1_loss = self.l1_loss(output, faces) * self.alpha

            if not validate:
                g_l1_loss.backward()

            fake_predictions = self.discriminator(generated)
            g_fake_predictions_loss = self.bce_loss(fake_predictions, label_real) * (1 - self.alpha)

            if not validate:
                g_fake_predictions_loss.backward()
                self.dec_optimizer.step()

            d_loss_mean += d_overall_loss
            g_l1_loss_mean += g_l1_loss
            iterations += 1

        d_loss_mean /= iterations
        g_l1_loss_mean /= iterations
        d_loss_mean = d_loss_mean.cpu().data.numpy()
        g_l1_loss_mean = g_l1_loss_mean.cpu().data.numpy()

        if not validate:
            self.dec_scheduler.step(g_l1_loss_mean, current_epoch)

        if not validate:
            log_info = {'loss': {'g_l1_loss': float(g_l1_loss_mean), 'disc_loss': d_loss_mean}}
        else:
            log_info = {'loss': {'g_l1_loss_val': float(g_l1_loss_mean), 'disc_loss_val': d_loss_mean}}

        return log_info, [faces, output]

    def anonymize(self, extracted_face, extracted_information):
        resized_image_flat = np.array(extracted_face.resize((8, 8))).transpose((2, 0, 1))

        resized_image_flat = resized_image_flat.reshape((1, -1)) / 255.0

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

    def get_models(self):
        return [self.discriminator, self.decoder]

    def get_model_names(self):
        return ['discriminator', 'decoder']

    def get_remaining_modules(self):
        return [self.dec_optimizer, self.disc_optimizer, self.dec_scheduler, self.l1_loss, self.bce_loss]

    def log_images(self, logger, epoch, images, validation=True):
        examples = int(len(images[0]))
        example_indices = random.sample(range(0, examples - 1), 5)
        A = []
        for idx, i in enumerate(example_indices):
            A.append(images[0].cpu()[i])
            A.append(images[1].cpu()[i])

        tag = 'validation_output' if validation else 'training_output'
        logger.log_images(epoch, A, tag, 2)
