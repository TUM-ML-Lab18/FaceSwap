import random
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Models.ModelUtils.ModelUtils import CombinedModel
from Models.LatentModel.Decoder import LatentDecoder
from .Discriminator import Discriminator


class LatentGAN(CombinedModel):

    def __init__(self, **kwargs):

        self.input_dim = kwargs.get('input_dim', 100)
        # TODO: parse remaining arguments

        self.decoder = LatentDecoder(self.input_dim)
        self.discriminator = Discriminator()

        self.l1_loss = torch.nn.L1Loss(size_average=True).cuda()
        self.bce_loss = torch.nn.BCELoss()

        self.dec_optimizer = Adam(params=self.decoder.parameters(), lr=1e-4)
        self.dec_scheduler = ReduceLROnPlateau(self.dec_optimizer, patience=100, cooldown=50)
        self.disc_optimizer = Adam(params=self.discriminator.parameters(), lr=1e-4)

        if torch.cuda.is_available():
            self.cuda = True
            self.decoder.cuda()
            self.discriminator.cuda()
            self.l1_loss.cuda()
            self.bce_loss.cuda()

    def train(self, train_data_loader, batch_size, validate, **kwargs):
        loss_mean = 0
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

            ############################
            # (1) Update G network (self.decoder)
            ###########################
            if not validate:
                self.dec_optimizer.zero_grad()

            output = self.decoder(latent_information)
            l1_loss = self.l1_loss(output, faces)

            if not validate:
                l1_loss.backward()
                self.dec_optimizer.step()

            loss_mean += loss
            iterations += 1

        loss_mean /= iterations
        loss_mean = loss_mean.cpu().data.numpy()

        if not validate:
            self.scheduler.step(loss_mean, current_epoch)

        if not validate:
            log_info = {'loss': float(loss_mean)}
        else:
            log_info = {'loss_val': float(loss_mean)}

        return log_info, [face, output]

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