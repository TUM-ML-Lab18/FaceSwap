import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal

from Configuration.config_general import ARRAY_LANDMARKS_28_MEAN, ARRAY_LANDMARKS_28_COV
from Models.CGAN.Discriminator import Discriminator
from Models.CGAN.Generator import Generator
from Models.ModelUtils.ModelUtils import CombinedModel, norm_img
from Preprocessor.FaceExtractor import extract_landmarks


class CGAN(CombinedModel):
    """
    Standard GAN implementation but enhanced with condition on some feature
    https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
    """

    def __init__(self, **kwargs):
        # get params from kwargs
        self.z_dim = kwargs.get('z_dim', 100)
        self.y_dim = kwargs.get('y_dim', 10)
        self.img_dim = kwargs.get('img_dim', (64, 64, 3))
        path_to_lm_mean = kwargs.get('lm_mean', ARRAY_LANDMARKS_28_MEAN)
        path_to_lm_cov = kwargs.get('lm_cov', ARRAY_LANDMARKS_28_COV)
        ngf = kwargs.get('ngf', 64)
        ndf = kwargs.get('ndf', 64)
        lrG = kwargs.get('lrG', 0.0002)
        lrD = kwargs.get('lrD', 0.0002)
        beta1 = kwargs.get('beta1', 0.5)
        beta2 = kwargs.get('beta2', 0.999)

        # setup generator and discriminator
        self.G = Generator(input_dim=(self.z_dim, self.y_dim), output_dim=self.img_dim, ngf=ngf)
        self.D = Discriminator(y_dim=self.y_dim, input_dim=self.img_dim, ndf=ndf)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=lrG, betas=(beta1, beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=lrD, betas=(beta1, beta2))

        # as loss we use standard bce loss to minimize the KL-divergence -> better use the wasserstein distance
        self.BCE_loss = nn.BCELoss()

        # gaussian distribution of our landmarks
        self.landmarks_mean = np.load(path_to_lm_mean)
        self.landmarks_cov = np.load(path_to_lm_cov)
        self.landmarks_mean = torch.from_numpy(self.landmarks_mean)
        self.landmarks_cov = torch.from_numpy(self.landmarks_cov)
        self.distribution_landmarks = MultivariateNormal(loc=self.landmarks_mean.type(torch.float64),
                                                         covariance_matrix=self.landmarks_cov.type(torch.float64))

        # Fixed noise for anonymization
        self.anonym_noise = torch.randn((1, self.z_dim))
        # Fixed noise for validation
        n_val_samples = 64
        self.static_noise = torch.randn((n_val_samples, self.z_dim))
        self.static_landmarks = 2 * (self.distribution_landmarks.sample((n_val_samples,)).type(torch.float32) - 0.5)

        if torch.cuda.is_available():
            self.cuda = True
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss.cuda()
            self.static_noise = self.static_noise.cuda()
            self.static_landmarks = self.static_landmarks.cuda()

    def train(self, data_loader, batch_size, validate, **kwargs):
        current_epoch = kwargs.get('current_epoch', 100)

        # Instance noise annealing scheme
        # http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
        # shouldn't be necessary for wasserstein distance => Wasserstein distance currently only in C-PGGAN implemented
        if 0 <= current_epoch < 10:
            instance_noise_factor = 1 - (current_epoch - 0) * (1 - 0.25) / 10
        elif 10 <= current_epoch < 20:
            instance_noise_factor = 0.25 - (current_epoch - 10) * (0.25 - 0.1) / 10
        elif 20 <= current_epoch < 30:
            instance_noise_factor = 0.1 - (current_epoch - 20) * (0.1 - 0) / 10
        else:
            instance_noise_factor = 0
        print('Current epoch', current_epoch, 'instance noise factor', instance_noise_factor, 'Validation', validate)

        # sum the loss for logging
        g_loss_summed, d_loss_summed = 0, 0
        iterations = 0

        for images, features in data_loader:
            # Label vectors for loss function
            label_real, label_fake = (torch.ones(batch_size, 1, 1, 1), torch.zeros(batch_size, 1, 1, 1))
            # differentiate between validation and training
            if validate:
                noise = self.static_noise
                features = self.static_landmarks
                generated_images = self.G(noise, features)
                break
            else:
                noise = torch.randn((batch_size, self.z_dim))
                # Generate landmarks
                features_gen = 2 * (self.distribution_landmarks.sample((batch_size,)).type(torch.float32) - 0.5)

            # transfer everything to the gpu
            if self.cuda:
                label_real, label_fake = label_real.cuda(), label_fake.cuda()
                images, features, noise = images.cuda(), features.cuda(), noise.cuda()
                features_gen = features_gen.cuda()

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(noise)))
            ###########################
            self.D_optimizer.zero_grad()

            # === Train on real example with real features
            real_predictions = self.D(images + torch.randn_like(images) * instance_noise_factor, features)
            d_real_predictions_loss = self.BCE_loss(real_predictions, label_real)  # corresponds to log(D_real)
            # make backward instantly
            d_real_predictions_loss.backward()

            # === Train on real example with fake features
            fake_labels_predictions = self.D(images + torch.randn_like(images) * instance_noise_factor, features_gen)
            d_fake_labels_loss = self.BCE_loss(fake_labels_predictions, label_fake) / 2
            # make backward instantly
            d_fake_labels_loss.backward()

            # === Train on fake example from generator
            generated_images = self.G(noise, features_gen)
            fake_images_predictions = self.D(generated_images.detach() + torch.randn_like(generated_images) *
                                             instance_noise_factor, features_gen)
            d_fake_images_loss = self.BCE_loss(fake_images_predictions, label_fake) / 2  # corresponds to log(1-D_fake)
            # make backward instantly
            d_fake_images_loss.backward()

            d_loss = d_real_predictions_loss + d_fake_labels_loss + d_fake_images_loss
            self.D_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(noise)))
            ###########################
            self.G_optimizer.zero_grad()

            # === Train on fooling the Discriminator
            fake_images_predictions = self.D(generated_images + torch.randn_like(generated_images) *
                                             instance_noise_factor, features_gen)
            g_loss = self.BCE_loss(fake_images_predictions, label_real)

            g_loss.backward()
            self.G_optimizer.step()

            # losses
            g_loss_summed += float(g_loss)
            d_loss_summed += float(d_loss)
            iterations += 1

        if not validate:
            g_loss_summed /= iterations
            d_loss_summed /= iterations
            log_info = {'loss': {'lossG': g_loss_summed,
                                 'lossD': d_loss_summed}}
            log_img = generated_images + torch.randn_like(generated_images) * instance_noise_factor
        else:
            log_info = {}
            log_img = generated_images

        return log_info, log_img

    def get_modules(self):
        return [self.G, self.D]

    def get_model_names(self):
        return ['generator', 'discriminator']

    def get_remaining_modules(self):
        return [self.G_optimizer, self.D_optimizer, self.BCE_loss]

    def anonymize(self, extracted_face, extracted_information):
        # ===== Landmarks
        # Normalize landmarks
        landmarks = np.array(extracted_information.landmarks) / extracted_information.size_fine
        landmarks = landmarks.reshape(-1)
        # Extract needed landmarks
        landmarks = extract_landmarks(landmarks, n=10)

        # ===== Creating feature vector
        feature = landmarks
        feature = torch.from_numpy(feature).type(torch.float32)

        # ===== Zero centering
        feature -= 0.5
        feature *= 2.0
        if self.cuda:
            self.anonym_noise, feature = self.anonym_noise.cuda(), feature.cuda()
        tensor_img = self.G(self.anonym_noise, feature)

        # ===== Denormalize generated image
        for t in tensor_img:  # loop over mini-batch dimension
            norm_img(t)
        tensor_img *= 255
        tensor_img = tensor_img.type(torch.uint8)
        return tensor_img

    def log_images(self, logger, epoch, images, validation=True):
        tag = 'validation_output' if validation else 'training_output'
        logger.log_images(epoch, images, tag, 8)
