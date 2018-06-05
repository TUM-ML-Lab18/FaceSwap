import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal

from Configuration.config_general import ARRAY_CELEBA_LANDMARKS_MEAN, ARRAY_CELEBA_LANDMARKS_COV
from Models.CGAN.Discriminator import Discriminator
from Models.CGAN.Generator import Generator
from Models.ModelUtils.ModelUtils import CombinedModel


class CGAN(CombinedModel):
    def __init__(self, **kwargs):
        self.z_dim = kwargs.get('z_dim', 100)
        self.y_dim = kwargs.get('y_dim', 10)
        self.img_dim = kwargs.get('img_dim', (64, 64, 3))
        path_to_y_mean = kwargs.get('y_mean', ARRAY_CELEBA_LANDMARKS_MEAN)
        path_to_y_cov = kwargs.get('y_cov', ARRAY_CELEBA_LANDMARKS_COV)
        lrG = kwargs.get('lrG', 0.0002)
        lrD = kwargs.get('lrD', 0.0002)
        beta1 = kwargs.get('beta1', 0.5)
        beta2 = kwargs.get('beta2', 0.999)

        # self.G = LatentDecoderGAN(input_dim=self.z_dim + self.y_dim)
        self.G = Generator(input_dim=(self.z_dim, self.y_dim), output_dim=self.img_dim, ngf=64)
        self.D = Discriminator(y_dim=self.y_dim, input_dim=self.img_dim, ndf=64)

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=lrG, betas=(beta1, beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=lrD, betas=(beta1, beta2))

        self.BCE_loss = nn.BCELoss()

        # gaussian distribution of our landmarks
        self.landmarks_mean = torch.from_numpy(np.load(path_to_y_mean))
        self.landmarks_cov = torch.from_numpy(np.load(path_to_y_cov))
        self.distribution = MultivariateNormal(loc=self.landmarks_mean.type(torch.float64),
                                               covariance_matrix=self.landmarks_cov.type(torch.float64))

        if torch.cuda.is_available():
            self.cuda = True
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss.cuda()

    def __str__(self):
        string = super().__str__()
        string += str(self.G_optimizer) + '\n'
        string += str(self.D_optimizer) + '\n'
        string += str(self.BCE_loss) + '\n'
        string = string.replace('\n', '\n\n')
        return string

    def _train(self, data_loader, batch_size, **kwargs):
        # indicates if the graph should get updated
        validate = kwargs.get('validate', False)

        # sum the loss for logging
        g_loss_summed, d_loss_summed = 0, 0
        iterations = 0

        # Label vectors for loss function
        label_real, label_fake = (torch.ones(batch_size, 1, 1, 1), torch.zeros(batch_size, 1, 1, 1))
        if self.cuda:
            label_real, label_fake = label_real.cuda(), label_fake.cuda()

        for images, features in data_loader:
            # generate random vector
            z = torch.randn((batch_size, self.z_dim))

            # transfer everything to the gpu
            if self.cuda:
                images, features, z = images.cuda(), features.cuda(), z.cuda()

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            if not validate:
                self.D_optimizer.zero_grad()

            # Train on real example with real features
            real_predictions = self.D(images, features)
            d_real_predictions_loss = self.BCE_loss(real_predictions,
                                                    label_real)  # real corresponds to log(D_real)

            # if not validate:
            #     # make backward instantly
            #     d_real_predictions_loss.backward()

            # Train on fake example from generator
            generated_images = self.G(z, features)
            fake_images_predictions = self.D(generated_images.detach(),
                                             features)  # todo what happens if we detach the output of the Discriminator
            d_fake_images_loss = self.BCE_loss(fake_images_predictions,
                                               label_fake)  # face corresponds to log(1-D_fake)

            # if not validate:
            #     # make backward instantly
            #     d_fake_images_loss.backward()

            d_loss = d_real_predictions_loss + d_fake_images_loss

            if not validate:
                d_loss.backward()
                self.D_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            if not validate:
                self.G_optimizer.zero_grad()

            # Train on fooling the Discriminator
            generated_images = self.G(z, features)
            fake_images_predictions = self.D(generated_images, features)
            g_loss = self.BCE_loss(fake_images_predictions, label_real)

            if not validate:
                g_loss.backward()
                self.G_optimizer.step()

            # losses
            g_loss_summed += g_loss
            d_loss_summed += d_loss
            iterations += 1

        g_loss_summed /= iterations
        d_loss_summed /= iterations

        return g_loss_summed.cpu().data.numpy(), d_loss_summed.cpu().data.numpy(), generated_images

    def train(self, train_data_loader, batch_size, **kwargs):
        g_loss, d_loss, generated_images = self._train(train_data_loader, batch_size, validate=False, **kwargs)
        return g_loss, d_loss, generated_images

    def validate(self, validation_data_loader, batch_size, **kwargs):
        g_loss, d_loss, generated_images = self._train(validation_data_loader, batch_size, validate=True, **kwargs)
        return g_loss, d_loss, generated_images

    def get_models(self):
        return [self.G, self.D]

    def get_model_names(self):
        return ['generator', 'discriminator']

    def log(self, logger, epoch, lossG, lossD, images, log_images=False):  # last parameter is not needed anymore
        """
        use logger to log current loss etc...
        :param logger: logger used to log
        :param epoch: current epoch
        """
        logger.log_loss(epoch=epoch, loss={'lossG': float(lossG), 'lossD': float(lossD)})
        logger.log_fps(epoch=epoch)
        logger.save_model(epoch)

        if log_images:
            self.log_images(logger, epoch, images, validation=False)

    def log_validation(self, logger, epoch, lossG, lossD, images):
        logger.log_loss(epoch=epoch, loss={'lossG_val': float(lossG), 'lossD_val': float(lossD)})

        self.log_images(logger, epoch, images, validation=True)

    def log_images(self, logger, epoch, images, validation=True):
        # images = images.cpu()
        # images *= .5
        # images += .5
        # examples = int(len(images))
        # example_indices = random.sample(range(0, examples - 1), 4 * 4)
        # A = []
        # for idx, i in enumerate(example_indices):
        #     A.append(images[i])
        tag = 'validation_output' if validation else 'training_output'
        logger.log_images(epoch, images, tag, 8)

    def anonymize(self, x):
        # z = torch.ones((x.shape[0], self.z_dim, 1, 1)).cuda() * 0.25
        z = torch.randn((x.shape[0], self.z_dim, 1, 1)).cuda()
        return self.G(z, x) * 0.5 + 0.5

    def img2latent_bridge(self, extracted_face, extracted_information):
        landmarks_normalized_flat = np.reshape(
            (np.array(extracted_information.landmarks) / extracted_information.size_fine).tolist(), -1).astype(
            np.float32)
        # landmarks_5
        landmarks_X = landmarks_normalized_flat[::2]
        landmarks_Y = landmarks_normalized_flat[1::2]
        eye_left_X = np.mean(landmarks_X[36:42])
        eye_left_Y = np.mean(landmarks_Y[36:42])
        eye_right_X = np.mean(landmarks_X[42:48])
        eye_right_Y = np.mean(landmarks_Y[42:48])
        nose_X = np.mean(landmarks_X[31:36])
        nose_Y = np.mean(landmarks_Y[31:36])
        mouth_left_X = landmarks_X[48]
        mouth_left_Y = landmarks_Y[48]
        mouth_right_X = landmarks_X[60]
        mouth_right_Y = landmarks_Y[60]
        landmarks_5 = np.vstack((eye_left_X, eye_left_Y, eye_right_X, eye_right_Y, nose_X, nose_Y, mouth_left_X,
                                 mouth_left_Y, mouth_right_X, mouth_right_Y)).T

        mean = np.array([0.18269604, 0.2612222, 0.5438053, 0.2612222, 0.28630707,
                         0.5341739, 0.18333682, 0.70732147, 0.45070747, 0.69178724]).astype(np.float32)

        # return torch.from_numpy(mean).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).cuda()
        return torch.from_numpy(landmarks_5).unsqueeze(-1).unsqueeze(-1).cuda()
        # return torch.from_numpy(landmarks_normalized_flat).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).cuda()
