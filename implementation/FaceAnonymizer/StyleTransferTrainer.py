import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from Logging.LoggingUtils import Logger
from configuration.general_config import MOST_RECENT_MODEL


class StyleTransferTrainer:

    def __init__(self, model_emotion, model_face, image, epochs, alpha, beta):
        self.model_emotion = model_emotion
        self.model_face = model_face
        self.image = image
        self.input_img = image.clone()
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta

        self.loss_emotion = MSELoss(self.model_emotion(self.image).detach())
        self.loss_face = MSELoss(self.model_face(self.image).detach())
        self.optimizer = optim.LBFGS([image.requires_grad_()])

    def train(self):
        print("Starting optimization")

        for i in range(self.epochs):

            def closure():
                self.input_img.data.clamp_(0, 1)
                self.optimizer.zero_grad()

                le = self.loss_emotion(self.model_emotion(self.input_img))
                lf = self.loss_face(self.model_face(self.input_img))
                loss = self.alpha * le + self.beta * lf
                loss.backward()

                if i % 50 == 0:
                    print(f"[Epoch {i}] alpha-loss: {le}, beta-loss: {lf}")

                return loss

            self.optimizer.step(closure)

        self.input_img.data.clamp_(0, 1)
        print(f"Reached {self.epochs} epochs, finishing optimization")


class MSELoss(nn.Module):

    def __init__(self, target):
        super(MSELoss, self).__init__()
        self.target = target.detach() #detach such that its not considered for autograd

    def forward(self, input):
        return F.mse_loss(input, self.target)