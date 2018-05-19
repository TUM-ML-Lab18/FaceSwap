import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
from Logging.LoggingUtils import Logger
from configuration.general_config import MOST_RECENT_MODEL


class StyleTransferTrainer:

    def __init__(self, image, epochs, alpha, beta):
        self.image = image
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta

        self.optimizer = optim.LBFGS([image.requires_grad_()])

    def train(self):
        print("Starting training")
        for i in range(self.epochs):

            def closure():
                self.optimizer.zero_grad()
                #compute and return loss

            self.optimizer.step(closure)

        print("Finished training")


class Loss(nn.Module):

    def __init__(self, target):
        super(Loss, self).__init__()
        self.target = target.detach() #detach such that its not considered for autograd

    def forward(self, *input):
        self.loss = F.mse_loss(input, self.target)
        return input