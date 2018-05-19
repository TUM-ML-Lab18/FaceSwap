import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy


class StyleTransfer:

    def __init__(self, emotion_model, face_model):
        self.emotion_model = emotion_model
        self.face_model = face_model

    def train(self):
        pass

    def validate(self):
        pass

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass


class Loss(nn.Module):

    def __init__(self, target):
        super(Loss, self).__init__()
        self.target = target.detach() #detach such that its not considered for autograd

    def forward(self, *input):
        self.loss = F.mse_loss(input, self.target)
        return input