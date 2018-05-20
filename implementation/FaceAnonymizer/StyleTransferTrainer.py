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

    def __init__(self, model_emotion, model_face, input_img, steps, alpha, beta):
        self.model_emotion = model_emotion
        self.model_face = model_face
        self.input_img = input_img
        self.result_img = torch.randn(input_img.data.size()) #input_img.clone()
        self.steps = steps
        self.alpha = alpha
        self.beta = beta

        self.loss_emotion = MSELoss(self.model_emotion(self.input_img).detach())
        self.loss_face = MSELoss(self.model_face(self.input_img).detach())
        self.optimizer = optim.LBFGS([self.result_img.requires_grad_()])

    def train(self):
        print("Starting optimization")
        i = [0]

        while i[0] < self.steps:

            def closure():
                self.result_img.data.clamp_(0, 1)
                self.optimizer.zero_grad()

                le = self.loss_emotion(self.model_emotion(self.result_img))
                lf = self.loss_face(self.model_face(self.result_img))
                loss = self.alpha * le + self.beta * lf
                loss.backward()

                if i[0] % 5 == 0:
                    print(f"[Step {i[0]}] alpha-loss: {le}, beta-loss: {lf}")
                i[0] += 1

                return loss

            self.optimizer.step(closure)

        self.result_img.data.clamp_(0, 1)
        print(f"Reached {self.steps} steps, finishing optimization")


class MSELoss(nn.Module):

    def __init__(self, target):
        super(MSELoss, self).__init__()
        self.target = target.detach() #detach such that its not considered for autograd

    def forward(self, input):
        return F.mse_loss(input, self.target)