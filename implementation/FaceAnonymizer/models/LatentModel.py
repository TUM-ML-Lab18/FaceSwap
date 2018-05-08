import torch
from torch.nn import DataParallel


class LatentModel:
    def __init__(self, optimizer, scheduler, data_loader, decoder, loss_function):
        self.data_loader = data_loader
        self.decoder = decoder().cuda()

        if torch.cuda.device_count() > 1:
            self.decoder = DataParallel(self.decoder)

        self.lossfn = loss_function.cuda()

        self.optimizer1 = optimizer(self.decoder.parameters())
        self.scheduler1 = scheduler(self.optimizer1)

    def train(self, current_epoch):
        loss1_mean, loss2_mean = 0, 0
        face1 = None
        output1 = None
        face2 = None
        output2 = None
        iterations = 0

        for (face1_landmarks, face1), (face2_landmarks, face2) in self.data_loader:
            # face1 and face2 contain a batch of images of the first and second face, respectively
            face1, face2 = face1.cuda(), face2.cuda()
            face1_landmarks, face2_landmarks = face1_landmarks.cuda(), face2_landmarks.cuda()

            self.optimizer1.zero_grad()
            output1 = self.decoder(face1_landmarks)
            loss1 = self.lossfn(output1, face1)
            loss1.backward()

            #output2 = self.decoder(face2_landmarks)
            #loss2 = self.lossfn(output2, face2)
            #loss2.backward()

            self.optimizer1.step()

            loss1_mean += loss1
            iterations += 1

        loss1_mean /= iterations
        loss1_mean = loss1_mean.cpu().data.numpy()
        loss2_mean = 0
        self.scheduler1.step(loss1_mean, current_epoch)

        return loss1_mean, loss2_mean, [face1, output1, face1, face1, output1, face1]

    def anonymize(self, x):
        return x

    def anonymize_2(self, x):
        return x