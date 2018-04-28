import datetime

import torch
from tensorboardX import SummaryWriter
from torchvision import utils as vutils


def log_first_layer(net, writer, frame_idx):
    first_layer = next(net.parameters()).data.cpu()

    q = vutils.make_grid(first_layer, normalize=True,
                         scale_each=True)
    writer.add_image('conv_layers/encoder', q, frame_idx)
    # this seems to be pretty slow
    # for name, param in net.named_parameters():
    #    writer.add_histogram(name, param.clone().cpu().data.numpy(), frame_idx)


def tensor2img(output):
    output = output.cpu()[0] * 255.0
    inv_idx = torch.arange(output.size(0) - 1, -1, -1).long()
    output = output[inv_idx]
    return output


class Logger:
    def __init__(self, steps_per_epoch):
        self.writer = SummaryWriter("./logs/" + str(datetime.datetime.now()))
        self.steps_per_epoch = steps_per_epoch
        self.t = datetime.datetime.now()

    def log(self, i, loss1, loss2, images):
        new_time = datetime.datetime.now()
        self.writer.add_scalar("fps", self.steps_per_epoch * 1.0 / (new_time - self.t).total_seconds(), i)
        self.t = new_time

        self.writer.add_scalars("loss", {'lossA': loss1, 'lossB': loss2}, i)

        if images and i % 20 == 0:
            images = list(map(tensor2img, images))
            grid = vutils.make_grid(images, normalize=True, scale_each=True, nrow=3)
            self.writer.add_image("sample_input", grid, i)

        print(f"[Epoch {i}] loss1: {loss1}, loss2: {loss2}", end='\n')
