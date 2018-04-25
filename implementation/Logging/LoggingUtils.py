import datetime

import torch
from tensorboardX import SummaryWriter
from torchvision import utils as vutils


def log_first_layer(net, writer, frame_idx):
    q = vutils.make_grid(torch.cat(torch.split(next(net.parameters()).data.cpu(), 1, 1), 0), normalize=True,
                         scale_each=True)
    writer.add_image('conv_layers/encoder', q, frame_idx)
    for name, param in net.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), frame_idx)

def tensor2img(output):
    output = output.cpu()[0]*255.0
    return output

class Logger:
    def __init__(self):
        self.writer = SummaryWriter("./logs/" + str(datetime.datetime.now()))

    def log(self, i, loss1, loss2, autoencoder, input, output, target):
        loss1 = loss1.cpu().data.numpy()
        loss2 = loss2.cpu().data.numpy()

        self.writer.add_scalar("loss/A", loss1, i)
        self.writer.add_scalar("loss/B", loss2, i)

        log_first_layer(autoencoder, self.writer, i)

        self.writer.add_image("img/input", tensor2img(input), i)
        self.writer.add_image("img/output", tensor2img(output), i)
        self.writer.add_image("img/target", tensor2img(target), i)

        print(f"[Epoch {i}] loss1: {loss1}, loss2: {loss2}", end='\n')
