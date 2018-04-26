import datetime

import torch
from tensorboardX import SummaryWriter
from torchvision import utils as vutils


def log_first_layer(net, writer, frame_idx):
    first_layer = next(net.parameters()).data.cpu()

    q = vutils.make_grid(first_layer, normalize=True,
                         scale_each=True)
    writer.add_image('conv_layers/encoder', q, frame_idx)
    for name, param in net.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), frame_idx)


def tensor2img(output):
    output = output.cpu()[0] * 255.0
    index = torch.LongTensor([2, 0, 1])
    output[index] = output
    # output = output[::-1, :, :]
    return output


class Logger:
    def __init__(self):
        self.writer = SummaryWriter("./logs/" + str(datetime.datetime.now()))

    def log(self, i, loss1, loss2, autoencoder, input, output, target):
        loss1 = loss1.cpu().data.numpy()
        loss2 = loss2.cpu().data.numpy()

        self.writer.add_scalars("loss", {'lossA': loss1, 'lossB': loss2}, i)

        log_first_layer(autoencoder, self.writer, i)

        input = torch.unsqueeze(tensor2img(input), 0)
        output = torch.unsqueeze(tensor2img(output), 0)
        target = torch.unsqueeze(tensor2img(target), 0)

        stacked = torch.cat([input, output, target])

        grid = vutils.make_grid(stacked.data, normalize=True, scale_each=True, )

        self.writer.add_image("sample_input", grid, i)

        print(f"[Epoch {i}] loss1: {loss1}, loss2: {loss2}", end='\n')