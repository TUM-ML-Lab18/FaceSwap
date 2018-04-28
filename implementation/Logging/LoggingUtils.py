import datetime
from random import randint

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
            rows = int(len(images) / 3)
            processed_images = []
            for i in range(rows):
                rand = randint(0, len(images[0]) - 1)
                for j in range(3):
                    processed_images.append(images[i*3+j].cpu()[rand]*255.0)
            grid = vutils.make_grid(processed_images, normalize=True, scale_each=True, nrow=3)
            self.writer.add_image("sample_input", grid, i)

        print(f"[Epoch {i}] loss1: {loss1}, loss2: {loss2}", end='\n')
