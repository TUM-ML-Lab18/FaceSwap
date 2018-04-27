import datetime

import torch
import numpy as np
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
    output = output[0].cpu().numpy() # Shift from to CPU into a Numpy array
    output = (output*255.0).astype(np.uint8) # Transform back into valid image range
    output = output.transpose(1, 2, 0) # Resort dimension channels: CHW -> HWC
    output = output[:,:,::-1] # Resort color channels: BGR -> RGB
    return output


class Logger:
    def __init__(self):
        self.writer = SummaryWriter("./logs/" + str(datetime.datetime.now()))

    def log(self, i, loss1, loss2, autoencoder, images):
        loss1 = loss1.cpu().data.numpy()
        loss2 = loss2.cpu().data.numpy()

        self.writer.add_scalars("loss", {'lossA': loss1, 'lossB': loss2}, i)

        # log_first_layer(autoencoder, self.writer, i)

        if images and i % 100 == 0:
            for idx, img in enumerate(images):
                images[idx] = tensor2img(img)
            stacked = torch.cat(images)
            grid = vutils.make_grid(stacked.data, normalize=True, scale_each=True, nrow=3)
            self.writer.add_image("sample_input", grid, i)

        print(f"[Epoch {i}] loss1: {loss1}, loss2: {loss2}", end='\n')
