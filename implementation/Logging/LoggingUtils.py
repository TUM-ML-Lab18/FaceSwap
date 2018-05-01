import _thread
import datetime

import random
from tensorboardX import SummaryWriter
from torchvision import utils as vutils

from config import MOST_RECENT_MODEL


def log_first_layer(net, writer, frame_idx):
    first_layer = next(net.parameters()).data.cpu()

    q = vutils.make_grid(first_layer, normalize=True,
                         scale_each=True)
    writer.add_image('conv_layers/encoder', q, frame_idx)
    # this seems to be pretty slow
    # for name, param in net.named_parameters():
    #    writer.add_histogram(name, param.clone().cpu().data.numpy(), frame_idx)


class Logger:
    def __init__(self, steps_per_epoch, anonymizer, save_model_every_nth=100):
        self.loggin_path = "./logs/" + str(datetime.datetime.now())
        self.writer = SummaryWriter(self.loggin_path)
        self.steps_per_epoch = steps_per_epoch
        self.t = datetime.datetime.now()
        self.anonymizer = anonymizer
        self.save_model_every_nth = save_model_every_nth

    # def log(self, epoch, loss1, loss2, images):
    #    _thread.start_new_thread(self.log_threaded, (epoch, loss1, loss2, images))

    def log(self, epoch, loss1, loss2, images):
        new_time = datetime.datetime.now()
        self.writer.add_scalar("fps", self.steps_per_epoch * 1.0 / (new_time - self.t).total_seconds(), epoch)
        self.t = new_time

        self.writer.add_scalars("loss", {'lossA': loss1, 'lossB': loss2}, epoch)

        if images and epoch % 20 == 0:
            examples = int(len(images[0]))

            example_indices = random.sample(range(0, examples - 1), 5)  # range(examples)  #

            trump = []
            cage = []
            for i in example_indices:
                for j in range(3):
                    trump.append(images[j].cpu()[i] * 255.00)
                    cage.append(images[3 + j].cpu()[i] * 255.00)

            trump_grid = vutils.make_grid(trump, normalize=True, scale_each=True, nrow=3)
            cage_grid = vutils.make_grid(cage, normalize=True, scale_each=True, nrow=3)
            self.writer.add_image("sample_input/trump", trump_grid, epoch)
            self.writer.add_image("sample_input/cage", cage_grid, epoch)

        print(f"[Epoch {epoch}] loss1: {loss1}, loss2: {loss2}", end='\n')

        if epoch % self.save_model_every_nth == 0 and epoch > 0:
            self.anonymizer.save_model(self.loggin_path)
            self.anonymizer.save_model(MOST_RECENT_MODEL)


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='')
    # Print New Line on Complete
    if iteration == total:
        print()
