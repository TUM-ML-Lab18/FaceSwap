import datetime
import json
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


class Logger:
    def __init__(self, steps_per_epoch, model, save_model_every_nth=100, shared_model_path='.'):
        self.shared_model_path = shared_model_path
        self.loggin_path = "./logs/" + str(datetime.datetime.now())
        self.writer = SummaryWriter(self.loggin_path)
        self.steps_per_epoch = steps_per_epoch
        self.t = datetime.datetime.now()
        self.model = model
        self.save_model_every_nth = save_model_every_nth

    def log_loss(self, epoch, loss: dict = None):
        """
        logs the loss of a model
        :param epoch: current epoch
        :param loss: dict containing different loss values (i.e.: {'lossA': 10, 'lossB': 12}
        """
        if loss is None:
            loss = {}
        self.writer.add_scalars("loss", loss, epoch)
        print(f"epoch: {epoch}"+json.dumps(loss), end='\n')

    def log_fps(self, epoch):
        """
        logs the current fps
        :param epoch: current epoch
        """
        new_time = datetime.datetime.now()
        self.writer.add_scalar("fps", self.steps_per_epoch * 1.0 / (new_time - self.t).total_seconds(), epoch)
        self.t = new_time

    def log_images(self, epoch, images, tag_name, columns):
        """
        logs images to the tensorboard
        :param epoch: current epoch
        :param images: list of images
        :param tag_name: tag_name of images in tensorboard
        :param columns: number of columns to display the images
        """
        grid = vutils.make_grid(images, normalize=False, scale_each=True, nrow=columns)
        self.writer.add_image(tag_name, grid, epoch)

    def save_model(self, epoch):
        if epoch % self.save_model_every_nth == 0:# and epoch > 0:
            self.model.save_model(self.loggin_path)
            self.model.save_model(self.shared_model_path)

    def log_config(self, config):
        text = f"batchsize: {config['batch_size']}\n\nnum_gpus: {torch.cuda.device_count()}"
        self.writer.add_text("hyperparameters",
                             text)
        text = str(self.model)
        self.writer.add_text("config", text)

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
