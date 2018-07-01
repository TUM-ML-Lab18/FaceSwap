import datetime
import inspect
import json

import torch
from tensorboardX import SummaryWriter
from torchvision import utils as vutils


class Logger:
    """
    wrapper for the tensorboardx | does some higher level logging
    """

    def __init__(self, steps_per_epoch, model, save_model_every_nth=100, shared_model_path='.'):
        self.shared_model_path = shared_model_path
        self.loggin_path = "./logs/" + str(datetime.datetime.now())
        self.writer = SummaryWriter(self.loggin_path)
        self.steps_per_epoch = steps_per_epoch
        self.t = datetime.datetime.now()
        self.model = model
        self.save_model_every_nth = save_model_every_nth

    def log_values(self, epoch, values: dict = None):
        """
        logs the loss of a model, can also log arbitrary values but if the dict contains a key named 'loss' the value
        of this key is printed
        :param epoch: current epoch
        :param values: dict containing different dicts containing values (i.e.: {'loss':{'lossA': 10, 'lossB': 12}, ...}
        If the value of a key is a dict itself use the add_scalars function to log all these values to one graph.
        Be aware that this leads to multiple entries in the runs section of the tensorboard.
        Otherwise log the information into a separate graph.
        PGGAN logging is a good example:
        {'loss': {'lossG': g_loss_summed, ####These values are logged into one graph and printed to the commandline
                  'lossD': d_loss_summed},
        'info/WassersteinDistance': wasserstein_d_summed, ####These values are all logged to different graphs
        'info/eps': eps_summed,
        'info/FadeInFactor': fade_in_factor,
        'info/Level': self.resolution_level,
        'info/curr_level': cur_level}

        """
        if values is None:
            values = {}
        for k, v in values.items():

            if type(v) is dict:
                self.writer.add_scalars(k, v, epoch)
            else:
                self.writer.add_scalar(k, v, epoch)
            if k is 'loss':
                print(f"epoch: {epoch}" + json.dumps(v), end='\n')

    def log_fps(self, epoch):
        """
        logs the current fps
        :param epoch: current epoch
        """
        new_time = datetime.datetime.now()
        self.writer.add_scalar("info/fps", self.steps_per_epoch * 1.0 / (new_time - self.t).total_seconds(), epoch)
        self.t = new_time

    def log_images(self, epoch, images, tag_name, columns):
        """
        logs images to the tensorboard
        :param epoch: current epoch
        :param images: list of images, scaled to 0-1
        :param tag_name: tag_name of images in tensorboard
        :param columns: number of columns to display the images
        """
        grid = vutils.make_grid(images, normalize=True, scale_each=False, nrow=columns)
        self.writer.add_image(tag_name, grid, epoch)

    def save_model(self, epoch):
        if epoch % self.save_model_every_nth == 0:  # and epoch > 0:
            self.model.save_model(self.loggin_path)
            self.model.save_model(self.shared_model_path)

    def log_config(self, config):
        base_tag = 'config'
        # log batch size and device count
        text = f"batch size: {config.batch_size}\n\nnum gpus: {torch.cuda.device_count()}"
        self.writer.add_text(base_tag + "/hyperparameters", text)
        # log model
        text = str(self.model)
        self.writer.add_text(base_tag + "/model", text)
        # log complete config
        text = inspect.getsource(config)
        self.writer.add_text(base_tag + '/config', '\t' + text.replace('\n', '\n\t'))

        # log rest of the config file for all params
        file = inspect.getfile(config)
        with open(file) as f:
            self.writer.add_text('rest', '\t' + f.read().replace('\n', '\n\t'))


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
