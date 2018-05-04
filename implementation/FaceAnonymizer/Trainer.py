import torch

from torch.utils.data import DataLoader

from Logging.LoggingUtils import Logger
from config import deep_fake_config


class Trainer:
    def __init__(self, dataset, config=deep_fake_config):
        self.batch_size = config['batch_size']
        self.epochs = config['num_epoch']

        if torch.cuda.device_count() > 1:
            self.batch_size *= torch.cuda.device_count()
            dataset.size_multiplicator *= torch.cuda.device_count()

        self.data_loader = DataLoader(dataset, self.batch_size, shuffle=True, num_workers=12, pin_memory=True,
                                      drop_last=True)

        self.model = config['model'](self.data_loader, **config['model_arguments'])
        self.logger = Logger(len(dataset) // dataset.size_multiplicator, self.model, save_model_every_nth=100)
        self.logger.log_config(config)

    def train(self):
        for i in range(self.epochs):
            info = self.model.train(i)
            self.logger.log(i, *info)
