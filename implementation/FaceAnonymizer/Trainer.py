import torch

from Logging.LoggingUtils import Logger
from configuration.gerneral_config import MOST_RECENT_MODEL


class Trainer:
    def __init__(self, dataset, config):
        self.batch_size = config['batch_size']
        self.epochs = config['num_epoch']

        if torch.cuda.device_count() > 1:
            self.batch_size *= torch.cuda.device_count()
            dataset.size_multiplicator *= torch.cuda.device_count()

        self.model = config['model2'](dataset)
        self.logger = Logger(len(dataset) // dataset.size_multiplicator, self.model, save_model_every_nth=100,
                             shared_model_path=MOST_RECENT_MODEL)
        self.logger.log_config(config)

    def train(self):
        for i in range(self.epochs):
            info = self.model.train(i)
            self.logger.log(i, *info)
