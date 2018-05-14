import torch

from Logging.LoggingUtils import Logger
from configuration.general_config import MOST_RECENT_MODEL


class Trainer:
    def __init__(self, root_folder, config):
        self.batch_size = config['batch_size']
        self.epochs = config['num_epoch']
        self.validation_freq = config['validation_freq']

        if torch.cuda.device_count() > 1:
            self.batch_size *= torch.cuda.device_count()
            # dataset.size_multiplicator *= torch.cuda.device_count()

        dataset = config['dataset'](root_folder, config['img_size'])
        self.data_loader = config['data_loader'](dataset, batch_size=self.batch_size)
        self.model = config['model'](config['img_size'])

        self.logger = Logger(len(dataset) // dataset.size_multiplicator, self.model, save_model_every_nth=100,
                             shared_model_path=MOST_RECENT_MODEL)
        self.logger.log_config(config)

    def train(self):
        for i in range(self.epochs):
            batches = self.data_loader.train()
            info = self.model.train(i, batches)
            self.model.log(self.logger, i, *info)

            if i % self.validation_freq == 0:
                # do validation
                self.model.set_train_mode(False)
                batches = self.data_loader.validation()
                loss1, loss2 = self.model.validate(batches)
                self.model.log_validate(self.logger, i, loss1, loss2)
                self.model.set_train_mode(True)
