import torch

from Configuration.config_general import MOST_RECENT_MODEL
from Utils.DataSplitter import DataSplitter
from Utils.Logging.LoggingUtils import Logger


class Trainer:
    def __init__(self, config):
        self.batch_size = config['batch_size']
        if torch.cuda.device_count() > 1:
            self.batch_size *= torch.cuda.device_count()
        print('BatchSize:', self.batch_size)
        self.dataset = config['dataset']()
        self.data_loader = DataSplitter(self.dataset, self.batch_size, validation_size=0.005)
        self.model = config['model'](**config['model_params'], dataset=self.dataset, initial_batch_size=self.batch_size,
                                     data_loader=self.data_loader)

        self.logger = Logger(len(self.dataset), self.model, save_model_every_nth=5,
                             shared_model_path=MOST_RECENT_MODEL)
        self.logger.log_config(config)

    def train(self):
        max_epochs = 101
        validate_index = 0
        validation_frequencies = [1, 1, 1]
        validation_periods = [0, 10, 20, max_epochs + 1]

        for current_epoch in range(max_epochs):
            self.model.set_train_mode(True)
            train_data_loader = self.data_loader.get_train_data_loader()
            info = self.model.train(train_data_loader, self.batch_size, current_epoch=current_epoch, validate=False, )

            # update frequency
            if current_epoch >= validation_periods[validate_index + 1]:
                validate_index += 1

            if current_epoch % validation_frequencies[validate_index] == 0:
                self.model.log(self.logger, current_epoch, *info, log_images=True)
                # do validation
                self.model.set_train_mode(False)
                val_data_loader = self.data_loader.get_validation_data_loader()
                info = self.model.train(val_data_loader, self.batch_size, current_epoch=current_epoch, validate=True)
                self.model.log_validation(self.logger, current_epoch, *info)
                self.model.set_train_mode(True)
            else:
                self.model.log(self.logger, current_epoch, *info)
