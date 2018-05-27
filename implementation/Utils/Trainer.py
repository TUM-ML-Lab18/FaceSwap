import torch

from Utils.Logging.LoggingUtils import Logger
from Configuration.config_general import MOST_RECENT_MODEL
from Utils.DataSplitter import DataSplitter


class Trainer:
    def __init__(self, config):
        self.batch_size = config['batch_size']
        if torch.cuda.device_count() > 1:
            self.batch_size *= torch.cuda.device_count()
        self.dataset = config['dataset']()
        self.model = config['model'](**config['model_params'])
        self.data_loader = DataSplitter(self.dataset, self.batch_size)

        self.logger = Logger(len(self.dataset), self.model, save_model_every_nth=100,
                             shared_model_path=MOST_RECENT_MODEL)
        self.logger.log_config(config)

    def train(self):
        max_epochs = 5000
        validate_index = 0
        validation_frequencies = [2, 20]
        validation_periods = [0, 20, max_epochs + 1]

        for current_epoch in range(max_epochs):
            self.model.set_train_mode(True)
            train_data_loader = self.data_loader.get_train_data_loader()
            info = self.model.train(train_data_loader, self.batch_size)

            # update frequency
            if current_epoch >= validation_periods[validate_index + 1] :
                validate_index += 1

            if current_epoch % validation_frequencies[validate_index] == 0:
                self.model.log(self.logger, current_epoch, *info, log_images=True)
                # do validation
                self.model.set_train_mode(False)
                val_data_loader = self.data_loader.get_validation_data_loader()
                info = self.model.validate(val_data_loader, self.batch_size)
                self.model.log_validation(self.logger, current_epoch, *info)
                self.model.set_train_mode(True)
            else:
                self.model.log(self.logger, current_epoch, *info)
