import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class DataSplitter:
    def __init__(self, dataset, batch_size=64, num_workers=4, shuffle=True, validation_size=0.2):
        """
        Initialize train and validation data splitter
        :param dataset: Data set to use
        :param batch_size: Batch size
        :param num_workers: Number of workers to use
        :param shuffle: Shuffle data set before splitting in train and validation set
        :param validation_size: Size of the validation set in percent
        """
        # variables used to initialize the dataloaders
        self.batch_size = batch_size
        self.dataset = dataset
        self.num_workers = num_workers

        # setup sampler
        N = len(self.dataset)
        idx = list(range(N))
        if shuffle:
            np.random.shuffle(idx)
        split = int(np.floor(validation_size * len(self.dataset)))
        train_idx, validation_idx = idx[split:], idx[:split]
        self.training_sampler = SubsetRandomSampler(train_idx)
        self.validation_sampler = SubsetRandomSampler(validation_idx)

        self._init_data_loader()

    def get_train_data_loader(self):
        """
        :return: Dataloader object for training data
        """
        return self.train_data_loader

    def get_validation_data_loader(self):
        """
        :return: Dataloader object for validation data
        """
        return self.train_data_loader

    def _init_data_loader(self):
        """
        initialize data_loaders with current configuration
        """
        self.train_data_loader = DataLoader(self.dataset, sampler=self.training_sampler, batch_size=self.batch_size,
                                            num_workers=self.num_workers, pin_memory=True, drop_last=True)
        self.validation_data_loader = DataLoader(self.dataset, sampler=self.validation_sampler,
                                                 batch_size=self.batch_size,
                                                 num_workers=self.num_workers, pin_memory=True, drop_last=True)

    def adjusted_batch_size_and_increase_resolution(self, batch_size):
        """
        set the batch_size to a new value and also increase the resolution of the data_set
        :param batch_size: new batch_size
        """
        self.batch_size = batch_size

        # increase resolution by doubling it
        self.dataset.increase_resolution()

        # initialize data_loader again
        self._init_data_loader()
