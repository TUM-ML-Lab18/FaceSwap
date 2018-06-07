import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class DataSplitter:
    def __init__(self, dataset, batch_size=64, num_workers=12, shuffle=True, validation_size=0.2):
        """
        Initialize train and validation data splitter
        :param dataset: Data set to use
        :param batch_size: Batch size
        :param num_workers: Number of workers to use
        :param shuffle: Shuffle data set before splitting in train and validation set
        :param validation_size: Size of the validation set in percent
        """
        N = len(dataset)
        idx = list(range(N))

        if shuffle:
            np.random.shuffle(idx)

        split = int(np.floor(validation_size * len(dataset)))
        train_idx, validation_idx = idx[split:], idx[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        validation_sampler = SubsetRandomSampler(validation_idx)

        self.train_data_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size,
                                            num_workers=num_workers, pin_memory=True, drop_last=True)
        self.validation_data_loader = DataLoader(dataset, sampler=validation_sampler, batch_size=batch_size,
                                                 num_workers=num_workers, pin_memory=True, drop_last=True)

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
