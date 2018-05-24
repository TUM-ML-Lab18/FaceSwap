import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class TrainValidationLoader:
    def __init__(self, dataset, batch_size=64, num_workers=12, shuffle=True, validation_size=0.2):
        if dataset is None:
            return

        idxs = list(range(len(dataset)))

        if shuffle:
            np.random.shuffle(idxs)

        split = int(np.floor(validation_size * len(dataset)))
        train_idxs, validation_idxs = idxs[split:], idxs[:split]

        train_sampler = SubsetRandomSampler(train_idxs)
        validation_sampler = SubsetRandomSampler(validation_idxs)

        self.train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size,
                                       num_workers=num_workers, pin_memory=True, drop_last=True)
        self.validation_loader = DataLoader(dataset, sampler=validation_sampler, batch_size=batch_size,
                                            num_workers=num_workers, pin_memory=True, drop_last=True)

    def train(self):
        return self.train_loader

    def validation(self):
        return self.validation_loader
