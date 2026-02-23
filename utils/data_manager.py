import logging

import numpy as np
from PIL import Image
from jittor.dataset import Dataset
from jittor import transform

from utils.data import iCIFAR100_224


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        '''Get the number of tasks.'''
        return len(self._increments)

    def get_task_size(self, task):
        '''Get the number of classes in the given task.'''
        return self._increments[task]

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False):
        '''Get the dataset with class index in indices from the given source and mode.'''
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transform.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transform.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx + 1)
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf)
        return DummyDataset(data, targets, trsf)

    def _setup_data(self, dataset_name, shuffle, seed):
        '''Setup the data according to the dataset name.'''
        idata = _get_idata(dataset_name)
        idata.download_data()

        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets

        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        self._train_targets = _remap_class_index(self._train_targets, self._class_order)
        self._test_targets = _remap_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        '''Select the data with class index in [low_range, high_range).'''
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf):
        super().__init__()
        
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.images[idx].shape[0] == 3:
            image = self.trsf(Image.fromarray(self.images[idx].swapaxes(0, 1).swapaxes(1, 2), "RGB"))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]
        return idx, image, label


def _remap_class_index(y, order):
    '''Map the original class index to the new class index according to the given order.'''
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    '''Get the iData instance according to the dataset name.'''
    name = dataset_name.lower()
    if name == "cifar100_224":
        return iCIFAR100_224()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))
