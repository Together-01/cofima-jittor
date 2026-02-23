import numpy as np
from jittor.dataset.cifar import CIFAR100
from jittor import transform


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR100_224(iData):
    use_path = False
    train_trsf = [
        transform.RandomResizedCrop(224, interpolation=3),
        transform.RandomHorizontalFlip(),
        transform.ColorJitter(brightness=63/255),
    ]
    test_trsf = [
        transform.Resize(256, mode=3),
        transform.CenterCrop(224),
    ]
    common_trsf = [
        transform.ToTensor(),
        transform.ImageNormalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ]
    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = CIFAR100('./data', train=True, download=True)
        test_dataset = CIFAR100('./data', train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)
