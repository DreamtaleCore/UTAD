import os
import cv2
import sys
from PIL import ImageFilter, Image
import numpy as np
from torchvision.transforms import transforms
import torchvision.datasets as datasets
from torchvision.datasets.folder import is_image_file, default_loader, IMG_EXTENSIONS, make_dataset
import random
# from RandAugment import RandAugment


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def train_augmentation(input_size=256):
    return [
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            # ], p=0.8),
            # RandAugment(2, 9),                              # https://github.com/ildoonet/pytorch-randaugment
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.Resize((input_size, input_size)),
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.)),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
        ]

def test_augmentation(input_size=256):
    return [
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            # ], p=0.8),
            # RandAugment(2, 9),                              # https://github.com/ildoonet/pytorch-randaugment
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(45),
            transforms.Resize((input_size, input_size)),
            # transforms.RandomResizedCrop(256, scale=(0.8, 1.)),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
        ]

small_aug = transforms.Compose(([transforms.Resize((64, 64)), transforms.ToTensor()]))

def get_trainset(s_dir, small_dataset_name=None, class_id=0, input_size=256, is_boulon=False):
    if small_dataset_name is None:
        return datasets.ImageFolder(os.path.join(s_dir, 'train'), transform=transforms.Compose(train_augmentation(input_size)))
    else:
        return SmallCVDataset(s_dir, is_train=True, download=True, name=small_dataset_name, transform=small_aug)


def get_testset(s_dir, sub_name=None, small_dataset_name=None, class_id=0, input_size=256, is_boulon=False):
    if small_dataset_name is None:
        sub_name = 'test' if sub_name is None else sub_name
        return DatasetFolder(os.path.join(s_dir, sub_name), transform=transforms.Compose(test_augmentation(input_size)), with_pwd=True)
    else:
        is_train = True if sub_name == 'train' else False
        return SmallCVDataset(s_dir, is_train=is_train, download=True, name=small_dataset_name, transform=small_aug)


def get_valset(s_dir, small_dataset_name=None, class_id=0, input_size=256):
    if small_dataset_name is None:
        return DatasetFolder(os.path.join(s_dir, 'test'), transform=transforms.Compose(test_augmentation(input_size)))
    else:
        return SmallCVDataset(s_dir, is_train=False, download=True, name=small_dataset_name, transform=small_aug)


class DatasetFolder(datasets.VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader=default_loader, extensions=None, 
                 transform=None, target_transform=None, is_valid_file=None, with_pwd=False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        extensions = IMG_EXTENSIONS if extensions is None else extensions
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions
        self.with_pwd = with_pwd

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
            Finds the class folders in a dataset.

            Args:
                dir (string): Root directory path.

            Returns:
                tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

            Ensures:
                No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.with_pwd:
            return sample, target, path
        return sample, target

    def __len__(self):
        return len(self.samples)


class SmallCVDataset(datasets.VisionDataset):
    def __init__(self, root, is_train=True, download=False, name='mnist', 
                    transform=None, target_transform=None, class_id=0):
        super().__init__(root, transform=transform, target_transform=target_transform)
        if name == 'mnist':
            self.dataset =  datasets.MNIST(root, is_train, transform, target_transform, download)
        elif name == 'fashion_mnist':
            self.dataset =  datasets.FashionMNIST(root, is_train, transform, target_transform, download)
        elif name == 'cifar10':
            self.dataset =  datasets.CIFAR10(root, is_train, transform, target_transform, download)
        else:
            raise ValueError('name must in [mnist, fashion_mnist, cifar10]')
        ori_len = len(self.dataset.data)
        self.dataset.data = [self.dataset.data[index, :] for index in range(ori_len) if self.dataset.targets[index] == class_id]
        # import pdb; pdb.set_trace()
        self.dataset.targets = [self.dataset.targets[index] for index in range(ori_len) if self.dataset.targets[index] == class_id]
    
    def __len__(self):
        return len(self.dataset.data)

    def __getitem__(self, index):
        index = index % self.__len__()
        return self.dataset[index]
