# -*- coding: UTF-8 -*-

import os
import random
from torch.utils.data import Dataset, ConcatDataset
import torchvision
import numpy as np
from PIL import Image


def channel_shuffle_fn(img):
    img = np.array(img, dtype=np.uint8)

    channel_idx = list(range(img.shape[-1]))
    random.shuffle(channel_idx)

    img = img[:, :, channel_idx]

    img = Image.fromarray(img, 'RGB')
    return img


class ClusterDataset(Dataset):
    def __init__(self, root, dataset_type, img_type, training=True):

        assert img_type in ['rgb', 'grayscale', 'sobel']

        self.training = training

        if dataset_type == 'MNIST':
            dataset_train = torchvision.datasets.MNIST(root, train=True)
            dataset_test = torchvision.datasets.MNIST(root, train=False)
            self.dataset = ConcatDataset([dataset_train, dataset_test])
        elif dataset_type == 'FashionMNIST':
            dataset_train = torchvision.datasets.FashionMNIST(root, train=True)
            dataset_test = torchvision.datasets.FashionMNIST(root, train=False)
            self.dataset = ConcatDataset([dataset_train, dataset_test])
        elif dataset_type == 'CIFAR10':
            dataset_train = torchvision.datasets.CIFAR10(root, train=True)
            dataset_test = torchvision.datasets.CIFAR10(root, train=False)
            self.dataset = ConcatDataset([dataset_train, dataset_test])
        elif dataset_type == 'STL10':
            dataset_train = torchvision.datasets.STL10(root, split='train')
            dataset_test = torchvision.datasets.STL10(root, split='test')
            self.dataset = ConcatDataset([dataset_train, dataset_test])
        elif dataset_type in ['ImageNet10']:
            # The directory is like:
            # root
            # |-- cls1
            # |-- |-- img1
            # |-- |-- img2
            # |-- |-- ...
            # |-- cls2
            # |-- ...
            classes = sorted(os.listdir(root))
            self.dataset = list()
            for idx, cls in enumerate(classes):
                for img_fp in sorted(os.listdir(os.path.join(root, cls))):
                    img = Image.open(os.path.join(root, cls, img_fp)).convert('RGB')
                    self.dataset.append((img, idx))
        else:
            raise NotImplementedError

        if dataset_type in ['MNIST', 'FashionMNIST']:
            if img_type == 'rgb':
                raise ValueError

            self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

            aug_list = list()
            aug_list.append(torchvision.transforms.RandomResizedCrop(28, scale=(0.4, 1.0), ratio=(3. / 4., 4. / 3.)))
            if dataset_type != 'MNIST':
                aug_list.append(torchvision.transforms.RandomHorizontalFlip(p=0.5))
            aug_list.append(torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.125))
            aug_list.append(torchvision.transforms.ToTensor())
            aug_list.append(torchvision.transforms.Normalize(mean=[0.5], std=[0.5]))
            self.transforms_aug = torchvision.transforms.Compose(aug_list)

        elif dataset_type in ['CIFAR10', 'STL10', 'ImageNet10']:
            if dataset_type == 'CIFAR10':
                img_size = 32
            else:
                img_size = 96

            if img_type == 'rgb':
                self.transforms = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
            else:
                self.transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Grayscale(num_output_channels=1),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
                ])

            aug_list = list()
            aug_list.append(
                torchvision.transforms.RandomResizedCrop(img_size, scale=(0.4, 1.0), ratio=(3. / 4., 4. / 3.)))
            aug_list.append(torchvision.transforms.RandomHorizontalFlip(p=0.5))
            aug_list.append(channel_shuffle_fn)
            if img_type != 'rgb':
                aug_list.append(torchvision.transforms.Grayscale(num_output_channels=1))
            aug_list.append(torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.125))
            aug_list.append(torchvision.transforms.ToTensor())
            if img_type == 'rgb':
                aug_list.append(torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
            else:
                aug_list.append(torchvision.transforms.Normalize(mean=[0.5], std=[0.5]))
            self.transforms_aug = torchvision.transforms.Compose(aug_list)
        else:
            raise NotImplementedError

    def __getitem__(self, item):
        img_raw, label = self.dataset[item]
        img = self.transforms(img_raw)
        if self.training:
            img_aug = self.transforms_aug(img_raw)
            return img, img_aug
        else:
            return img, label

    def __len__(self):
        return len(self.dataset)
