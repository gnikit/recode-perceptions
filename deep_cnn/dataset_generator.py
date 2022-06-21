import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from .logger import logger


def preprocessing(transform):
    if transform == "resnet":
        preprocess = transforms.Compose(
            [
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return preprocess
    else:
        # create custom transform for model
        pass


def dataloader(data_dir, root_dir, transform, split, params):
    """Creates dataloader from
    train, val data folders"""

    dir = os.path.join(root_dir, data_dir, split)

    # get normalisation
    preprocess = preprocessing(transform)

    if os.path.isdir(dir):
        # Data loading
        if split == "train":
            data_iterator = datasets.ImageFolder(dir, preprocess)
            L = len(data_iterator)
            train_it, val_it = random_split(
                data_iterator,
                [int(0.95 * L), int(0.05 * L)],
                generator=torch.Generator().manual_seed(42),
            )
            train_loader = DataLoader(train_it, **params)
            val_loader = DataLoader(val_it, **params)
            logger.info(
                "There are %s images in the %s DataLoader"
                % (str(train_loader.__len__() * params["batch_size"]), split)
            )
            logger.info(
                "There are %s images in the %s DataLoader"
                % (str(val_loader.__len__() * params["batch_size"]), "val")
            )
            classes = len(os.listdir(dir))
            test_loader = None
        else:
            data_iterator = datasets.ImageFolder(dir, preprocess)
            test_loader = DataLoader(data_iterator, **params)
            logger.info(
                "There are %s images in the %s DataLoader"
                % (str(test_loader.__len__() * params["batch_size"]), split)
            )
            classes = len(os.listdir(dir))
            train_loader = None
            val_loader = None
    else:
        train_loader = None
        val_loader = None
        classes = None
    return train_loader, val_loader, test_loader, classes
