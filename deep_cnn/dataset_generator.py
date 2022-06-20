import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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
        data_iterator = datasets.ImageFolder(dir, preprocess)
        loader = DataLoader(data_iterator, **params)
        logger.info(
            "There are %s images in the %s DataLoader"
            % (str(loader.__len__() * params["batch_size"]), split)
        )
        classes = len(os.listdir(dir))
    else:
        loader = None
        classes = None
    return loader, classes
