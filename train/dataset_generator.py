from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
import torch
import numpy as np

class CustomImageDataset(Dataset):
    """ Creates a custom Image Dataset which will be loaded at each iteration through the dataloader 
    
    Args:
        img_dir: pandas dataframe of image metadata
        root_dir: path to recode-perceptions repo 
        transform: string for pretrained model preprocessing
        target_transform: processing of output label
    """
    def __init__(self, img_dir, root_dir = '', transform='resnet', target_transform=None):
        self.img_dir = img_dir
        self.transform = preprocessing(transform)
        self.target_transform = target_transform
        self.root = root_dir

    def __len__(self):
        return self.img_dir.shape[0]

    def __getitem__(self, idx):
        img_path = self.img_dir.iloc[idx]['file']               # locates filename for next image
        image = read_image(self.root + img_path)                # loads image to memory
        label = self.img_dir.iloc[idx]['trueskill.score_norm']  # gets outcome label
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return (image, label)

def preprocessing(transform):
    if transform == 'resnet':
        preprocess = transforms.Compose([
            transforms.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float64)/255)),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess
    else:
        # create custom transform for model
        pass
