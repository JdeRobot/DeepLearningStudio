import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import numpy as np

all_augs_dict = {
    'gaussian': transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
    'jitter': transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    'perspective':transforms.RandomPerspective(distortion_scale=0.3, p=1.0),
    'affine':transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.2), scale=(0.9, 1)),
    'posterize':transforms.RandomPosterize(bits=2)
}


def createTransform(augmentations):

    augs_to_compose = []

    if 'auto' in augmentations:
        for data_aug in all_augs_dict.keys():
            if np.random.rand() > 0.5:
                action = all_augs_dict[data_aug]
                augs_to_compose.append(action)
    elif 'all' in augmentations:
        for data_aug in all_augs_dict.keys():
            action = all_augs_dict[data_aug]
            augs_to_compose.append(action)
    else:
        for data_aug in augmentations:
            action = all_augs_dict[data_aug]
            augs_to_compose.append(action)

    augs_to_compose.append(transforms.ToTensor())
    createdTransform = transforms.Compose(augs_to_compose)

    return createdTransform
