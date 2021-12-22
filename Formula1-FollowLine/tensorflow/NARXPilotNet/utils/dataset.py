import numpy as np
import math

from tensorflow.keras.utils import Sequence
from skimage.io import imread
from skimage.transform import resize
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast,
    HueSaturationValue, FancyPCA, RandomGamma, GaussNoise,
    GaussianBlur, ToFloat, Normalize, ColorJitter, ChannelShuffle, Equalize, ReplayCompose
)


class DatasetSequence(Sequence):
    def __init__(self, x_img_set, x_ann_set, y_set, batch_size, augmentations):
        self.x_img, self.x_ann = x_img_set, x_ann_set
        self.y = y_set
        self.batch_size = batch_size
        self.augment = augmentations

    def __len__(self):
        return math.ceil(len(self.x_img) / self.batch_size)

    def __getitem__(self, idx):
        batch_x_img = self.x_img[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x_img = np.stack([
              self.augment(image = x)["image"] for x in batch_x_img
        ], axis = 0)
        
        batch_x_ann = np.array(self.x_ann[idx * self.batch_size:(idx + 1) * self.batch_size])
        batch_x = [batch_x_img, batch_x_ann]
        
        batch_y = np.array(self.y[idx * self.batch_size:(idx + 1) * self.batch_size])

        return batch_x, batch_y


def get_augmentations(data_augs):
    if data_augs:
        AUGMENTATIONS_TRAIN = Compose([
            RandomBrightnessContrast(),
            HueSaturationValue(),
            FancyPCA(),
            RandomGamma(),
            GaussianBlur(),
            GaussNoise(),
            #
            #ColorJitter(),
            #Equalize(),
            #ChannelShuffle(),
            #
            Normalize()
        ])
    else:
        AUGMENTATIONS_TRAIN = Compose([
            Normalize()
        ])


    AUGMENTATIONS_TEST = Compose([
        Normalize()
    ])
    
    return AUGMENTATIONS_TRAIN, AUGMENTATIONS_TEST