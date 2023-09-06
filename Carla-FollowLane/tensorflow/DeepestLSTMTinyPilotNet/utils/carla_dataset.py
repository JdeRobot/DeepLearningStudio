import math

import numpy as np

from tensorflow.keras.utils import Sequence
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast, Affine,
    HueSaturationValue, FancyPCA, RandomGamma, GaussNoise,
    GaussianBlur, ToFloat, Normalize, ColorJitter, ChannelShuffle, Equalize,
    RandomRain, RandomShadow, RandomSnow, RandomFog, RandomSunFlare, ReplayCompose,
)
from albumentations.core.composition import OneOf


class DatasetSequenceAffine(Sequence):
    def __init__(self, x_set, y_set, batch_size, augmentations):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = augmentations

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]

        new_batch = []
        new_batch_y = np.array(batch_y, copy=True)
        for x, img in enumerate(batch_x):
            aug = self.augment(image=img)
            new_batch.append(aug["image"])
            if aug["replay"]["transforms"][0]["applied"] == True:
                x_transformation_value = aug["replay"]["transforms"][0]["translate_percent"]["x"][1]
                value = aug["replay"]["transforms"][0]["params"]["matrix"].params[0][2]
                new_value = value / 10 * x_transformation_value
                new_batch_y[x][1] = new_batch_y[x][1] + new_value

        return np.stack(new_batch, axis=0), np.array(new_batch_y)

class DatasetSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size, augmentations):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = augmentations

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        
        aug = self.augment(image=batch_x[0])
        new_batch = []  
        
        for x, img in enumerate(batch_x):
            aug = self.augment(image=img)["image"]
            new_batch.append(aug)
            
        return np.stack(new_batch, axis=0), np.array(batch_y)
    
    
def get_augmentations(data_augs):
    if data_augs == 1:
        AUGMENTATIONS_TRAIN = ReplayCompose([
            Affine(p=0.5, rotate=0, translate_percent={'x':(-0.2, 0.2)}),
            RandomBrightnessContrast(),
            HueSaturationValue(),
            FancyPCA(),
            RandomGamma(),
            GaussianBlur(),
            Normalize()
        ])
    else:
        AUGMENTATIONS_TRAIN = Compose([
            Normalize()
        ])

    AUGMENTATIONS_TEST = ReplayCompose([
        Normalize()
    ])
    
    return AUGMENTATIONS_TRAIN, AUGMENTATIONS_TEST

