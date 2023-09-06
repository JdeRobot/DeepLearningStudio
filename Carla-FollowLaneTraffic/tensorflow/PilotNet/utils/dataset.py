import math

import numpy as np

from tensorflow.keras.utils import Sequence
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast, 
    HueSaturationValue, FancyPCA, RandomGamma, GaussNoise,
    GaussianBlur, ToFloat, Normalize, ColorJitter, ChannelShuffle, Equalize,
    RandomRain, RandomShadow, RandomSnow, RandomFog, RandomSunFlare, ReplayCompose
)
from albumentations.core.composition import OneOf


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
        
        new_img_batch = []  
        for x, img in enumerate(batch_x):
            aug = self.augment(image=img)["image"]
            velocity_dim = np.full((66, 200), batch_y[x][2])
            new_img_vel = np.dstack((aug, velocity_dim))
            new_img_vel = aug
            new_img_batch.append(new_img_vel)
            
        new_ann_batch = []
        for x, ann in enumerate(batch_y):
            new_ann_batch.append(np.array((ann[0], ann[1])))
        
        a, b = np.stack(new_img_batch, axis=0), np.array(new_ann_batch)
        return a, b
    
    
def get_augmentations(data_augs):
    if data_augs == 1:
        AUGMENTATIONS_TRAIN = ReplayCompose([
            RandomBrightnessContrast(),
            HueSaturationValue(),
            RandomGamma(),
            GaussianBlur(),
            Normalize()
        ])
    elif data_augs == 2:
        AUGMENTATIONS_TRAIN = ReplayCompose([
            RandomBrightnessContrast(),
            HueSaturationValue(),
            RandomGamma(),
            GaussianBlur(),
            OneOf([
                RandomRain(),
                RandomSnow(),
                RandomFog(),
                RandomSunFlare(),
                RandomShadow()
            ]),
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

