import math

import numpy as np

from tensorflow.keras.utils import Sequence
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast,
    HueSaturationValue, FancyPCA, RandomGamma, GaussNoise,
    GaussianBlur, ToFloat, Normalize, ColorJitter, ChannelShuffle, Equalize, ReplayCompose,
    RandomRain, RandomShadow, RandomSnow, RandomFog, RandomSunFlare,
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

        new_batch = []
        for x, img in enumerate(batch_x):
            aug = self.augment(image=img[0])
            augmented_0 = self.augment.replay(saved_augmentations=aug['replay'], image=img[0])["image"]
            augmented_1 = self.augment.replay(saved_augmentations=aug['replay'], image=img[1])["image"]
            augmented_2 = self.augment.replay(saved_augmentations=aug['replay'], image=img[2])["image"]
            new_image = [augmented_0, augmented_1, augmented_2]
            new_batch.append(np.array(new_image))

        new_batch = np.array(new_batch)

        return np.stack(new_batch, axis=0), np.array(batch_y)


def get_augmentations(data_augs):
    if data_augs == 1:
        AUGMENTATIONS_TRAIN = ReplayCompose([
            RandomBrightnessContrast(),
            HueSaturationValue(),
            FancyPCA(),
            RandomGamma(),
            GaussianBlur(),
            Normalize()
        ])
    elif data_augs == 2:
        AUGMENTATIONS_TRAIN = ReplayCompose([
            RandomBrightnessContrast(),
            HueSaturationValue(),
            FancyPCA(),
            RandomGamma(),
            GaussianBlur(),
            OneOf([
                RandomRain(),
                RandomSnow(),
                RandomFog(),
                RandomSunFlare()
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
