import numpy as np
import math

from tensorflow.keras.utils import Sequence
from skimage.io import imread
from skimage.transform import resize
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
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        sample_weights = []
        for ann in batch_y:
            if ann[1] >= 0.7 or ann[1] <= 0.3:
                if ann[1] >= 0.9 or ann[1] <= 0.1:
                    sample_weights.append(3)
                elif ann[1] >= 0.8 or ann[1] <= 0.2:
                    sample_weights.append(2)
                else:
                    sample_weights.append(1.5)
            elif ann[0] <= 0.2:
                sample_weights.append(2)
            else:
                sample_weights.append(1)
        aug = self.augment(image=batch_x[0])
        new_batch = []

        for x, img in enumerate(batch_x):
            new_batch.append(self.augment.replay(saved_augmentations=aug['replay'], image=img)["image"])

        return np.stack(new_batch, axis=0), np.array(batch_y), np.array(sample_weights)


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