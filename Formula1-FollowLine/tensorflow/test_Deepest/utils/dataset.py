from tensorflow.keras.utils import Sequence
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import math


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
            new_batch.append(self.augment.replay(saved_augmentations=aug['replay'], image=img)["image"])

        return np.stack(new_batch, axis=0), np.array(batch_y)



from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast,
    HueSaturationValue, FancyPCA, RandomGamma, GaussNoise,
    GaussianBlur, ToFloat, Normalize, ColorJitter, ChannelShuffle, Equalize, ReplayCompose
)

def get_augmentations():
    AUGMENTATIONS_TRAIN = ReplayCompose([
        RandomBrightnessContrast(),
        HueSaturationValue(),
        FancyPCA(),
        RandomGamma(),
        GaussianBlur(),
        # GaussNoise(),
        #
        # ColorJitter(),
        # Equalize(),
        # ChannelShuffle(),
        #
        Normalize()
    ])

    AUGMENTATIONS_TEST = ReplayCompose([
        Normalize()
    ])
    return AUGMENTATIONS_TRAIN, AUGMENTATIONS_TEST