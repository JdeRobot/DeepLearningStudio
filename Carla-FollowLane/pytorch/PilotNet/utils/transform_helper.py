from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast, 
    HueSaturationValue, FancyPCA, RandomGamma, GaussNoise,
    GaussianBlur, ToFloat, Normalize, ColorJitter, ChannelShuffle, Equalize, ReplayCompose, CoarseDropout,
    Affine
)

from albumentations.pytorch.transforms  import ToTensorV2

all_augs_dict = ReplayCompose([
    Affine(p=0.5, rotate=0, translate_percent={'x':(-0.2, 0.2)}),
    RandomBrightnessContrast(),
    HueSaturationValue(),
    FancyPCA(),
    RandomGamma(),
    GaussianBlur(),
    Normalize(),
    ToTensorV2()
])

all_augs_dict_test = ReplayCompose([
    Normalize(),
    ToTensorV2()
])


def createTransform(augmentations):
    if augmentations == ['all']:
        augs_to_compose = []
        augs_to_compose = all_augs_dict
        createdTransform = augs_to_compose
    else:
        augs_to_compose = []
        augs_to_compose = all_augs_dict_test
        createdTransform = augs_to_compose

    return createdTransform
