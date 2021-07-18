import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from utils.processing import *
import numpy as np

class PilotNetDataset(Dataset):
    def __init__(self, path_to_data, horizon, transforms=None, preprocessing=None):

        self.data_path = path_to_data
        self.horizon = horizon

        self.images = []
        self.labels = []

        if preprocessing is not None:
            if 'nocrop' in preprocessing:
                type_image = None
            else:
                type_image = 'cropped'
            
            if 'extreme' in preprocessing:
                data_type = 'extreme'
            else:
                data_type = None
        else:
            type_image = 'cropped'
            data_type = None

        for path in path_to_data:
            all_images, all_data = load_data(path)
            self.images = get_images(all_images, type_image, self.images)        
            self.labels = parse_json(all_data, self.labels)

        self.labels, self.images = preprocess_data(self.labels, self.images, horizon, data_type)

        self.transforms = transforms

        self.image_shape = self.images[0].shape
        self.num_labels = np.array(self.labels[0]).shape[0]

        self.count = len(self.images)
        
    def __getitem__(self, index):

        all_imgs = self.images[index]
        all_labels = self.labels[index]

        new_set_imgs = []
        label = all_labels[-1:]

        for iter in range(self.horizon):
            img = all_imgs[iter]
            label = np.array(all_labels[iter])
            data = Image.fromarray(img)

            if self.transforms is not None:
                data = self.transforms(data)

            new_set_imgs.append(data)
            
        set_data = torch.vstack(new_set_imgs)

        return (set_data, label)

    def __len__(self):
        return self.count
