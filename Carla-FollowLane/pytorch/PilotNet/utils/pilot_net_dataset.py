from torch.utils.data.dataset import Dataset
from PIL import Image
from utils.processing_carla import *
from pathlib import Path

class PilotNetDataset(Dataset):
    def __init__(self, path_to_data, transforms=None, preprocessing=None):

        self.data_path = path_to_data

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
        
        print('*'*8, "Loading Datasets", '*'*8)    
        for path in path_to_data: 
            all_images, all_data = load_data(path)
            self.images = get_images(all_images, type_image, self.images)
            self.labels += all_data

        self.labels, self.images = preprocess_data(self.labels, self.images, data_type)

        self.transforms = transforms

        self.image_shape = self.images[0].shape
        self.num_labels = np.array(self.labels[0]).shape[0]

        self.count = len(self.images)
        
    def __getitem__(self, index):

        img = self.images[index]
        label = np.array(self.labels[index])
        data = Image.fromarray(img)
        if self.transforms is not None:
            aug = self.transforms(image=img)
            
            if aug["replay"]["transforms"][0]["applied"] == True:
                x_transformation_value = aug["replay"]["transforms"][0]["translate_percent"]["x"][1]
                value = aug["replay"]["transforms"][0]["params"]["matrix"].params[0][2]
                new_value = value/10*x_transformation_value
                label[1] = label[1]+new_value
            
            data = aug['image']
        
        return (data, label)

    def __len__(self):
        return self.count

class PilotNetDatasetTest(Dataset):
    def __init__(self, path_to_data, transforms=None, preprocessing=None):

        self.data_path = path_to_data

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
        
        print('*'*8, "Loading Datasets", '*'*8)    
        for path in path_to_data: 
            all_images, all_data = load_data(path)
            self.images = get_images(all_images, type_image, self.images)
            self.labels += all_data

        self.labels, self.images = preprocess_data(self.labels, self.images, data_type)

        self.transforms = transforms

        self.image_shape = self.images[0].shape
        self.num_labels = np.array(self.labels[0]).shape[0]

        self.count = len(self.images)
        
    def __getitem__(self, index):

        img = self.images[index]
        label = np.array(self.labels[index])
        data = Image.fromarray(img)

        if self.transforms is not None:
            aug = self.transforms(image=img)
            data = aug['image']
        
        return (data, label)

    def __len__(self):
        return self.count