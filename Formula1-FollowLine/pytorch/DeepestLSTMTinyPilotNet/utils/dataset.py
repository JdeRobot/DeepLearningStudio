from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from utils.processing import *

class DLTNetDataset(Dataset):
    def __init__(self, path_to_data, img_shape, transforms=None, preprocessing=None, trainset=True):

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

        if trainset:
            self.images, self.labels, _, _ = process_dataset(path_to_data, type_image, data_type, img_shape)
        else:
            _, _, self.images, self.labels = process_dataset(path_to_data, type_image, data_type, img_shape)
        

        # for path in path_to_data: # only single dataset
            # all_images, all_data = load_data(path)
            # self.images = get_images(all_images, type_image, self.images)        
            # self.labels = parse_json(all_data, self.labels)

        # self.labels, self.images = preprocess_data(self.labels, self.images, data_type)

        self.transforms = transforms

        # self.image_shape = self.images[0].shape
        self.image_shape = img_shape
        self.num_labels = np.array(self.labels[0]).shape[0]

        self.count = len(self.images)
        
    def __getitem__(self, index):

        img = self.images[index]
        label = np.array(self.labels[index])
        data = Image.fromarray(img)

        if self.transforms is not None:
            data = self.transforms(data)

        return (data, label)

    def __len__(self):
        return self.count
