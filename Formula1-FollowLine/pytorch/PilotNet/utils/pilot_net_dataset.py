from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from utils.processing import *
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
        
        # collect all directories with image and data
        path_to_data = glob.glob(path_to_data[0] + '**/data.csv', recursive = True)
        # path_to_data = list(Path(path_to_data[0]).rglob('*/data.csv'))
        path_to_data = list(map(lambda path: "/".join(path.split("/")[:-1]), path_to_data)) # crop to only folder name

        for path in path_to_data[3:6]: ### TODO: remove this
            all_images, all_data = load_data(path)
            self.images = get_images(all_images, type_image, self.images)        
            # self.labels = parse_json(all_data, self.labels)
            self.labels = parse_csv(all_data, self.labels)

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
            data = self.transforms(data)

        return (data, label)

    def __len__(self):
        return self.count
