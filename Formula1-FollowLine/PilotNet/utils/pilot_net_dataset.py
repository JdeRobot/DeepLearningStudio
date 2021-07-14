from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from utils.processing import *

class PilotNetDataset(Dataset):
    def __init__(self, path_to_data, transforms=None):

        self.data_path = path_to_data

        self.images = []
        self.labels = []
        type_image = 'cropped'

        for path in path_to_data:
            all_images, all_data = load_data(path)
            self.images = get_images(all_images, type_image, self.images)        
            self.labels = parse_json(all_data, self.labels)

        self.labels, self.images = preprocess_data(self.labels, self.images)

        self.transforms = transforms

        self.image_shape = self.images[0].shape
        self.num_labels = np.array(self.labels[0]).shape[0]

        self.count = len(self.images)
        self.max_V = 13
        self.max_W = 3
        
    def __getitem__(self, index):

        img = self.images[index]
        label = np.array(self.labels[index])
        #label[0] = label[0]/self.max_V
        #label[1] = label[1]/self.max_W
        data = Image.fromarray(img)

        if self.transforms is not None:
            data = self.transforms(data)

        return (data, label)

    def __len__(self):
        return self.count
