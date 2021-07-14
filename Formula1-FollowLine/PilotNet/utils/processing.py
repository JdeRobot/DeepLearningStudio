import os
import glob
import numpy as np
import cv2
from tqdm import tqdm


def load_data(folder):
    name_folder = folder + '/Images/'
    list_images = glob.glob(name_folder + '*')
    images = sorted(list_images, key=lambda x: int(x.split('/')[-1].split('.png')[0]))
    name_file = folder + '/data.json'
    file = open(name_file, 'r')
    data = file.read()
    file.close()
    return images, data

def get_images(list_images, type_image, array_imgs):
    # Read the images
    for name in tqdm(list_images):
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if type_image == 'cropped':
            img = img[240:480, 0:640]
        # img = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)))
        img = cv2.resize(img, (int(200), int(66)))
        array_imgs.append(img)

    return array_imgs

def parse_json(data, array):
    # Process json
    data_parse = data.split('}')[:-1]
    for d in data_parse:
        v = d.split('"v": ')[1]
        d_parse = d.split(', "v":')[0]
        w = d_parse.split(('"w": '))[1]
        array.append((float(v), float(w)))

    return array

def preprocess_data(array, imgs):
    # Data augmentation
    # Take the image and just flip it and negate the measurement
    flip_imgs = []
    array_flip = []
    for i in tqdm(range(len(imgs))):
        flip_imgs.append(cv2.flip(imgs[i], 1))
        array_flip.append((array[i][0], -array[i][1]))
    new_array = array + array_flip
    new_array_imgs = imgs + flip_imgs

    extreme_case_1_img = []
    extreme_case_2_img = []
    extreme_case_1_array = []
    extreme_case_2_array = []

    for i in tqdm(range(len(new_array_imgs))):
        if abs(new_array[i][1]) > 2:
            extreme_case_2_img.append(new_array_imgs[i])
            extreme_case_2_array.append(new_array[i])
        elif abs(new_array[i][1]) > 1:
            extreme_case_1_img.append(new_array_imgs[i])
            extreme_case_1_array.append(new_array[i])

    new_array += extreme_case_1_array*5 + extreme_case_2_array*10
    new_array_imgs += extreme_case_1_img*5 + extreme_case_2_img*10

    return new_array, new_array_imgs

def normalize(x):
    x = np.asarray(x)
    return (x - x.min()) / (np.ptp(x))

def check_path(path):
    if not os.path.exists(path):
        print(f"{path} not exist")
        os.makedirs(path)
        print(f"Create {path} success")