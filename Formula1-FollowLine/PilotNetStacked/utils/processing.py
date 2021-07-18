import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
from collections import deque

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

def preprocess_data(array, imgs, horizon):
    # Data augmentation
    # Take the image and just flip it and negate the measurement

    image_trace = deque([], maxlen=horizon)
    array_trace = deque([], maxlen=horizon)

    flip_image_trace = deque([], maxlen=horizon)
    flip_array_trace = deque([], maxlen=horizon)
    
    for _ in range(horizon):
        image_trace.append(imgs[0])
        array_trace.append(array[0])
        flip_image_trace.append(cv2.flip(imgs[0], 1))
        flip_array_trace.append((array[0][0], -array[0][1]))

    extreme_case_0_img = []
    extreme_case_1_img = []
    extreme_case_2_img = []
    extreme_case_0_array = []
    extreme_case_1_array = []
    extreme_case_2_array = []

    for i in tqdm(range(len(imgs))):
        image_trace.append(imgs[i])
        array_trace.append(array[i])
        flip_image_trace.append(cv2.flip(imgs[i], 1))
        flip_array_trace.append((array[i][0], -array[i][1]))
        if abs(array[i][1]) > 2:
            extreme_case_2_img.append(list(image_trace.copy()))
            extreme_case_2_array.append(list(array_trace.copy()))
            extreme_case_2_img.append(list(flip_image_trace.copy()))
            extreme_case_2_array.append(list(flip_array_trace.copy()))
        elif abs(array[i][1]) > 1:
            extreme_case_1_img.append(list(image_trace.copy()))
            extreme_case_1_array.append(list(array_trace.copy()))
            extreme_case_1_img.append(list(flip_image_trace.copy()))
            extreme_case_1_array.append(list(flip_array_trace.copy()))
        else:
            extreme_case_0_img.append(list(image_trace.copy()))
            extreme_case_0_array.append(list(array_trace.copy()))
            extreme_case_0_img.append(list(flip_image_trace.copy()))
            extreme_case_0_array.append(list(flip_array_trace.copy()))

    new_array = extreme_case_0_array + extreme_case_1_array*5 + extreme_case_2_array*10
    new_array_imgs = extreme_case_0_img + extreme_case_1_img*5 + extreme_case_2_img*10

    return np.array(new_array), np.array(new_array_imgs)

def normalize(x):
    x = np.asarray(x)
    return (x - x.min()) / (np.ptp(x))

def check_path(path):
    if not os.path.exists(path):
        print(f"{path} not exist")
        os.makedirs(path)
        print(f"Create {path} success")
