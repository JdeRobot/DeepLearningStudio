import os
import glob
import numpy as np
from numpy.core.numeric import asarray
import cv2
from tqdm import tqdm
from utils.bm_utils import calculate_deviation
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
    deviations = [0]
    for name in tqdm(list_images):
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if type_image == 'cropped':
            img = img[240:480, 0:640]

        # cv2.imshow('frame_0', img)
        # cv2.waitKey(30)
        deviations.append(calculate_deviation(img) - deviations[-1])
        # img = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)))
        img = cv2.resize(img, (int(200), int(66)))
        array_imgs.append(img)

    # import matplotlib.pyplot as plt
    # plt.plot(deviations)
    # plt.show()

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

def preprocess_data(array, imgs, horizon, data_type):
    # Data augmentation
    # Take the image and just flip it and negate the measurement

    image_trace = deque([], maxlen=horizon*5)
    array_trace = deque([], maxlen=horizon*5)

    flip_image_trace = deque([], maxlen=horizon*5)
    flip_array_trace = deque([], maxlen=horizon*5)
    
    for _ in range(horizon*5):
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

        selected_image_trace = [image_trace[0], image_trace[4], image_trace[9]]
        selected_array_trace = [array_trace[0], array_trace[4], array_trace[9]]
        selected_flip_image_trace = [flip_image_trace[0], flip_image_trace[4], flip_image_trace[9]]
        selected_flip_array_trace = [flip_array_trace[0], flip_array_trace[4], flip_array_trace[9]]

        if abs(array[i][1]) > 2:
            extreme_case_2_img.append(list(selected_image_trace.copy()))
            extreme_case_2_array.append(list(selected_array_trace.copy()))
            extreme_case_2_img.append(list(selected_flip_image_trace.copy()))
            extreme_case_2_array.append(list(selected_flip_array_trace.copy()))
        elif abs(array[i][1]) > 1:
            extreme_case_1_img.append(list(selected_image_trace.copy()))
            extreme_case_1_array.append(list(selected_array_trace.copy()))
            extreme_case_1_img.append(list(selected_flip_image_trace.copy()))
            extreme_case_1_array.append(list(selected_flip_array_trace.copy()))
        else:
            extreme_case_0_img.append(list(selected_image_trace.copy()))
            extreme_case_0_array.append(list(selected_array_trace.copy()))
            extreme_case_0_img.append(list(selected_flip_image_trace.copy()))
            extreme_case_0_array.append(list(selected_flip_array_trace.copy()))

    if data_type == 'extreme':
        new_array = extreme_case_0_array + extreme_case_1_array*5 + extreme_case_2_array*10
        new_array_imgs = extreme_case_0_img + extreme_case_1_img*5 + extreme_case_2_img*10
    else:
        new_array = extreme_case_0_array + extreme_case_1_array + extreme_case_2_array
        new_array_imgs = extreme_case_0_img + extreme_case_1_img + extreme_case_2_img

    return np.array(new_array), np.array(new_array_imgs)

def normalize(x):
    x = np.asarray(x)
    return (x - x.min()) / (np.ptp(x))

def check_path(path):
    if not os.path.exists(path):
        print(f"{path} not exist")
        os.makedirs(path)
        print(f"Create {path} success")
