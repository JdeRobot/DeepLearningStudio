import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
import csv


def load_data(folder):
    name_folder = folder #+ '/' #+ '/Images/'
    list_images = glob.glob(name_folder + '*.png')
    images = sorted(list_images, key=lambda x: int(x.split('/')[-1].split('.png')[0]))
    name_file = folder + 'data.csv' #'/data.json'
    file = open(name_file, 'r')
    reader = csv.DictReader(file)
    data = []
    for row in reader: # reading all values
        data.append((row['v'], row['w']))
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
        else:
            target_height = int(66)
            target_width = int(target_height * img.shape[1]/img.shape[0])
            img_resized = cv2.resize(img, (target_width, target_height))
            padding_left = int((200 - target_width)/2)
            padding_right = 200 - target_width - padding_left
            img = cv2.copyMakeBorder(img_resized.copy(),0,0,padding_left,padding_right,cv2.BORDER_CONSTANT,value=[0, 0, 0])
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

def parse_csv(data, array):
    # Process csv
    for v, w in data:
        array.append((float(v), float(w)))

    return array

def preprocess_data(array, imgs, data_type):
    # Data augmentation
    # Take the image and just flip it and negate the measurement
    flip_imgs = []
    array_flip = []
    for i in tqdm(range(len(imgs))):
        flip_imgs.append(cv2.flip(imgs[i], 1))
        array_flip.append((array[i][0], -array[i][1]))
    new_array = array + array_flip
    new_array_imgs = imgs + flip_imgs

    if data_type == 'extreme':
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

    new_array = normalize_annotations(new_array)

    return new_array, new_array_imgs

def normalize_annotations(array_annotations):
    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])
        
    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_X = normalize(array_annotations_v, min=6.5, max=24)
    normalized_Y = normalize(array_annotations_w, min=-7.1, max=7.1)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    return normalized_annotations

def normalize(x, min, max):
    x = np.asarray(x)
    return (x - min) / (max - min)

def check_path(path):
    if not os.path.exists(path):
        print(f"{path} not exist")
        os.makedirs(path)
        print(f"Create {path} success")
