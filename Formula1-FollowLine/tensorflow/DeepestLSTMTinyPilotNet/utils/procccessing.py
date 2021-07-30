import glob
import os
import cv2

import numpy as np

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt



def parse_json(data):
    array_annotations_v = []
    array_annotations_w = []
    array = []
    data_parse = data.split('}')[:-1]

    for number, d in enumerate(data_parse):
        v = d.split('"v": ')[1]
        d_parse = d.split(', "v":')[0]
        w = d_parse.split(('"w": '))[1]
        array_annotations_v.append(float(v))
        array_annotations_w.append(float(w))
        array.append((float(v), float(w)))
    return array

def get_images(list_images, type_image, image_shape):
    array_imgs = []
    for name in list_images:
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if type_image == 'crop':
            img = img[240:480, 0:640]
        img = cv2.resize(img, (image_shape[0], image_shape[1]))
        array_imgs.append(img)

    return array_imgs

def flip_images(images, array_annotations):
    flipped_images = []
    flipped_annotations = []
    for i, image in enumerate(images):
        flipped_images.append(cv2.flip(image, 1))
        flipped_annotations.append((array_annotations[i][0], -array_annotations[i][1]))
    
    images += flipped_images
    array_annotations += flipped_annotations
    return images, array_annotations

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

    normalized_X = normalize(array_annotations_v)
    normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    return normalized_annotations

def normalize(x):
    x = np.asarray(x)
    return (x - x.min()) / (np.ptp(x))

def read_dataset(path_to_data, type_image, image_shape, data_type):
    complete_name_file = '../complete_dataset/data.json'
    complete_file = open(complete_name_file, 'r')
    data_complete = complete_file.read()
    complete_file.close()

    array_annotations_complete = []
    DIR_complete_images = '../complete_dataset/Images/'
    list_images_complete = glob.glob(DIR_complete_images + '*')
    images_paths_complete = sorted(list_images_complete, key=lambda x: int(x.split('/')[3].split('.png')[0]))
    array_annotations_complete = parse_json(data_complete)

    images_complete = get_images(images_paths_complete, 'cropped')
    images_complete, array_annotations_complete = flip_images(images_complete, array_annotations_complete)

    array_annotations_complete = normalize_annotations(array_annotations_complete)

    print('---- Curves ----')
    curves_name_file = '../curves_only/data.json'
    file_curves = open(curves_name_file, 'r')
    data_curves = file_curves.read()
    file_curves.close()

    DIR_curves_images = '../curves_only/Images/'
    list_images_curves = glob.glob(DIR_curves_images + '*')
    images_paths_curves = sorted(list_images_curves, key=lambda x: int(x.split('/')[3].split('.png')[0]))
    array_annotations_curves = parse_json(data_curves)

    images_curves = get_images(images_paths_curves, 'cropped')
    images_curves, array_annotations_curves = flip_images(images_curves, array_annotations_curves)

    array_annotations_curves = normalize_annotations(array_annotations_curves)
