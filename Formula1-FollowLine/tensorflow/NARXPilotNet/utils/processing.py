import glob
import os
import cv2
import random

import numpy as np

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def get_images(list_images, type_image, img_shape):
    # Read the images
    array_imgs = []
    for name in list_images:
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if type_image == 'cropped':
            img = img[240:480, 0:640]
        img = cv2.resize(img, (img_shape[1], img_shape[0]))
        array_imgs.append(img)

    return array_imgs
    

def parse_json(data):
    # Process json
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
    

def flip_images(images, array_annotations):
    flipped_images = []
    flipped_annotations = []
    for i, image in enumerate(images):
        flipped_images.append(cv2.flip(image, 1))
        flipped_annotations.append((array_annotations[i][0], -array_annotations[i][1]))

    images += flipped_images
    array_annotations += flipped_annotations
    return images, array_annotations
    
    
def normalize(x):
    x = np.asarray(x)
    return (x - x.min()) / (np.ptp(x))
    

def get_images_and_annotations(path_to_data, type_image, img_shape):
    print('---- Complete ----')
    complete_name_file = path_to_data + 'complete_dataset/data.json'
    complete_file = open(complete_name_file, 'r')
    data_complete = complete_file.read()
    complete_file.close()

    DIR_complete_images = path_to_data + 'complete_dataset/Images/'
    list_images_complete = glob.glob(DIR_complete_images + '*')
    images_paths_complete = sorted(list_images_complete, key=lambda x: int(x.split('/')[-1].split('.png')[0]))
    array_annotations_complete = parse_json(data_complete)

    images_complete = get_images(images_paths_complete, type_image, img_shape)
    images_complete, array_annotations_complete = flip_images(images_complete, array_annotations_complete)
    print(len(images_complete))
    print(type(images_complete))
    print(len(array_annotations_complete))

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_complete:
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

    array_annotations_complete = normalized_annotations

    print(len(images_complete))
    print(type(images_complete))
    print(len(array_annotations_complete))

    print('---- Curves ----')
    curves_name_file = path_to_data + 'curves_only/data.json'
    file_curves = open(curves_name_file, 'r')
    data_curves = file_curves.read()
    file_curves.close()

    DIR_curves_images = path_to_data + 'curves_only/Images/'
    list_images_curves = glob.glob(DIR_curves_images + '*')
    images_paths_curves = sorted(list_images_curves, key=lambda x: int(x.split('/')[-1].split('.png')[0]))
    array_annotations_curves = parse_json(data_curves)

    images_curves = get_images(images_paths_curves, type_image, img_shape)
    images_curves, array_annotations_curves = flip_images(images_curves, array_annotations_curves)
    print(len(images_curves))
    print(type(images_curves))
    print(len(array_annotations_curves))

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_curves:
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

    array_annotations_curves = normalized_annotations

    print(len(images_curves))
    print(type(images_curves))
    print(len(array_annotations_curves))

    ###############################################################################################################

    array_imgs = images_complete + images_curves
    array_annotations = array_annotations_complete + array_annotations_curves

    return array_imgs, array_annotations
    
    
def separate_dataset(array_imgs, array_annotations):
    sequences = [
        (0, 3700), (3745, 5045), (5067, 9717), (9721, 10371), (10388, 10688),
        (10696, 11246), (11284, 11334), (11355, 11455), (11493, 11943),
        (11981, 12581), (12619, 13219), (13232, 14082), (14108, 15758), 
        (15791, 17291), (17341, 20491), (20498, 22598), (22609, 26309), 
        (26354, 27654), (27676, 32326), (32330, 32930), (32997, 33297),
        (33305, 33855), (33893, 33943), (33964, 34064), (34102, 34552),
        (34590, 35190), (35228, 35828), (35841, 36691), (36717, 38367),
        (38400, 39300), (39405, 39905), (39950, 43100), (43107, 45157)
    ]
    
    img_array = []
    prev_ann_array = []
    ann_array = []

    for sequence in sequences:
        for i in range(sequence[0] + 1, sequence[1]):
            img_array.append(array_imgs[i])
            prev_ann_array.append(array_annotations[i - 1])
            ann_array.append(array_annotations[i])
            
    return img_array, prev_ann_array, ann_array
    

def add_extreme_data(images, prev_annotations, annotations):
    for i in range(0, len(annotations)):
        if abs(annotations[i][1]) >= 1:
            if abs(annotations[i][1]) >= 2:
                num_iter = 10
            else:
                num_iter = 5
                
            if annotations[i][1] > 2:
                num_iter += 10
            elif annotations[i][1] > 1:
                num_iter += 5
                
            for j in range(0, num_iter):
                annotations.append(annotations[i])
                prev_annotations.append(prev_annotations[i])
                images.append(images[i])

                
    return images, prev_annotations, annotations


def separate_dataset_into_train_validation(images, prev_annotations, annotations):
    output = train_test_split(images, prev_annotations, annotations, test_size = 0.30, random_state = 42, shuffle = False)
    
    images_train = output[0]
    images_validation = output[1]
    p_annotations_train = output[2]
    p_annotations_validation = output[3]
    annotations_train = output[4]
    annotations_validation = output[5]

    # Adapt the data
    images_train = np.stack(images_train, axis=0)
    annotations_train = np.stack(annotations_train, axis=0)
    p_annotations_train = np.stack(p_annotations_train, axis = 0)
    images_validation = np.stack(images_validation, axis=0)
    annotations_validation = np.stack(annotations_validation, axis=0)
    p_annotations_validation = np.stack(p_annotations_validation, axis = 0)

    return images_train, p_annotations_train, annotations_train, images_validation, p_annotations_validation, annotations_validation
    

def process_dataset(path_to_data, type_image, data_type, img_shape):
    array_imgs, array_annotations = get_images_and_annotations(path_to_data, type_image, img_shape)
    array_imgs, array_prev_annotations, array_annotations = separate_dataset(array_imgs, array_annotations)
    if data_type == 'extreme':
        array_imgs, array_prev_annotations, array_annotations = add_extreme_data(array_imgs, array_prev_annotations, array_annotations)
    
    train_val_data = separate_dataset_into_train_validation(array_imgs, array_prev_annotations, array_annotations)

    return train_val_data