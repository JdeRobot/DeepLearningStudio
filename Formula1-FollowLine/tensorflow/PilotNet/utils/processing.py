import glob
import cv2

import numpy as np

from skimage.io import imread
from skimage.transform import resize

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

def add_extreme_data(images, array_annotations):
    for i in range(0, len(array_annotations)):
        if abs(array_annotations[i][1]) >= 1:
            if abs(array_annotations[i][1]) >= 2:
                #num_iter = 10
                #num_iter = 15
                num_iter = 20
            else:
                #num_iter = 5
                #num_iter = 10
                num_iter = 15
            for j in range(0, num_iter):
                array_annotations.append(array_annotations[i])
                images.append(images[i])
        if float(array_annotations[i][0]) <= 2:
            #for j in range(0, 1):
            #for j in range(0, 5):
            for j in range(0, 10):
                array_annotations.append(array_annotations[i])
                images.append(images[i])
                
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
    train_name_file = path_to_data + 'Train/train.json'
    file_train = open(train_name_file, 'r')
    data_train = file_train.read()
    file_train.close()

    array_annotations_train = []
    DIR_train_images = path_to_data + 'Train/Images/'
    list_images_train = glob.glob(DIR_train_images + '*')
    images_paths_train = sorted(list_images_train, key=lambda x: int(x.split('/')[7].split('.png')[0]))
    array_annotations_train = parse_json(data_train)

    images_train = get_images(images_paths_train, type_image, image_shape)
    images_train, array_annotations_train = flip_images(images_train, array_annotations_train)
    if data_type == 'extreme':
        images_train, array_annotations_train = add_extreme_data(images_train, array_annotations_train)

    array_annotations_train = normalize_annotations(array_annotations_train)

    # Validation
    test_name_file = path_to_data + 'Test/test.json'
    file_test = open(test_name_file, 'r')
    data_test = file_test.read()
    file_test.close()

    DIR_test_images = path_to_data + 'Test/Images/'
    list_images_val = glob.glob(DIR_test_images + '*')
    images_paths_val = sorted(list_images_val, key=lambda x: int(x.split('/')[7].split('.png')[0]))
    array_annotations_val = parse_json(data_test)

    images_val = get_images(images_paths_val, type_image, image_shape)
    images_val, array_annotations_val = flip_images(images_val, array_annotations_val)
    if data_type == 'extreme':
        images_val, array_annotations_val = add_extreme_data(images_val, array_annotations_val)

    array_annotations_val = normalize_annotations(array_annotations_val)

    return images_train, array_annotations_train, images_val, array_annotations_val