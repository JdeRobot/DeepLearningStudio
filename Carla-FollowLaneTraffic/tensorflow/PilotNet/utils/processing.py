import os
import glob
import cv2
import sys
import pandas
import random

import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from skimage.io import imread
from sklearn import preprocessing
from skimage.transform import resize
from sklearn.model_selection import train_test_split


def delete_ratio(annotations, images, value_min, value_max, x, type):
    new_annotations = []
    new_images = []

    annotations_steer = []
    for annotation in annotations:
        annotations_steer.append(annotation[type])
    print(sum(map(lambda i: ((i >= value_min) and (i <= value_max)), annotations_steer)))
    number_to_delete = np.round(sum(map(lambda i: ((i >= value_min) and (i <= value_max)), annotations_steer)) * x)
    eliminate = random.sample([i for i, val in enumerate(annotations_steer) if ((val >= value_min) and (val <= value_max))], int(number_to_delete))
    for index, (annotations_val, images_val) in enumerate(zip(annotations, images)):
        if index in eliminate:
            continue
        else:
            new_annotations.append(annotations_val)
            new_images.append(images_val)

    return new_annotations, new_images


def delete_until(annotations, images, x, counter, bins, type):
    new_annotations = []
    new_images = []

    annotations_steer = []
    for annotation in annotations:
        annotations_steer.append(annotation[type])

    total_elimination = []
    for index, counts in enumerate(counter):
        if counts > x:
            number_to_delete = counts - x
            eliminate = random.sample([i for i, val in enumerate(annotations_steer) if ((val >= bins[index]) and (val <= bins[index+1]))], int(number_to_delete))
            total_elimination.extend(eliminate)

    for index, (annotations_val, images_val) in enumerate(zip(annotations, images)):
        if index in total_elimination:
            continue
        else:
            new_annotations.append(annotations_val)
            new_images.append(images_val)

    return new_annotations, new_images


def get_images(list_images, type_image, image_shape, array_annotations):
    image_shape = (image_shape[1], image_shape[0])
    # Read the images
    array_imgs = []
    for index, name in enumerate(list_images):
        print(name)
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if type_image == 'crop':
            img = img[200:-1, :]
        img = cv2.resize(img, image_shape)
        array_imgs.append(img)

    return array_imgs


def parse_csv(csv_data):
    array = []
    linear_speeds = csv_data['throttle'].tolist()
    angular_speeds = csv_data['steer'].tolist()
    brake = csv_data['brake'].tolist()
    prevelocity = csv_data['prevelocity'].tolist()
    for x, linear_speed in enumerate(linear_speeds):
        array.append((float(linear_speed), float(angular_speeds[x]), float(brake[x]), float(prevelocity[x])))
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


def add_extreme_data(images, array_annotations):
    array_annotations = list(array_annotations)
    images = list(images)
    for i in range(0, len(array_annotations)):
        if abs(array_annotations[i][1]) >= 0.83015237:
            num_iter = 0
        elif abs(array_annotations[i][1]) >= 0.8825381:
            num_iter = 0
        elif abs(array_annotations[i][1]) >= 0.83015237:
            num_iter = 3
        elif abs(array_annotations[i][1]) >= 0.6555333:
            num_iter = 0
        elif abs(array_annotations[i][1]) >= 0.60314757:
            num_iter = 1
        elif abs(array_annotations[i][1]) >= 0.58568566:
            num_iter = 1
        elif abs(array_annotations[i][1]) >= 0.439:
            num_iter = 0
        elif abs(array_annotations[i][1]) >= 0.341219:
            num_iter = 0
        elif abs(array_annotations[i][1]) >= 0.30629514:
            num_iter = 0
        elif abs(array_annotations[i][1]) >= 0.25390942:
            num_iter = 5
        elif abs(array_annotations[i][1]) >= 0.20152369:
            num_iter = 25
        elif abs(array_annotations[i][1]) >= 0.114216:
            num_iter = 30
        else:
            num_iter = 0
        for j in range(0, num_iter):
            array_annotations.append(array_annotations[i])
            images.append(images[i])

    return images, array_annotations


def compute_image_annotations(id, path_to_data, type_image, img_shape, data_type):
    id = '_' + str(id)
    name_file = path_to_data + id + '/data.csv'
    dir_images = path_to_data + id + '/'
    list_images = glob.glob(dir_images + '*')
    new_list_images = []
    for image in list_images:
        if image != path_to_data + id + '/data.csv':
            new_list_images.append(image)
    list_images = new_list_images

    print(path_to_data)
    if 'weird' in path_to_data:
        images_paths = sorted(list_images, key=lambda x: int(x.split('/')[3].split('.png')[0]))
    elif 'extreme' in path_to_data:
        images_paths = sorted(list_images, key=lambda x: int(x.split('/')[3].split('.png')[0]))
    elif 'stops' in path_to_data:
        images_paths = sorted(list_images, key=lambda x: int(x.split('/')[3].split('.png')[0]))
    elif 'weather' in path_to_data:
        images_paths = sorted(list_images, key=lambda x: int(x.split('/')[3].split('.png')[0]))
    else:
        images_paths = sorted(list_images, key=lambda x: int(x.split('/')[2].split('.png')[0]))

    array_annotations = pandas.read_csv(name_file)
    array_annotations = parse_csv(array_annotations)

    images = get_images(images_paths, type_image, img_shape, array_annotations)
    if data_type == 'extreme':
        images, array_annotations = add_extreme_data(images, array_annotations)

    array_annotations_throttle = []
    array_annotations_steer = []
    array_annotations_brake = []
    array_annotations_throttle_brake = []
    array_annotations_prevelocity = []
    for annotation in array_annotations:
        print(annotation)
        array_annotations_throttle.append(annotation[0])
        array_annotations_steer.append(annotation[1])
        array_annotations_brake.append(annotation[2])
        array_annotations_prevelocity.append(annotation[3])

    array_annotations_throttle_brake = [x if x != 0.0 else -y for x, y in zip(array_annotations_throttle, array_annotations_brake)]

    # START NORMALIZE DATA
    array_annotations_throttle_brake = np.stack(array_annotations_throttle_brake, axis=0)
    array_annotations_throttle_brake = array_annotations_throttle_brake.reshape(-1, 1)

    array_annotations_steer = np.stack(array_annotations_steer, axis=0)
    array_annotations_steer = array_annotations_steer.reshape(-1, 1)

    array_annotations_prevelocity = np.stack(array_annotations_prevelocity, axis=0)
    array_annotations_prevelocity = array_annotations_prevelocity.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_throttle_brake, (-1, 1), (0, 1))
    normalized_y = np.interp(array_annotations_steer, (-1, 1), (0, 1))
    normalized_z = np.interp(array_annotations_prevelocity, (0, 100), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i), normalized_z.item(i)])

    array_annotations = normalized_annotations

    return images, array_annotations


def get_images_and_annotations(path_to_data, type_image, img_shape, data_type):
    list_dataset = glob.glob(path_to_data + '*_*')

    array_imgs = []
    array_annotations = []

    counter = 0
    for data in list_dataset:
        print(counter)
        id = data.split('_')[2]
        images, annotations = compute_image_annotations(id, path_to_data, type_image, img_shape, data_type)
        array_imgs += images
        array_annotations += annotations
        counter += 1

    list_weird_start = glob.glob(path_to_data + 'weird/*')
    for data in list_weird_start:
        id = data.split('_')[2]
        images, annotations = compute_image_annotations(id, path_to_data + 'weird/', type_image, img_shape, data_type)
        array_imgs += images
        array_annotations += annotations

    list_weird_start = glob.glob(path_to_data + 'extreme/*')
    for data in list_weird_start:
        id = data.split('_')[2]
        images, annotations = compute_image_annotations(id, path_to_data + 'extreme/', type_image, img_shape, data_type)
        array_imgs += images
        array_annotations += annotations

    list_weird_start = glob.glob(path_to_data + 'stops/*')
    for data in list_weird_start:
        id = data.split('_')[2]
        images, annotations = compute_image_annotations(id, path_to_data + 'stops/', type_image, img_shape, data_type)
        array_imgs += images
        array_annotations += annotations

    list_weird_start = glob.glob(path_to_data + 'weather/*')
    for data in list_weird_start:
        id = data.split('_')[2]
        images, annotations = compute_image_annotations(id, path_to_data + 'weather/', type_image, img_shape, data_type)
        array_imgs += images
        array_annotations += annotations

    return array_imgs, array_annotations


def separate_dataset_into_train_validation(array_x, array_y):
    images_train, images_validation, annotations_train, annotations_validation = train_test_split(array_x, array_y,
                                                                                                  test_size=0.30,
                                                                                                  random_state=42,
                                                                                                  shuffle=True)

    print('Images train -> ' + str(len(images_train)))
    print('Images validation -> ' + str(len(images_validation)))
    print('Annotations train -> ' + str(len(annotations_train)))
    print('Annotations validation -> ' + str(len(annotations_validation)))
    # Adapt the data
    images_train = np.stack(images_train, axis=0)
    annotations_train = np.stack(annotations_train, axis=0)
    images_validation = np.stack(images_validation, axis=0)
    annotations_validation = np.stack(annotations_validation, axis=0)

    print('Images train -> ' + str(images_train.shape))
    print('Images validation -> ' + str(images_validation.shape))
    print('Annotations train -> ' + str(annotations_train.shape))
    print('Annotations validation -> ' + str(annotations_validation.shape))

    return images_train, annotations_train, images_validation, annotations_validation


def process_dataset(path_to_data, type_image, data_type, img_shape):
    array_imgs = []
    array_annotations = []
    if os.path.exists('array_imgs.npy'):
        array_imgs = np.load('array_imgs.npy', allow_pickle=True)
        array_annotations = np.load('array_annotations.npy', allow_pickle=True)
    else:
        array_imgs, array_annotations = get_images_and_annotations(path_to_data, type_image, img_shape, data_type)
        # NORMALIZAMOS LAS IMAGENES
        np.save('array_imgs.npy', array_imgs, allow_pickle=True)
        np.save('array_annotations.npy', array_annotations, allow_pickle=True)



    # STEERING
    plt.hist(np.array(array_annotations)[:,1],bins=50)
    plt.show()


    # ACCELERATION AND BRAKE
    plt.hist(np.array(array_annotations)[:,0],bins=50)
    plt.show()

    images_train, annotations_train, images_validation, annotations_validation = separate_dataset_into_train_validation(
        array_imgs, array_annotations)

    return images_train, annotations_train, images_validation, annotations_validation
