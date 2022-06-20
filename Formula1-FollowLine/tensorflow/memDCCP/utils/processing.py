import glob
import os
import cv2
import random
import pandas

import numpy as np

from sklearn.model_selection import train_test_split


def get_images(list_images, type_image, image_shape):
    image_shape = (image_shape[1], image_shape[2])
    # Read the images
    array_imgs = []
    for name in list_images:
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if type_image == 'cropped':
            img = img[240:480, 0:640]
        img = cv2.resize(img, image_shape)
        array_imgs.append(img)

    return array_imgs


def parse_csv(csv_data):
    array = []
    linear_speeds = csv_data['v'].tolist()
    angular_speeds = csv_data['w'].tolist()
    for x, linear_speed in enumerate(linear_speeds):
        array.append((float(linear_speed), float(angular_speeds[x])))
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
    ######################################### 1 ########################################
    many_curves_1_name_file = path_to_data + 'many_curves_01_04_2022_clockwise_1/data.csv'

    dir_many_curves_1_images = path_to_data + 'many_curves_01_04_2022_clockwise_1/'
    list_images_many_curves_1 = glob.glob(dir_many_curves_1_images + '*')
    new_list_images_many_curves_1 = []
    for image in list_images_many_curves_1:
        if image != path_to_data + 'many_curves_01_04_2022_clockwise_1/data.csv':
            new_list_images_many_curves_1.append(image)
    list_images_many_curves_1 = new_list_images_many_curves_1
    images_paths_many_curves_1 = sorted(list_images_many_curves_1, key=lambda x: int(x.split('/')[6].split('.png')[0]))

    array_annotations_many_curves_1 = pandas.read_csv(many_curves_1_name_file)
    array_annotations_many_curves_1 = parse_csv(array_annotations_many_curves_1)

    images_many_curves_1 = get_images(images_paths_many_curves_1, type_image, img_shape)
    images_many_curves_1, array_annotations_many_curves_1 = flip_images(images_many_curves_1,
                                                                        array_annotations_many_curves_1)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_many_curves_1:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_many_curves_1 = normalized_annotations

    ########################################## 2 ########################################
    nurburgring_1_name_file = path_to_data + 'nurburgring_01_04_2022_clockwise_1/data.csv'

    dir_nurburgring_1_images = path_to_data + 'nurburgring_01_04_2022_clockwise_1/'
    list_images_nurburgring_1 = glob.glob(dir_nurburgring_1_images + '*')
    new_list_images_nurburgring_1 = []
    for image in list_images_nurburgring_1:
        if image != path_to_data + 'nurburgring_01_04_2022_clockwise_1/data.csv':
            new_list_images_nurburgring_1.append(image)
    list_images_nurburgring_1 = new_list_images_nurburgring_1
    images_paths_nurburgring_1 = sorted(list_images_nurburgring_1, key=lambda x: int(x.split('/')[6].split('.png')[0]))

    array_annotations_nurburgring_1 = pandas.read_csv(nurburgring_1_name_file)
    array_annotations_nurburgring_1 = parse_csv(array_annotations_nurburgring_1)

    images_nurburgring_1 = get_images(images_paths_nurburgring_1, type_image, img_shape)
    images_nurburgring_1, array_annotations_nurburgring_1 = flip_images(images_nurburgring_1,
                                                                        array_annotations_nurburgring_1)
    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_nurburgring_1:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_nurburgring_1 = normalized_annotations

    ######################################### 3 #########################################
    monaco_1_name_file = path_to_data + 'monaco_01_04_2022_clockwise_1/data.csv'

    dir_monaco_1_images = path_to_data + 'monaco_01_04_2022_clockwise_1/'
    list_images_monaco_1 = glob.glob(dir_monaco_1_images + '*')
    new_list_images_monaco_1 = []
    for image in list_images_monaco_1:
        if image != path_to_data + 'monaco_01_04_2022_clockwise_1/data.csv':
            new_list_images_monaco_1.append(image)
    list_images_monaco_1 = new_list_images_monaco_1
    images_paths_monaco_1 = sorted(list_images_monaco_1, key=lambda x: int(x.split('/')[6].split('.png')[0]))

    array_annotations_monaco_1 = pandas.read_csv(monaco_1_name_file)
    array_annotations_monaco_1 = parse_csv(array_annotations_monaco_1)

    images_monaco_1 = get_images(images_paths_monaco_1, type_image, img_shape)
    images_monaco_1, array_annotations_monaco_1 = flip_images(images_monaco_1, array_annotations_monaco_1)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_monaco_1:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_monaco_1 = normalized_annotations

    ######################################### 4 #########################################
    extended_simple_1_name_file = path_to_data + 'extended_simple_circuit_01_04_2022_clockwise_1/data.csv'

    dir_extended_simple_1_images = path_to_data + 'extended_simple_circuit_01_04_2022_clockwise_1/'
    list_images_extended_simple_1 = glob.glob(dir_extended_simple_1_images + '*')
    new_list_images_extended_simple_1 = []
    for image in list_images_extended_simple_1:
        if image != path_to_data + 'extended_simple_circuit_01_04_2022_clockwise_1/data.csv':
            new_list_images_extended_simple_1.append(image)
    list_images_extended_simple_1 = new_list_images_extended_simple_1
    images_paths_extended_simple_1 = sorted(list_images_extended_simple_1,
                                            key=lambda x: int(x.split('/')[6].split('.png')[0]))

    array_annotations_extended_simple_1 = pandas.read_csv(extended_simple_1_name_file)
    array_annotations_extended_simple_1 = parse_csv(array_annotations_extended_simple_1)

    images_extended_simple_1 = get_images(images_paths_extended_simple_1, type_image, img_shape)
    images_extended_simple_1, array_annotations_extended_simple_1 = flip_images(images_extended_simple_1,
                                                                                array_annotations_extended_simple_1)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_extended_simple_1:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_extended_simple_1 = normalized_annotations

    ######################################### 5 #########################################
    only_curves_1_name_file = path_to_data + 'only_curves_01_04_2022/nurburgring_1/data.csv'

    dir_only_curves_1_images = path_to_data + 'only_curves_01_04_2022/nurburgring_1/'
    list_images_only_curves_1 = glob.glob(dir_only_curves_1_images + '*')
    new_list_images_only_curves_1 = []
    for image in list_images_only_curves_1:
        if image != path_to_data + 'only_curves_01_04_2022/nurburgring_1/data.csv':
            new_list_images_only_curves_1.append(image)
    list_images_only_curves_1 = new_list_images_only_curves_1
    images_paths_only_curves_1 = sorted(list_images_only_curves_1, key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_1 = pandas.read_csv(only_curves_1_name_file)
    array_annotations_only_curves_1 = parse_csv(array_annotations_only_curves_1)

    images_only_curves_1 = get_images(images_paths_only_curves_1, type_image, img_shape)
    images_only_curves_1, array_annotations_only_curves_1 = flip_images(images_only_curves_1,
                                                                        array_annotations_only_curves_1)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_only_curves_1:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_only_curves_1 = normalized_annotations

    ######################################### 6 #########################################
    only_curves_2_name_file = path_to_data + 'only_curves_01_04_2022/nurburgring_2/data.csv'

    dir_only_curves_2_images = path_to_data + 'only_curves_01_04_2022/nurburgring_2/'
    list_images_only_curves_2 = glob.glob(dir_only_curves_2_images + '*')
    new_list_images_only_curves_2 = []
    for image in list_images_only_curves_2:
        if image != path_to_data + 'only_curves_01_04_2022/nurburgring_2/data.csv':
            new_list_images_only_curves_2.append(image)
    list_images_only_curves_2 = new_list_images_only_curves_2
    images_paths_only_curves_2 = sorted(list_images_only_curves_2, key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_2 = pandas.read_csv(only_curves_2_name_file)
    array_annotations_only_curves_2 = parse_csv(array_annotations_only_curves_2)

    images_only_curves_2 = get_images(images_paths_only_curves_2, type_image, img_shape)
    images_only_curves_2, array_annotations_only_curves_2 = flip_images(images_only_curves_2,
                                                                        array_annotations_only_curves_2)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_only_curves_2:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_only_curves_2 = normalized_annotations

    ######################################### 7 #########################################
    only_curves_3_name_file = path_to_data + 'only_curves_01_04_2022/nurburgring_3/data.csv'

    dir_only_curves_3_images = path_to_data + 'only_curves_01_04_2022/nurburgring_3/'
    list_images_only_curves_3 = glob.glob(dir_only_curves_3_images + '*')
    new_list_images_only_curves_3 = []
    for image in list_images_only_curves_3:
        if image != path_to_data + 'only_curves_01_04_2022/nurburgring_3/data.csv':
            new_list_images_only_curves_3.append(image)
    list_images_only_curves_3 = new_list_images_only_curves_3
    images_paths_only_curves_3 = sorted(list_images_only_curves_3, key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_3 = pandas.read_csv(only_curves_3_name_file)
    array_annotations_only_curves_3 = parse_csv(array_annotations_only_curves_3)

    images_only_curves_3 = get_images(images_paths_only_curves_3, type_image, img_shape)
    images_only_curves_3, array_annotations_only_curves_3 = flip_images(images_only_curves_3,
                                                                        array_annotations_only_curves_3)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_only_curves_3:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_only_curves_3 = normalized_annotations

    ######################################### 8 #########################################
    only_curves_4_name_file = path_to_data + 'only_curves_01_04_2022/nurburgring_4/data.csv'

    dir_only_curves_4_images = path_to_data + 'only_curves_01_04_2022/nurburgring_4/'
    list_images_only_curves_4 = glob.glob(dir_only_curves_4_images + '*')
    new_list_images_only_curves_4 = []
    for image in list_images_only_curves_4:
        if image != path_to_data + 'only_curves_01_04_2022/nurburgring_4/data.csv':
            new_list_images_only_curves_4.append(image)
    list_images_only_curves_4 = new_list_images_only_curves_4
    images_paths_only_curves_4 = sorted(list_images_only_curves_4, key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_4 = pandas.read_csv(only_curves_4_name_file)
    array_annotations_only_curves_4 = parse_csv(array_annotations_only_curves_4)

    images_only_curves_4 = get_images(images_paths_only_curves_4, type_image, img_shape)
    images_only_curves_4, array_annotations_only_curves_4 = flip_images(images_only_curves_4,
                                                                        array_annotations_only_curves_4)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_only_curves_4:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_only_curves_4 = normalized_annotations

    ######################################### 9 #########################################
    only_curves_5_name_file = path_to_data + 'only_curves_01_04_2022/nurburgring_5/data.csv'

    dir_only_curves_5_images = path_to_data + 'only_curves_01_04_2022/nurburgring_5/'
    list_images_only_curves_5 = glob.glob(dir_only_curves_5_images + '*')
    new_list_images_only_curves_5 = []
    for image in list_images_only_curves_5:
        if image != path_to_data + 'only_curves_01_04_2022/nurburgring_5/data.csv':
            new_list_images_only_curves_5.append(image)
    list_images_only_curves_5 = new_list_images_only_curves_5
    images_paths_only_curves_5 = sorted(list_images_only_curves_5, key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_5 = pandas.read_csv(only_curves_5_name_file)
    array_annotations_only_curves_5 = parse_csv(array_annotations_only_curves_5)

    images_only_curves_5 = get_images(images_paths_only_curves_5, type_image, img_shape)
    images_only_curves_5, array_annotations_only_curves_5 = flip_images(images_only_curves_5,
                                                                        array_annotations_only_curves_5)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_only_curves_5:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_only_curves_5 = normalized_annotations

    ######################################### 10 #########################################
    only_curves_6_name_file = path_to_data + 'only_curves_01_04_2022/nurburgring_6/data.csv'

    dir_only_curves_6_images = path_to_data + 'only_curves_01_04_2022/nurburgring_6/'
    list_images_only_curves_6 = glob.glob(dir_only_curves_6_images + '*')
    new_list_images_only_curves_6 = []
    for image in list_images_only_curves_6:
        if image != path_to_data + 'only_curves_01_04_2022/nurburgring_6/data.csv':
            new_list_images_only_curves_6.append(image)
    list_images_only_curves_6 = new_list_images_only_curves_6
    images_paths_only_curves_6 = sorted(list_images_only_curves_6, key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_6 = pandas.read_csv(only_curves_6_name_file)
    array_annotations_only_curves_6 = parse_csv(array_annotations_only_curves_6)

    images_only_curves_6 = get_images(images_paths_only_curves_6, type_image, img_shape)
    images_only_curves_6, array_annotations_only_curves_6 = flip_images(images_only_curves_6,
                                                                        array_annotations_only_curves_6)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_only_curves_6:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_only_curves_6 = normalized_annotations

    ######################################### 11 #########################################
    only_curves_7_name_file = path_to_data + '/only_curves_01_04_2022/monaco_1/data.csv'

    dir_only_curves_7_images = path_to_data + 'only_curves_01_04_2022/monaco_1/'
    list_images_only_curves_7 = glob.glob(dir_only_curves_7_images + '*')
    new_list_images_only_curves_7 = []
    for image in list_images_only_curves_7:
        if image != path_to_data + 'only_curves_01_04_2022/monaco_1/data.csv':
            new_list_images_only_curves_7.append(image)
    list_images_only_curves_7 = new_list_images_only_curves_7
    images_paths_only_curves_7 = sorted(list_images_only_curves_7, key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_7 = pandas.read_csv(only_curves_7_name_file)
    array_annotations_only_curves_7 = parse_csv(array_annotations_only_curves_7)

    images_only_curves_7 = get_images(images_paths_only_curves_7, type_image, img_shape)
    images_only_curves_7, array_annotations_only_curves_7 = flip_images(images_only_curves_7,
                                                                        array_annotations_only_curves_7)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_only_curves_7:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_only_curves_7 = normalized_annotations

    ######################################### 12 #########################################
    only_curves_8_name_file = path_to_data + 'only_curves_01_04_2022/monaco_2/data.csv'

    dir_only_curves_8_images = path_to_data + 'only_curves_01_04_2022/monaco_2/'
    list_images_only_curves_8 = glob.glob(dir_only_curves_8_images + '*')
    new_list_images_only_curves_8 = []
    for image in list_images_only_curves_8:
        if image != path_to_data + 'only_curves_01_04_2022/monaco_2/data.csv':
            new_list_images_only_curves_8.append(image)
    list_images_only_curves_8 = new_list_images_only_curves_8
    images_paths_only_curves_8 = sorted(list_images_only_curves_8, key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_8 = pandas.read_csv(only_curves_8_name_file)
    array_annotations_only_curves_8 = parse_csv(array_annotations_only_curves_8)

    images_only_curves_8 = get_images(images_paths_only_curves_8, type_image, img_shape)
    images_only_curves_8, array_annotations_only_curves_8 = flip_images(images_only_curves_8,
                                                                        array_annotations_only_curves_8)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_only_curves_8:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_only_curves_8 = normalized_annotations

    ######################################### 13 #########################################
    only_curves_9_name_file = path_to_data + 'only_curves_01_04_2022/monaco_3/data.csv'

    dir_only_curves_9_images = path_to_data + 'only_curves_01_04_2022/monaco_3/'
    list_images_only_curves_9 = glob.glob(dir_only_curves_9_images + '*')
    new_list_images_only_curves_9 = []
    for image in list_images_only_curves_9:
        if image != path_to_data + 'only_curves_01_04_2022/monaco_3/data.csv':
            new_list_images_only_curves_9.append(image)
    list_images_only_curves_9 = new_list_images_only_curves_9
    images_paths_only_curves_9 = sorted(list_images_only_curves_9, key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_9 = pandas.read_csv(only_curves_9_name_file)
    array_annotations_only_curves_9 = parse_csv(array_annotations_only_curves_9)

    images_only_curves_9 = get_images(images_paths_only_curves_9, type_image, img_shape)
    images_only_curves_9, array_annotations_only_curves_9 = flip_images(images_only_curves_9,
                                                                        array_annotations_only_curves_9)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_only_curves_9:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_only_curves_9 = normalized_annotations

    ######################################### 14 #########################################
    only_curves_10_name_file = path_to_data + 'only_curves_01_04_2022/monaco_4/data.csv'

    dir_only_curves_10_images = path_to_data + 'only_curves_01_04_2022/monaco_4/'
    list_images_only_curves_10 = glob.glob(dir_only_curves_10_images + '*')
    new_list_images_only_curves_10 = []
    for image in list_images_only_curves_10:
        if image != path_to_data + 'only_curves_01_04_2022/monaco_4/data.csv':
            new_list_images_only_curves_10.append(image)
    list_images_only_curves_10 = new_list_images_only_curves_10
    images_paths_only_curves_10 = sorted(list_images_only_curves_10,
                                         key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_10 = pandas.read_csv(only_curves_10_name_file)
    array_annotations_only_curves_10 = parse_csv(array_annotations_only_curves_10)

    images_only_curves_10 = get_images(images_paths_only_curves_10, type_image, img_shape)
    images_only_curves_10, array_annotations_only_curves_10 = flip_images(images_only_curves_10,
                                                                          array_annotations_only_curves_10)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_only_curves_10:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_only_curves_10 = normalized_annotations

    ######################################### 15 #########################################
    only_curves_11_name_file = path_to_data + 'only_curves_01_04_2022/many_curves_1/data.csv'

    dir_only_curves_11_images = path_to_data + 'only_curves_01_04_2022/many_curves_1/'
    list_images_only_curves_11 = glob.glob(dir_only_curves_11_images + '*')
    new_list_images_only_curves_11 = []
    for image in list_images_only_curves_11:
        if image != path_to_data + 'only_curves_01_04_2022/many_curves_1/data.csv':
            new_list_images_only_curves_11.append(image)
    list_images_only_curves_11 = new_list_images_only_curves_11
    images_paths_only_curves_11 = sorted(list_images_only_curves_11,
                                         key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_11 = pandas.read_csv(only_curves_11_name_file)
    array_annotations_only_curves_11 = parse_csv(array_annotations_only_curves_11)

    images_only_curves_11 = get_images(images_paths_only_curves_11, type_image, img_shape)
    images_only_curves_11, array_annotations_only_curves_11 = flip_images(images_only_curves_11,
                                                                          array_annotations_only_curves_11)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_only_curves_11:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_only_curves_11 = normalized_annotations

    ######################################### 16 #########################################
    only_curves_12_name_file = path_to_data + 'only_curves_01_04_2022/many_curves_2/data.csv'

    dir_only_curves_12_images = path_to_data + 'only_curves_01_04_2022/many_curves_2/'
    list_images_only_curves_12 = glob.glob(dir_only_curves_12_images + '*')
    new_list_images_only_curves_12 = []
    for image in list_images_only_curves_12:
        if image != path_to_data + 'only_curves_01_04_2022/many_curves_2/data.csv':
            new_list_images_only_curves_12.append(image)
    list_images_only_curves_12 = new_list_images_only_curves_12
    images_paths_only_curves_12 = sorted(list_images_only_curves_12,
                                         key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_12 = pandas.read_csv(only_curves_12_name_file)
    array_annotations_only_curves_12 = parse_csv(array_annotations_only_curves_12)

    images_only_curves_12 = get_images(images_paths_only_curves_12, type_image, img_shape)
    images_only_curves_12, array_annotations_only_curves_12 = flip_images(images_only_curves_12,
                                                                          array_annotations_only_curves_12)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_only_curves_12:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_only_curves_12 = normalized_annotations

    ######################################### 17 #########################################
    difficult_situations_1_name_file = path_to_data + 'difficult_situations_01_04_2022/many_curves_1/data.csv'

    dir_difficult_situations_1_images = path_to_data + 'difficult_situations_01_04_2022/many_curves_1/'
    list_images_difficult_situations_1 = glob.glob(dir_difficult_situations_1_images + '*')
    new_list_images_difficult_situations_1 = []
    for image in list_images_difficult_situations_1:
        if image != path_to_data + 'difficult_situations_01_04_2022/many_curves_1/data.csv':
            new_list_images_difficult_situations_1.append(image)
    list_images_difficult_situations_1 = new_list_images_difficult_situations_1
    images_paths_difficult_situations_1 = sorted(list_images_difficult_situations_1,
                                                 key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_1 = pandas.read_csv(difficult_situations_1_name_file)
    array_annotations_difficult_situations_1 = parse_csv(array_annotations_difficult_situations_1)

    images_difficult_situations_1 = get_images(images_paths_difficult_situations_1, type_image, img_shape)
    images_difficult_situations_1, array_annotations_difficult_situations_1 = flip_images(images_difficult_situations_1,
                                                                                          array_annotations_difficult_situations_1)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_1:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_1 = normalized_annotations

    ######################################### 18 #########################################
    difficult_situations_2_name_file = path_to_data + 'difficult_situations_01_04_2022/many_curves_2/data.csv'

    dir_difficult_situations_2_images = path_to_data + 'difficult_situations_01_04_2022/many_curves_2/'
    list_images_difficult_situations_2 = glob.glob(dir_difficult_situations_2_images + '*')
    new_list_images_difficult_situations_2 = []
    for image in list_images_difficult_situations_2:
        if image != path_to_data + 'difficult_situations_01_04_2022/many_curves_2/data.csv':
            new_list_images_difficult_situations_2.append(image)
    list_images_difficult_situations_2 = new_list_images_difficult_situations_2
    images_paths_difficult_situations_2 = sorted(list_images_difficult_situations_2,
                                                 key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_2 = pandas.read_csv(difficult_situations_2_name_file)
    array_annotations_difficult_situations_2 = parse_csv(array_annotations_difficult_situations_2)

    images_difficult_situations_2 = get_images(images_paths_difficult_situations_2, type_image, img_shape)
    images_difficult_situations_2, array_annotations_difficult_situations_2 = flip_images(images_difficult_situations_2,
                                                                                          array_annotations_difficult_situations_2)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_2:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_2 = normalized_annotations

    ######################################### 19 #########################################
    difficult_situations_3_name_file = path_to_data + 'difficult_situations_01_04_2022/many_curves_3/data.csv'

    dir_difficult_situations_3_images = path_to_data + 'difficult_situations_01_04_2022/many_curves_3/'
    list_images_difficult_situations_3 = glob.glob(dir_difficult_situations_3_images + '*')
    new_list_images_difficult_situations_3 = []
    for image in list_images_difficult_situations_3:
        if image != path_to_data + 'difficult_situations_01_04_2022/many_curves_3/data.csv':
            new_list_images_difficult_situations_3.append(image)
    list_images_difficult_situations_3 = new_list_images_difficult_situations_3
    images_paths_difficult_situations_3 = sorted(list_images_difficult_situations_3,
                                                 key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_3 = pandas.read_csv(difficult_situations_3_name_file)
    array_annotations_difficult_situations_3 = parse_csv(array_annotations_difficult_situations_3)

    images_difficult_situations_3 = get_images(images_paths_difficult_situations_3, type_image, img_shape)
    images_difficult_situations_3, array_annotations_difficult_situations_3 = flip_images(images_difficult_situations_3,
                                                                                          array_annotations_difficult_situations_3)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_3:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_3 = normalized_annotations

    ######################################### 20 #########################################
    difficult_situations_4_name_file = path_to_data + 'difficult_situations_01_04_2022/many_curves_4/data.csv'

    dir_difficult_situations_4_images = path_to_data + 'difficult_situations_01_04_2022/many_curves_4/'
    list_images_difficult_situations_4 = glob.glob(dir_difficult_situations_4_images + '*')
    new_list_images_difficult_situations_4 = []
    for image in list_images_difficult_situations_4:
        if image != path_to_data + 'difficult_situations_01_04_2022/many_curves_4/data.csv':
            new_list_images_difficult_situations_4.append(image)
    list_images_difficult_situations_4 = new_list_images_difficult_situations_4
    images_paths_difficult_situations_4 = sorted(list_images_difficult_situations_4,
                                                 key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_4 = pandas.read_csv(difficult_situations_4_name_file)
    array_annotations_difficult_situations_4 = parse_csv(array_annotations_difficult_situations_4)

    images_difficult_situations_4 = get_images(images_paths_difficult_situations_4, type_image, img_shape)
    images_difficult_situations_4, array_annotations_difficult_situations_4 = flip_images(images_difficult_situations_4,
                                                                                          array_annotations_difficult_situations_4)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_4:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_4 = normalized_annotations

    ######################################### 21 #########################################
    difficult_situations_5_name_file = path_to_data + 'difficult_situations_01_04_2022/monaco_1/data.csv'

    dir_difficult_situations_5_images = path_to_data + 'difficult_situations_01_04_2022/monaco_1/'
    list_images_difficult_situations_5 = glob.glob(dir_difficult_situations_5_images + '*')
    new_list_images_difficult_situations_5 = []
    for image in list_images_difficult_situations_5:
        if image != path_to_data + 'difficult_situations_01_04_2022/monaco_1/data.csv':
            new_list_images_difficult_situations_5.append(image)
    list_images_difficult_situations_5 = new_list_images_difficult_situations_5
    images_paths_difficult_situations_5 = sorted(list_images_difficult_situations_5,
                                                 key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_5 = pandas.read_csv(difficult_situations_5_name_file)
    array_annotations_difficult_situations_5 = parse_csv(array_annotations_difficult_situations_5)

    images_difficult_situations_5 = get_images(images_paths_difficult_situations_5, type_image, img_shape)
    images_difficult_situations_5, array_annotations_difficult_situations_5 = flip_images(images_difficult_situations_5,
                                                                                          array_annotations_difficult_situations_5)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_5:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_5 = normalized_annotations

    ######################################### 22 #########################################
    difficult_situations_6_name_file = path_to_data + 'difficult_situations_01_04_2022/monaco_2/data.csv'

    dir_difficult_situations_6_images = path_to_data + 'difficult_situations_01_04_2022/monaco_2/'
    list_images_difficult_situations_6 = glob.glob(dir_difficult_situations_6_images + '*')
    new_list_images_difficult_situations_6 = []
    for image in list_images_difficult_situations_6:
        if image != path_to_data + 'difficult_situations_01_04_2022/monaco_2/data.csv':
            new_list_images_difficult_situations_6.append(image)
    list_images_difficult_situations_6 = new_list_images_difficult_situations_6
    images_paths_difficult_situations_6 = sorted(list_images_difficult_situations_6,
                                                 key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_6 = pandas.read_csv(difficult_situations_6_name_file)
    array_annotations_difficult_situations_6 = parse_csv(array_annotations_difficult_situations_6)

    images_difficult_situations_6 = get_images(images_paths_difficult_situations_6, type_image, img_shape)
    images_difficult_situations_6, array_annotations_difficult_situations_6 = flip_images(images_difficult_situations_6,
                                                                                          array_annotations_difficult_situations_6)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_6:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_6 = normalized_annotations

    ######################################### 23 #########################################
    difficult_situations_7_name_file = path_to_data + 'difficult_situations_01_04_2022/monaco_3/data.csv'

    dir_difficult_situations_7_images = path_to_data + 'difficult_situations_01_04_2022/monaco_3/'
    list_images_difficult_situations_7 = glob.glob(dir_difficult_situations_7_images + '*')
    new_list_images_difficult_situations_7 = []
    for image in list_images_difficult_situations_7:
        if image != path_to_data + 'difficult_situations_01_04_2022/monaco_3/data.csv':
            new_list_images_difficult_situations_7.append(image)
    list_images_difficult_situations_7 = new_list_images_difficult_situations_7
    images_paths_difficult_situations_7 = sorted(list_images_difficult_situations_7,
                                                 key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_7 = pandas.read_csv(difficult_situations_7_name_file)
    array_annotations_difficult_situations_7 = parse_csv(array_annotations_difficult_situations_7)

    images_difficult_situations_7 = get_images(images_paths_difficult_situations_7, type_image, img_shape)
    images_difficult_situations_7, array_annotations_difficult_situations_7 = flip_images(images_difficult_situations_7,
                                                                                          array_annotations_difficult_situations_7)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_7:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_7 = normalized_annotations

    ######################################### 24 #########################################
    difficult_situations_8_name_file = path_to_data + 'difficult_situations_01_04_2022/monaco_4/data.csv'

    dir_difficult_situations_8_images = path_to_data + 'difficult_situations_01_04_2022/monaco_4/'
    list_images_difficult_situations_8 = glob.glob(dir_difficult_situations_8_images + '*')
    new_list_images_difficult_situations_8 = []
    for image in list_images_difficult_situations_8:
        if image != path_to_data + 'difficult_situations_01_04_2022/monaco_4/data.csv':
            new_list_images_difficult_situations_8.append(image)
    list_images_difficult_situations_8 = new_list_images_difficult_situations_8
    images_paths_difficult_situations_8 = sorted(list_images_difficult_situations_8,
                                                 key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_8 = pandas.read_csv(difficult_situations_8_name_file)
    array_annotations_difficult_situations_8 = parse_csv(array_annotations_difficult_situations_8)

    images_difficult_situations_8 = get_images(images_paths_difficult_situations_8, type_image, img_shape)
    images_difficult_situations_8, array_annotations_difficult_situations_8 = flip_images(images_difficult_situations_8,
                                                                                          array_annotations_difficult_situations_8)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_8:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_8 = normalized_annotations

    ######################################### 25 #########################################
    difficult_situations_9_name_file = path_to_data + 'difficult_situations_01_04_2022/monaco_5/data.csv'

    dir_difficult_situations_9_images = path_to_data + 'difficult_situations_01_04_2022/monaco_5/'
    list_images_difficult_situations_9 = glob.glob(dir_difficult_situations_9_images + '*')
    new_list_images_difficult_situations_9 = []
    for image in list_images_difficult_situations_9:
        if image != path_to_data + 'difficult_situations_01_04_2022/monaco_5/data.csv':
            new_list_images_difficult_situations_9.append(image)
    list_images_difficult_situations_9 = new_list_images_difficult_situations_9
    images_paths_difficult_situations_9 = sorted(list_images_difficult_situations_9,
                                                 key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_9 = pandas.read_csv(difficult_situations_9_name_file)
    array_annotations_difficult_situations_9 = parse_csv(array_annotations_difficult_situations_9)

    images_difficult_situations_9 = get_images(images_paths_difficult_situations_9, type_image, img_shape)
    images_difficult_situations_9, array_annotations_difficult_situations_9 = flip_images(images_difficult_situations_9,
                                                                                          array_annotations_difficult_situations_9)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_9:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_9 = normalized_annotations

    ######################################### 26 #########################################
    difficult_situations_10_name_file = path_to_data + 'difficult_situations_01_04_2022/monaco_6/data.csv'

    dir_difficult_situations_10_images = path_to_data + 'difficult_situations_01_04_2022/monaco_6/'
    list_images_difficult_situations_10 = glob.glob(dir_difficult_situations_10_images + '*')
    new_list_images_difficult_situations_10 = []
    for image in list_images_difficult_situations_10:
        if image != path_to_data + 'difficult_situations_01_04_2022/monaco_6/data.csv':
            new_list_images_difficult_situations_10.append(image)
    list_images_difficult_situations_10 = new_list_images_difficult_situations_10
    images_paths_difficult_situations_10 = sorted(list_images_difficult_situations_10,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_10 = pandas.read_csv(difficult_situations_10_name_file)
    array_annotations_difficult_situations_10 = parse_csv(array_annotations_difficult_situations_10)

    images_difficult_situations_10 = get_images(images_paths_difficult_situations_10, type_image, img_shape)
    images_difficult_situations_10, array_annotations_difficult_situations_10 = flip_images(
        images_difficult_situations_10, array_annotations_difficult_situations_10)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_10:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_10 = normalized_annotations

    ######################################### 27 #########################################
    difficult_situations_11_name_file = path_to_data + 'difficult_situations_01_04_2022/nurburgring_1/data.csv'

    dir_difficult_situations_11_images = path_to_data + 'difficult_situations_01_04_2022/nurburgring_1/'
    list_images_difficult_situations_11 = glob.glob(dir_difficult_situations_11_images + '*')
    new_list_images_difficult_situations_11 = []
    for image in list_images_difficult_situations_11:
        if image != path_to_data + 'difficult_situations_01_04_2022/nurburgring_1/data.csv':
            new_list_images_difficult_situations_11.append(image)
    list_images_difficult_situations_11 = new_list_images_difficult_situations_11
    images_paths_difficult_situations_11 = sorted(list_images_difficult_situations_11,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_11 = pandas.read_csv(difficult_situations_11_name_file)
    array_annotations_difficult_situations_11 = parse_csv(array_annotations_difficult_situations_11)

    images_difficult_situations_11 = get_images(images_paths_difficult_situations_11, type_image, img_shape)
    images_difficult_situations_11, array_annotations_difficult_situations_11 = flip_images(
        images_difficult_situations_11, array_annotations_difficult_situations_11)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_11:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_11 = normalized_annotations

    ######################################### 28 #########################################
    difficult_situations_12_name_file = path_to_data + 'difficult_situations_01_04_2022/nurburgring_2/data.csv'

    dir_difficult_situations_12_images = path_to_data + 'difficult_situations_01_04_2022/nurburgring_2/'
    list_images_difficult_situations_12 = glob.glob(dir_difficult_situations_12_images + '*')
    new_list_images_difficult_situations_12 = []
    for image in list_images_difficult_situations_12:
        if image != path_to_data + 'difficult_situations_01_04_2022/nurburgring_2/data.csv':
            new_list_images_difficult_situations_12.append(image)
    list_images_difficult_situations_12 = new_list_images_difficult_situations_12
    images_paths_difficult_situations_12 = sorted(list_images_difficult_situations_12,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_12 = pandas.read_csv(difficult_situations_12_name_file)
    array_annotations_difficult_situations_12 = parse_csv(array_annotations_difficult_situations_12)

    images_difficult_situations_12 = get_images(images_paths_difficult_situations_12, type_image, img_shape)
    images_difficult_situations_12, array_annotations_difficult_situations_12 = flip_images(
        images_difficult_situations_12, array_annotations_difficult_situations_12)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_12:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_12 = normalized_annotations

    ######################################### 29 #########################################
    difficult_situations_13_name_file = path_to_data + 'difficult_situations_01_04_2022/nurburgring_3/data.csv'

    dir_difficult_situations_13_images = path_to_data + 'difficult_situations_01_04_2022/nurburgring_3/'
    list_images_difficult_situations_13 = glob.glob(dir_difficult_situations_13_images + '*')
    new_list_images_difficult_situations_13 = []
    for image in list_images_difficult_situations_13:
        if image != path_to_data + 'difficult_situations_01_04_2022/nurburgring_3/data.csv':
            new_list_images_difficult_situations_13.append(image)
    list_images_difficult_situations_13 = new_list_images_difficult_situations_13
    images_paths_difficult_situations_13 = sorted(list_images_difficult_situations_13,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_13 = pandas.read_csv(difficult_situations_13_name_file)
    array_annotations_difficult_situations_13 = parse_csv(array_annotations_difficult_situations_13)

    images_difficult_situations_13 = get_images(images_paths_difficult_situations_13, type_image, img_shape)
    images_difficult_situations_13, array_annotations_difficult_situations_13 = flip_images(
        images_difficult_situations_13, array_annotations_difficult_situations_13)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_13:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_13 = normalized_annotations

    ######################################### 30 #########################################
    difficult_situations_14_name_file = path_to_data + 'difficult_situations_01_04_2022/nurburgring_4/data.csv'

    dir_difficult_situations_14_images = path_to_data + 'difficult_situations_01_04_2022/nurburgring_4/'
    list_images_difficult_situations_14 = glob.glob(dir_difficult_situations_14_images + '*')
    new_list_images_difficult_situations_14 = []
    for image in list_images_difficult_situations_14:
        if image != path_to_data + 'difficult_situations_01_04_2022/nurburgring_4/data.csv':
            new_list_images_difficult_situations_14.append(image)
    list_images_difficult_situations_14 = new_list_images_difficult_situations_14
    images_paths_difficult_situations_14 = sorted(list_images_difficult_situations_14,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_14 = pandas.read_csv(difficult_situations_14_name_file)
    array_annotations_difficult_situations_14 = parse_csv(array_annotations_difficult_situations_14)

    images_difficult_situations_14 = get_images(images_paths_difficult_situations_14, type_image, img_shape)
    images_difficult_situations_14, array_annotations_difficult_situations_14 = flip_images(
        images_difficult_situations_14, array_annotations_difficult_situations_14)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_14:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_14 = normalized_annotations

    ######################################### 31 #########################################
    difficult_situations_15_name_file = path_to_data + 'difficult_situations_01_04_2022_2/many_curves_1/data.csv'

    dir_difficult_situations_15_images = path_to_data + 'difficult_situations_01_04_2022_2/many_curves_1/'
    list_images_difficult_situations_15 = glob.glob(dir_difficult_situations_15_images + '*')
    new_list_images_difficult_situations_15 = []
    for image in list_images_difficult_situations_15:
        if image != path_to_data + 'difficult_situations_01_04_2022_2/many_curves_1/data.csv':
            new_list_images_difficult_situations_15.append(image)
    list_images_difficult_situations_15 = new_list_images_difficult_situations_15
    images_paths_difficult_situations_15 = sorted(list_images_difficult_situations_15,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_15 = pandas.read_csv(difficult_situations_15_name_file)
    array_annotations_difficult_situations_15 = parse_csv(array_annotations_difficult_situations_15)

    images_difficult_situations_15 = get_images(images_paths_difficult_situations_15, type_image, img_shape)
    images_difficult_situations_15, array_annotations_difficult_situations_15 = flip_images(
        images_difficult_situations_15, array_annotations_difficult_situations_15)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_15:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_15 = normalized_annotations

    ######################################### 32 #########################################
    difficult_situations_16_name_file = path_to_data + 'difficult_situations_01_04_2022_2/many_curves_2/data.csv'

    dir_difficult_situations_16_images = path_to_data + 'difficult_situations_01_04_2022_2/many_curves_2/'
    list_images_difficult_situations_16 = glob.glob(dir_difficult_situations_16_images + '*')
    new_list_images_difficult_situations_16 = []
    for image in list_images_difficult_situations_16:
        if image != path_to_data + 'difficult_situations_01_04_2022_2/many_curves_2/data.csv':
            new_list_images_difficult_situations_16.append(image)
    list_images_difficult_situations_16 = new_list_images_difficult_situations_16
    images_paths_difficult_situations_16 = sorted(list_images_difficult_situations_16,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_16 = pandas.read_csv(difficult_situations_16_name_file)
    array_annotations_difficult_situations_16 = parse_csv(array_annotations_difficult_situations_16)

    images_difficult_situations_16 = get_images(images_paths_difficult_situations_16, type_image, img_shape)
    images_difficult_situations_16, array_annotations_difficult_situations_16 = flip_images(
        images_difficult_situations_16, array_annotations_difficult_situations_16)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_16:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_16 = normalized_annotations

    ######################################### 33 #########################################
    difficult_situations_17_name_file = path_to_data + 'difficult_situations_01_04_2022_2/montreal_1/data.csv'

    dir_difficult_situations_17_images = path_to_data + 'difficult_situations_01_04_2022_2/montreal_1/'
    list_images_difficult_situations_17 = glob.glob(dir_difficult_situations_17_images + '*')
    new_list_images_difficult_situations_17 = []
    for image in list_images_difficult_situations_17:
        if image != path_to_data + 'difficult_situations_01_04_2022_2/montreal_1/data.csv':
            new_list_images_difficult_situations_17.append(image)
    list_images_difficult_situations_17 = new_list_images_difficult_situations_17
    images_paths_difficult_situations_17 = sorted(list_images_difficult_situations_17,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_17 = pandas.read_csv(difficult_situations_17_name_file)
    array_annotations_difficult_situations_17 = parse_csv(array_annotations_difficult_situations_17)

    images_difficult_situations_17 = get_images(images_paths_difficult_situations_17, type_image, img_shape)
    images_difficult_situations_17, array_annotations_difficult_situations_17 = flip_images(
        images_difficult_situations_17, array_annotations_difficult_situations_17)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_17:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_17 = normalized_annotations

    ######################################### 34 #########################################
    difficult_situations_18_name_file = path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_1/data.csv'

    dir_difficult_situations_18_images = path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_1/'
    list_images_difficult_situations_18 = glob.glob(dir_difficult_situations_18_images + '*')
    new_list_images_difficult_situations_18 = []
    for image in list_images_difficult_situations_18:
        if image != path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_1/data.csv':
            new_list_images_difficult_situations_18.append(image)
    list_images_difficult_situations_18 = new_list_images_difficult_situations_18
    images_paths_difficult_situations_18 = sorted(list_images_difficult_situations_18,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_18 = pandas.read_csv(difficult_situations_18_name_file)
    array_annotations_difficult_situations_18 = parse_csv(array_annotations_difficult_situations_18)

    images_difficult_situations_18 = get_images(images_paths_difficult_situations_18, type_image, img_shape)
    images_difficult_situations_18, array_annotations_difficult_situations_18 = flip_images(
        images_difficult_situations_18, array_annotations_difficult_situations_18)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_18:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_18 = normalized_annotations

    ######################################### 35 #########################################
    difficult_situations_19_name_file = path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_2/data.csv'

    dir_difficult_situations_19_images = path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_2/'
    list_images_difficult_situations_19 = glob.glob(dir_difficult_situations_19_images + '*')
    new_list_images_difficult_situations_19 = []
    for image in list_images_difficult_situations_19:
        if image != path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_2/data.csv':
            new_list_images_difficult_situations_19.append(image)
    list_images_difficult_situations_19 = new_list_images_difficult_situations_19
    images_paths_difficult_situations_19 = sorted(list_images_difficult_situations_19,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_19 = pandas.read_csv(difficult_situations_19_name_file)
    array_annotations_difficult_situations_19 = parse_csv(array_annotations_difficult_situations_19)

    images_difficult_situations_19 = get_images(images_paths_difficult_situations_19, type_image, img_shape)
    images_difficult_situations_19, array_annotations_difficult_situations_19 = flip_images(
        images_difficult_situations_19, array_annotations_difficult_situations_19)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_19:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_19 = normalized_annotations

    ######################################### 36 #########################################
    difficult_situations_20_name_file = path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_3/data.csv'

    dir_difficult_situations_20_images = path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_3/'
    list_images_difficult_situations_20 = glob.glob(dir_difficult_situations_20_images + '*')
    new_list_images_difficult_situations_20 = []
    for image in list_images_difficult_situations_20:
        if image != path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_3/data.csv':
            new_list_images_difficult_situations_20.append(image)
    list_images_difficult_situations_20 = new_list_images_difficult_situations_20
    images_paths_difficult_situations_20 = sorted(list_images_difficult_situations_20,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_20 = pandas.read_csv(difficult_situations_20_name_file)
    array_annotations_difficult_situations_20 = parse_csv(array_annotations_difficult_situations_20)

    images_difficult_situations_20 = get_images(images_paths_difficult_situations_20, type_image, img_shape)
    images_difficult_situations_20, array_annotations_difficult_situations_20 = flip_images(
        images_difficult_situations_20, array_annotations_difficult_situations_20)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_20:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_20 = normalized_annotations

    ######################################### 37 #########################################
    difficult_situations_21_name_file = path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_4/data.csv'

    dir_difficult_situations_21_images = path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_4/'
    list_images_difficult_situations_21 = glob.glob(dir_difficult_situations_21_images + '*')
    new_list_images_difficult_situations_21 = []
    for image in list_images_difficult_situations_21:
        if image != path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_4/data.csv':
            new_list_images_difficult_situations_21.append(image)
    list_images_difficult_situations_21 = new_list_images_difficult_situations_21
    images_paths_difficult_situations_21 = sorted(list_images_difficult_situations_21,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_21 = pandas.read_csv(difficult_situations_21_name_file)
    array_annotations_difficult_situations_21 = parse_csv(array_annotations_difficult_situations_21)

    images_difficult_situations_21 = get_images(images_paths_difficult_situations_21, type_image, img_shape)
    images_difficult_situations_21, array_annotations_difficult_situations_21 = flip_images(
        images_difficult_situations_21, array_annotations_difficult_situations_21)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_21:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_21 = normalized_annotations

    ######################################### 38 #########################################
    difficult_situations_22_name_file = path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_5/data.csv'

    dir_difficult_situations_22_images = path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_5/'
    list_images_difficult_situations_22 = glob.glob(dir_difficult_situations_22_images + '*')
    new_list_images_difficult_situations_22 = []
    for image in list_images_difficult_situations_22:
        if image != path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_5/data.csv':
            new_list_images_difficult_situations_22.append(image)
    list_images_difficult_situations_22 = new_list_images_difficult_situations_22
    images_paths_difficult_situations_22 = sorted(list_images_difficult_situations_22,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_22 = pandas.read_csv(difficult_situations_22_name_file)
    array_annotations_difficult_situations_22 = parse_csv(array_annotations_difficult_situations_22)

    images_difficult_situations_22 = get_images(images_paths_difficult_situations_22, type_image, img_shape)
    images_difficult_situations_22, array_annotations_difficult_situations_22 = flip_images(
        images_difficult_situations_22, array_annotations_difficult_situations_22)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_22:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_22 = normalized_annotations

    ######################################### 39 #########################################
    difficult_situations_23_name_file = path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_6/data.csv'

    dir_difficult_situations_23_images = path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_6/'
    list_images_difficult_situations_23 = glob.glob(dir_difficult_situations_23_images + '*')
    new_list_images_difficult_situations_23 = []
    for image in list_images_difficult_situations_23:
        if image != path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_6/data.csv':
            new_list_images_difficult_situations_23.append(image)
    list_images_difficult_situations_23 = new_list_images_difficult_situations_23
    images_paths_difficult_situations_23 = sorted(list_images_difficult_situations_23,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_23 = pandas.read_csv(difficult_situations_23_name_file)
    array_annotations_difficult_situations_23 = parse_csv(array_annotations_difficult_situations_23)

    images_difficult_situations_23 = get_images(images_paths_difficult_situations_23, type_image, img_shape)
    images_difficult_situations_23, array_annotations_difficult_situations_23 = flip_images(
        images_difficult_situations_23, array_annotations_difficult_situations_23)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_23:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_23 = normalized_annotations

    ######################################### 40 #########################################
    difficult_situations_24_name_file = path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_7/data.csv'

    dir_difficult_situations_24_images = path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_7/'
    list_images_difficult_situations_24 = glob.glob(dir_difficult_situations_24_images + '*')
    new_list_images_difficult_situations_24 = []
    for image in list_images_difficult_situations_24:
        if image != path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_7/data.csv':
            new_list_images_difficult_situations_24.append(image)
    list_images_difficult_situations_24 = new_list_images_difficult_situations_24
    images_paths_difficult_situations_24 = sorted(list_images_difficult_situations_24,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_24 = pandas.read_csv(difficult_situations_24_name_file)
    array_annotations_difficult_situations_24 = parse_csv(array_annotations_difficult_situations_24)

    images_difficult_situations_24 = get_images(images_paths_difficult_situations_24, type_image, img_shape)
    images_difficult_situations_24, array_annotations_difficult_situations_24 = flip_images(
        images_difficult_situations_24, array_annotations_difficult_situations_24)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_24:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_24 = normalized_annotations

    ######################################### 41 #########################################
    difficult_situations_25_name_file = path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_8/data.csv'

    dir_difficult_situations_25_images = path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_8/'
    list_images_difficult_situations_25 = glob.glob(dir_difficult_situations_25_images + '*')
    new_list_images_difficult_situations_25 = []
    for image in list_images_difficult_situations_25:
        if image != path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_8/data.csv':
            new_list_images_difficult_situations_25.append(image)
    list_images_difficult_situations_25 = new_list_images_difficult_situations_25
    images_paths_difficult_situations_25 = sorted(list_images_difficult_situations_25,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_25 = pandas.read_csv(difficult_situations_25_name_file)
    array_annotations_difficult_situations_25 = parse_csv(array_annotations_difficult_situations_25)

    images_difficult_situations_25 = get_images(images_paths_difficult_situations_25, type_image, img_shape)
    images_difficult_situations_25, array_annotations_difficult_situations_25 = flip_images(
        images_difficult_situations_25, array_annotations_difficult_situations_25)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_25:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_25 = normalized_annotations

    ######################################### 42 #########################################
    difficult_situations_26_name_file = path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_9/data.csv'

    dir_difficult_situations_26_images = path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_9/'
    list_images_difficult_situations_26 = glob.glob(dir_difficult_situations_26_images + '*')
    new_list_images_difficult_situations_26 = []
    for image in list_images_difficult_situations_26:
        if image != path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_9/data.csv':
            new_list_images_difficult_situations_26.append(image)
    list_images_difficult_situations_26 = new_list_images_difficult_situations_26
    images_paths_difficult_situations_26 = sorted(list_images_difficult_situations_26,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_26 = pandas.read_csv(difficult_situations_26_name_file)
    array_annotations_difficult_situations_26 = parse_csv(array_annotations_difficult_situations_26)

    images_difficult_situations_26 = get_images(images_paths_difficult_situations_26, type_image, img_shape)
    images_difficult_situations_26, array_annotations_difficult_situations_26 = flip_images(
        images_difficult_situations_26, array_annotations_difficult_situations_26)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_26:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_26 = normalized_annotations

    ######################################### 43 #########################################
    difficult_situations_27_name_file = path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_10/data.csv'

    dir_difficult_situations_27_images = path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_10/'
    list_images_difficult_situations_27 = glob.glob(dir_difficult_situations_27_images + '*')
    new_list_images_difficult_situations_27 = []
    for image in list_images_difficult_situations_27:
        if image != path_to_data + 'difficult_situations_01_04_2022_2/nurburgring_10/data.csv':
            new_list_images_difficult_situations_27.append(image)
    list_images_difficult_situations_27 = new_list_images_difficult_situations_27
    images_paths_difficult_situations_27 = sorted(list_images_difficult_situations_27,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_27 = pandas.read_csv(difficult_situations_27_name_file)
    array_annotations_difficult_situations_27 = parse_csv(array_annotations_difficult_situations_27)

    images_difficult_situations_27 = get_images(images_paths_difficult_situations_27, type_image, img_shape)
    images_difficult_situations_27, array_annotations_difficult_situations_27 = flip_images(
        images_difficult_situations_27, array_annotations_difficult_situations_27)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_difficult_situations_27:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations_difficult_situations_27 = normalized_annotations

    images_many_curves_1_1 = images_many_curves_1[:len(images_many_curves_1) // 2]
    images_many_curves_1_2 = images_many_curves_1[len(images_many_curves_1) // 2:]
    images_many_curves_1_1 = images_many_curves_1_1[:-15]
    images_many_curves_1_2 = images_many_curves_1_2[:-15]
    array_annotations_many_curves_1_1 = array_annotations_many_curves_1[:len(array_annotations_many_curves_1) // 2]
    array_annotations_many_curves_1_2 = array_annotations_many_curves_1[len(array_annotations_many_curves_1) // 2:]
    array_annotations_many_curves_1_1 = array_annotations_many_curves_1_1[:-15]
    array_annotations_many_curves_1_2 = array_annotations_many_curves_1_2[:-15]

    images_nurburgring_1_1 = images_nurburgring_1[:len(images_nurburgring_1) // 2]
    images_nurburgring_1_2 = images_nurburgring_1[len(images_nurburgring_1) // 2:]
    images_nurburgring_1_1 = images_nurburgring_1_1[:-8]
    images_nurburgring_1_2 = images_nurburgring_1_2[:-8]
    array_annotations_nurburgring_1_1 = array_annotations_nurburgring_1[:len(array_annotations_nurburgring_1) // 2]
    array_annotations_nurburgring_1_2 = array_annotations_nurburgring_1[len(array_annotations_nurburgring_1) // 2:]
    array_annotations_nurburgring_1_1 = array_annotations_nurburgring_1_1[:-8]
    array_annotations_nurburgring_1_2 = array_annotations_nurburgring_1_2[:-8]

    images_monaco_1_1 = images_monaco_1[:len(images_monaco_1) // 2]
    images_monaco_1_2 = images_monaco_1[len(images_monaco_1) // 2:]
    images_monaco_1_1 = images_monaco_1_1[:-3]
    images_monaco_1_2 = images_monaco_1_2[:-3]
    array_annotations_monaco_1_1 = array_annotations_monaco_1[:len(array_annotations_monaco_1) // 2]
    array_annotations_monaco_1_2 = array_annotations_monaco_1[len(array_annotations_monaco_1) // 2:]
    array_annotations_monaco_1_1 = array_annotations_monaco_1_1[:-3]
    array_annotations_monaco_1_2 = array_annotations_monaco_1_2[:-3]

    images_extended_simple_1_1 = images_extended_simple_1[:len(images_extended_simple_1) // 2]
    images_extended_simple_1_2 = images_extended_simple_1[len(images_extended_simple_1) // 2:]
    images_extended_simple_1_1 = images_extended_simple_1_1[:-40]
    images_extended_simple_1_2 = images_extended_simple_1_2[:-40]
    array_annotations_extended_simple_1_1 = array_annotations_extended_simple_1[
                                            :len(array_annotations_extended_simple_1) // 2]
    array_annotations_extended_simple_1_2 = array_annotations_extended_simple_1[
                                            len(array_annotations_extended_simple_1) // 2:]
    array_annotations_extended_simple_1_1 = array_annotations_extended_simple_1_1[:-40]
    array_annotations_extended_simple_1_2 = array_annotations_extended_simple_1_2[:-40]

    images_only_curves_1_1 = images_only_curves_1[:len(images_only_curves_1) // 2]
    images_only_curves_1_2 = images_only_curves_1[len(images_only_curves_1) // 2:]
    images_only_curves_1_1 = images_only_curves_1_1[:-15]
    images_only_curves_1_2 = images_only_curves_1_2[:-15]
    array_annotations_only_curves_1_1 = array_annotations_only_curves_1[:len(array_annotations_only_curves_1) // 2]
    array_annotations_only_curves_1_2 = array_annotations_only_curves_1[len(array_annotations_only_curves_1) // 2:]
    array_annotations_only_curves_1_1 = array_annotations_only_curves_1_1[:-15]
    array_annotations_only_curves_1_2 = array_annotations_only_curves_1_2[:-15]

    images_only_curves_2_1 = images_only_curves_2[:len(images_only_curves_2) // 2]
    images_only_curves_2_2 = images_only_curves_2[len(images_only_curves_2) // 2:]
    images_only_curves_2_1 = images_only_curves_2_1[:-23]
    images_only_curves_2_2 = images_only_curves_2_2[:-23]
    array_annotations_only_curves_2_1 = array_annotations_only_curves_2[:len(array_annotations_only_curves_2) // 2]
    array_annotations_only_curves_2_2 = array_annotations_only_curves_2[len(array_annotations_only_curves_2) // 2:]
    array_annotations_only_curves_2_1 = array_annotations_only_curves_2_1[:-23]
    array_annotations_only_curves_2_2 = array_annotations_only_curves_2_2[:-23]

    images_only_curves_3_1 = images_only_curves_3[:len(images_only_curves_3) // 2]
    images_only_curves_3_2 = images_only_curves_3[len(images_only_curves_3) // 2:]
    images_only_curves_3_1 = images_only_curves_3_1[:-43]
    images_only_curves_3_2 = images_only_curves_3_2[:-43]
    array_annotations_only_curves_3_1 = array_annotations_only_curves_3[:len(array_annotations_only_curves_3) // 2]
    array_annotations_only_curves_3_2 = array_annotations_only_curves_3[len(array_annotations_only_curves_3) // 2:]
    array_annotations_only_curves_3_1 = array_annotations_only_curves_3_1[:-43]
    array_annotations_only_curves_3_2 = array_annotations_only_curves_3_2[:-43]

    images_only_curves_4_1 = images_only_curves_4[:len(images_only_curves_4) // 2]
    images_only_curves_4_2 = images_only_curves_4[len(images_only_curves_4) // 2:]
    images_only_curves_4_1 = images_only_curves_4_1[:-27]
    images_only_curves_4_2 = images_only_curves_4_2[:-27]
    array_annotations_only_curves_4_1 = array_annotations_only_curves_4[:len(array_annotations_only_curves_4) // 2]
    array_annotations_only_curves_4_2 = array_annotations_only_curves_4[len(array_annotations_only_curves_4) // 2:]
    array_annotations_only_curves_4_1 = array_annotations_only_curves_4_1[:-27]
    array_annotations_only_curves_4_2 = array_annotations_only_curves_4_2[:-27]

    images_only_curves_5_1 = images_only_curves_5[:len(images_only_curves_5) // 2]
    images_only_curves_5_2 = images_only_curves_5[len(images_only_curves_5) // 2:]
    images_only_curves_5_1 = images_only_curves_5_1[:-49]
    images_only_curves_5_2 = images_only_curves_5_2[:-49]
    array_annotations_only_curves_5_1 = array_annotations_only_curves_5[:len(array_annotations_only_curves_5) // 2]
    array_annotations_only_curves_5_2 = array_annotations_only_curves_5[len(array_annotations_only_curves_5) // 2:]
    array_annotations_only_curves_5_1 = array_annotations_only_curves_5_1[:-49]
    array_annotations_only_curves_5_2 = array_annotations_only_curves_5_2[:-49]

    images_only_curves_6_1 = images_only_curves_6[:len(images_only_curves_6) // 2]
    images_only_curves_6_2 = images_only_curves_6[len(images_only_curves_6) // 2:]
    images_only_curves_6_1 = images_only_curves_6_1[:-39]
    images_only_curves_6_2 = images_only_curves_6_2[:-39]
    array_annotations_only_curves_6_1 = array_annotations_only_curves_6[:len(array_annotations_only_curves_6) // 2]
    array_annotations_only_curves_6_2 = array_annotations_only_curves_6[len(array_annotations_only_curves_6) // 2:]
    array_annotations_only_curves_6_1 = array_annotations_only_curves_6_1[:-39]
    array_annotations_only_curves_6_2 = array_annotations_only_curves_6_2[:-39]

    images_only_curves_7_1 = images_only_curves_7[:len(images_only_curves_7) // 2]
    images_only_curves_7_2 = images_only_curves_7[len(images_only_curves_7) // 2:]
    images_only_curves_7_1 = images_only_curves_7_1[:-30]
    images_only_curves_7_2 = images_only_curves_7_2[:-30]
    array_annotations_only_curves_7_1 = array_annotations_only_curves_7[:len(array_annotations_only_curves_7) // 2]
    array_annotations_only_curves_7_2 = array_annotations_only_curves_7[len(array_annotations_only_curves_7) // 2:]
    array_annotations_only_curves_7_1 = array_annotations_only_curves_7_1[:-30]
    array_annotations_only_curves_7_2 = array_annotations_only_curves_7_2[:-30]

    images_only_curves_8_1 = images_only_curves_8[:len(images_only_curves_8) // 2]
    images_only_curves_8_2 = images_only_curves_8[len(images_only_curves_8) // 2:]
    images_only_curves_8_1 = images_only_curves_8_1[:-16]
    images_only_curves_8_2 = images_only_curves_8_2[:-16]
    array_annotations_only_curves_8_1 = array_annotations_only_curves_8[:len(array_annotations_only_curves_8) // 2]
    array_annotations_only_curves_8_2 = array_annotations_only_curves_8[len(array_annotations_only_curves_8) // 2:]
    array_annotations_only_curves_8_1 = array_annotations_only_curves_8_1[:-16]
    array_annotations_only_curves_8_2 = array_annotations_only_curves_8_2[:-16]

    images_only_curves_9_1 = images_only_curves_9[:len(images_only_curves_9) // 2]
    images_only_curves_9_2 = images_only_curves_9[len(images_only_curves_9) // 2:]
    images_only_curves_9_1 = images_only_curves_9_1[:-20]
    images_only_curves_9_2 = images_only_curves_9_2[:-20]
    array_annotations_only_curves_9_1 = array_annotations_only_curves_9[:len(array_annotations_only_curves_9) // 2]
    array_annotations_only_curves_9_2 = array_annotations_only_curves_9[len(array_annotations_only_curves_9) // 2:]
    array_annotations_only_curves_9_1 = array_annotations_only_curves_9_1[:-20]
    array_annotations_only_curves_9_2 = array_annotations_only_curves_9_2[:-20]

    images_only_curves_10_1 = images_only_curves_10[:len(images_only_curves_10) // 2]
    images_only_curves_10_2 = images_only_curves_10[len(images_only_curves_10) // 2:]
    images_only_curves_10_1 = images_only_curves_10_1[:-32]
    images_only_curves_10_2 = images_only_curves_10_2[:-32]
    array_annotations_only_curves_10_1 = array_annotations_only_curves_10[:len(array_annotations_only_curves_10) // 2]
    array_annotations_only_curves_10_2 = array_annotations_only_curves_10[len(array_annotations_only_curves_10) // 2:]
    array_annotations_only_curves_10_1 = array_annotations_only_curves_10_1[:-32]
    array_annotations_only_curves_10_2 = array_annotations_only_curves_10_2[:-32]

    images_only_curves_11_1 = images_only_curves_11[:len(images_only_curves_11) // 2]
    images_only_curves_11_2 = images_only_curves_11[len(images_only_curves_11) // 2:]
    images_only_curves_11_1 = images_only_curves_11_1[:-16]
    images_only_curves_11_2 = images_only_curves_11_2[:-16]
    array_annotations_only_curves_11_1 = array_annotations_only_curves_11[:len(array_annotations_only_curves_11) // 2]
    array_annotations_only_curves_11_2 = array_annotations_only_curves_11[len(array_annotations_only_curves_11) // 2:]
    array_annotations_only_curves_11_1 = array_annotations_only_curves_11_1[:-16]
    array_annotations_only_curves_11_2 = array_annotations_only_curves_11_2[:-16]

    images_only_curves_12_1 = images_only_curves_12[:len(images_only_curves_12) // 2]
    images_only_curves_12_2 = images_only_curves_12[len(images_only_curves_12) // 2:]
    images_only_curves_12_1 = images_only_curves_12_1[:-48]
    images_only_curves_12_2 = images_only_curves_12_2[:-48]
    array_annotations_only_curves_12_1 = array_annotations_only_curves_12[:len(array_annotations_only_curves_12) // 2]
    array_annotations_only_curves_12_2 = array_annotations_only_curves_12[len(array_annotations_only_curves_12) // 2:]
    array_annotations_only_curves_12_1 = array_annotations_only_curves_12_1[:-48]
    array_annotations_only_curves_12_2 = array_annotations_only_curves_12_2[:-48]

    images_difficult_situations_1_1 = images_difficult_situations_1[:len(images_difficult_situations_1) // 2]
    images_difficult_situations_1_2 = images_difficult_situations_1[len(images_difficult_situations_1) // 2:]
    images_difficult_situations_1_1 = images_difficult_situations_1_1[:-42]
    images_difficult_situations_1_2 = images_difficult_situations_1_2[:-42]
    array_annotations_difficult_situations_1_1 = array_annotations_difficult_situations_1[
                                                 :len(array_annotations_difficult_situations_1) // 2]
    array_annotations_difficult_situations_1_2 = array_annotations_difficult_situations_1[
                                                 len(array_annotations_difficult_situations_1) // 2:]
    array_annotations_difficult_situations_1_1 = array_annotations_difficult_situations_1_1[:-42]
    array_annotations_difficult_situations_1_2 = array_annotations_difficult_situations_1_2[:-42]

    images_difficult_situations_2_1 = images_difficult_situations_2[:len(images_difficult_situations_2) // 2]
    images_difficult_situations_2_2 = images_difficult_situations_2[len(images_difficult_situations_2) // 2:]
    images_difficult_situations_2_1 = images_difficult_situations_2_1[:-42]
    images_difficult_situations_2_2 = images_difficult_situations_2_2[:-42]
    array_annotations_difficult_situations_2_1 = array_annotations_difficult_situations_2[
                                                 :len(array_annotations_difficult_situations_2) // 2]
    array_annotations_difficult_situations_2_2 = array_annotations_difficult_situations_2[
                                                 len(array_annotations_difficult_situations_2) // 2:]
    array_annotations_difficult_situations_2_1 = array_annotations_difficult_situations_2_1[:-42]
    array_annotations_difficult_situations_2_2 = array_annotations_difficult_situations_2_2[:-42]

    images_difficult_situations_3_1 = images_difficult_situations_3[:len(images_difficult_situations_3) // 2]
    images_difficult_situations_3_2 = images_difficult_situations_3[len(images_difficult_situations_3) // 2:]
    images_difficult_situations_3_1 = images_difficult_situations_3_1[:-23]
    images_difficult_situations_3_2 = images_difficult_situations_3_2[:-23]
    array_annotations_difficult_situations_3_1 = array_annotations_difficult_situations_3[
                                                 :len(array_annotations_difficult_situations_3) // 2]
    array_annotations_difficult_situations_3_2 = array_annotations_difficult_situations_3[
                                                 len(array_annotations_difficult_situations_3) // 2:]
    array_annotations_difficult_situations_3_1 = array_annotations_difficult_situations_3_1[:-23]
    array_annotations_difficult_situations_3_2 = array_annotations_difficult_situations_3_2[:-23]

    images_difficult_situations_4_1 = images_difficult_situations_4[:len(images_difficult_situations_4) // 2]
    images_difficult_situations_4_2 = images_difficult_situations_4[len(images_difficult_situations_4) // 2:]
    array_annotations_difficult_situations_4_1 = array_annotations_difficult_situations_4[
                                                 :len(array_annotations_difficult_situations_4) // 2]
    array_annotations_difficult_situations_4_2 = array_annotations_difficult_situations_4[
                                                 len(array_annotations_difficult_situations_4) // 2:]
    images_difficult_situations_5_1 = images_difficult_situations_5[:len(images_difficult_situations_5) // 2]
    images_difficult_situations_5_2 = images_difficult_situations_5[len(images_difficult_situations_5) // 2:]
    images_difficult_situations_5_1 = images_difficult_situations_5_1[:-39]
    images_difficult_situations_5_2 = images_difficult_situations_5_2[:-39]
    array_annotations_difficult_situations_5_1 = array_annotations_difficult_situations_5[
                                                 :len(array_annotations_difficult_situations_5) // 2]
    array_annotations_difficult_situations_5_2 = array_annotations_difficult_situations_5[
                                                 len(array_annotations_difficult_situations_5) // 2:]
    array_annotations_difficult_situations_5_1 = array_annotations_difficult_situations_5_1[:-39]
    array_annotations_difficult_situations_5_2 = array_annotations_difficult_situations_5_2[:-39]

    images_difficult_situations_6_1 = images_difficult_situations_6[:len(images_difficult_situations_6) // 2]
    images_difficult_situations_6_2 = images_difficult_situations_6[len(images_difficult_situations_6) // 2:]
    images_difficult_situations_6_1 = images_difficult_situations_6_1[:-33]
    images_difficult_situations_6_2 = images_difficult_situations_6_2[:-33]
    array_annotations_difficult_situations_6_1 = array_annotations_difficult_situations_6[
                                                 :len(array_annotations_difficult_situations_6) // 2]
    array_annotations_difficult_situations_6_2 = array_annotations_difficult_situations_6[
                                                 len(array_annotations_difficult_situations_6) // 2:]
    array_annotations_difficult_situations_6_1 = array_annotations_difficult_situations_6_1[:-33]
    array_annotations_difficult_situations_6_2 = array_annotations_difficult_situations_6_2[:-33]

    images_difficult_situations_7_1 = images_difficult_situations_7[:len(images_difficult_situations_7) // 2]
    images_difficult_situations_7_2 = images_difficult_situations_7[len(images_difficult_situations_7) // 2:]
    images_difficult_situations_7_1 = images_difficult_situations_7_1[:-5]
    images_difficult_situations_7_2 = images_difficult_situations_7_2[:-5]
    array_annotations_difficult_situations_7_1 = array_annotations_difficult_situations_7[
                                                 :len(array_annotations_difficult_situations_7) // 2]
    array_annotations_difficult_situations_7_2 = array_annotations_difficult_situations_7[
                                                 len(array_annotations_difficult_situations_7) // 2:]
    array_annotations_difficult_situations_7_1 = array_annotations_difficult_situations_7_1[:-5]
    array_annotations_difficult_situations_7_2 = array_annotations_difficult_situations_7_2[:-5]

    images_difficult_situations_8_1 = images_difficult_situations_8[:len(images_difficult_situations_8) // 2]
    images_difficult_situations_8_2 = images_difficult_situations_8[len(images_difficult_situations_8) // 2:]
    images_difficult_situations_8_1 = images_difficult_situations_8_1[:-32]
    images_difficult_situations_8_2 = images_difficult_situations_8_2[:-32]
    array_annotations_difficult_situations_8_1 = array_annotations_difficult_situations_8[
                                                 :len(array_annotations_difficult_situations_8) // 2]
    array_annotations_difficult_situations_8_2 = array_annotations_difficult_situations_8[
                                                 len(array_annotations_difficult_situations_8) // 2:]
    array_annotations_difficult_situations_8_1 = array_annotations_difficult_situations_8_1[:-32]
    array_annotations_difficult_situations_8_2 = array_annotations_difficult_situations_8_2[:-32]

    images_difficult_situations_9_1 = images_difficult_situations_9[:len(images_difficult_situations_9) // 2]
    images_difficult_situations_9_2 = images_difficult_situations_9[len(images_difficult_situations_9) // 2:]
    images_difficult_situations_9_1 = images_difficult_situations_9_1[:-26]
    images_difficult_situations_9_2 = images_difficult_situations_9_2[:-26]
    array_annotations_difficult_situations_9_1 = array_annotations_difficult_situations_9[
                                                 :len(array_annotations_difficult_situations_9) // 2]
    array_annotations_difficult_situations_9_2 = array_annotations_difficult_situations_9[
                                                 len(array_annotations_difficult_situations_9) // 2:]
    array_annotations_difficult_situations_9_1 = array_annotations_difficult_situations_9_1[:-26]
    array_annotations_difficult_situations_9_2 = array_annotations_difficult_situations_9_2[:-26]

    images_difficult_situations_10_1 = images_difficult_situations_10[:len(images_difficult_situations_10) // 2]
    images_difficult_situations_10_2 = images_difficult_situations_10[len(images_difficult_situations_10) // 2:]
    images_difficult_situations_10_1 = images_difficult_situations_10_1[:-42]
    images_difficult_situations_10_2 = images_difficult_situations_10_2[:-42]
    array_annotations_difficult_situations_10_1 = array_annotations_difficult_situations_10[
                                                  :len(array_annotations_difficult_situations_10) // 2]
    array_annotations_difficult_situations_10_2 = array_annotations_difficult_situations_10[
                                                  len(array_annotations_difficult_situations_10) // 2:]
    array_annotations_difficult_situations_10_1 = array_annotations_difficult_situations_10_1[:-42]
    array_annotations_difficult_situations_10_2 = array_annotations_difficult_situations_10_2[:-42]

    images_difficult_situations_11_1 = images_difficult_situations_11[:len(images_difficult_situations_11) // 2]
    images_difficult_situations_11_2 = images_difficult_situations_11[len(images_difficult_situations_11) // 2:]
    images_difficult_situations_11_1 = images_difficult_situations_11_1[:-13]
    images_difficult_situations_11_2 = images_difficult_situations_11_2[:-13]
    array_annotations_difficult_situations_11_1 = array_annotations_difficult_situations_11[
                                                  :len(array_annotations_difficult_situations_11) // 2]
    array_annotations_difficult_situations_11_2 = array_annotations_difficult_situations_11[
                                                  len(array_annotations_difficult_situations_11) // 2:]
    array_annotations_difficult_situations_11_1 = array_annotations_difficult_situations_11_1[:-13]
    array_annotations_difficult_situations_11_2 = array_annotations_difficult_situations_11_2[:-13]

    images_difficult_situations_12_1 = images_difficult_situations_12[:len(images_difficult_situations_12) // 2]
    images_difficult_situations_12_2 = images_difficult_situations_12[len(images_difficult_situations_12) // 2:]
    images_difficult_situations_12_1 = images_difficult_situations_12_1[:-3]
    images_difficult_situations_12_2 = images_difficult_situations_12_2[:-3]
    array_annotations_difficult_situations_12_1 = array_annotations_difficult_situations_12[
                                                  :len(array_annotations_difficult_situations_12) // 2]
    array_annotations_difficult_situations_12_2 = array_annotations_difficult_situations_12[
                                                  len(array_annotations_difficult_situations_12) // 2:]
    array_annotations_difficult_situations_12_1 = array_annotations_difficult_situations_12_1[:-3]
    array_annotations_difficult_situations_12_2 = array_annotations_difficult_situations_12_2[:-3]

    images_difficult_situations_13_1 = images_difficult_situations_13[:len(images_difficult_situations_13) // 2]
    images_difficult_situations_13_2 = images_difficult_situations_13[len(images_difficult_situations_13) // 2:]
    images_difficult_situations_13_1 = images_difficult_situations_13_1[:-29]
    images_difficult_situations_13_2 = images_difficult_situations_13_2[:-29]
    array_annotations_difficult_situations_13_1 = array_annotations_difficult_situations_13[
                                                  :len(array_annotations_difficult_situations_13) // 2]
    array_annotations_difficult_situations_13_2 = array_annotations_difficult_situations_13[
                                                  len(array_annotations_difficult_situations_13) // 2:]
    array_annotations_difficult_situations_13_1 = array_annotations_difficult_situations_13_1[:-29]
    array_annotations_difficult_situations_13_2 = array_annotations_difficult_situations_13_2[:-29]

    images_difficult_situations_14_1 = images_difficult_situations_14[:len(images_difficult_situations_14) // 2]
    images_difficult_situations_14_2 = images_difficult_situations_14[len(images_difficult_situations_14) // 2:]
    images_difficult_situations_14_1 = images_difficult_situations_14_1[:-2]
    images_difficult_situations_14_2 = images_difficult_situations_14_2[:-2]
    array_annotations_difficult_situations_14_1 = array_annotations_difficult_situations_14[
                                                  :len(array_annotations_difficult_situations_14) // 2]
    array_annotations_difficult_situations_14_2 = array_annotations_difficult_situations_14[
                                                  len(array_annotations_difficult_situations_14) // 2:]
    array_annotations_difficult_situations_14_1 = array_annotations_difficult_situations_14_1[:-2]
    array_annotations_difficult_situations_14_2 = array_annotations_difficult_situations_14_2[:-2]

    images_difficult_situations_15_1 = images_difficult_situations_15[:len(images_difficult_situations_15) // 2]
    images_difficult_situations_15_2 = images_difficult_situations_15[len(images_difficult_situations_15) // 2:]
    images_difficult_situations_15_1 = images_difficult_situations_15_1[:-11]
    images_difficult_situations_15_2 = images_difficult_situations_15_2[:-11]
    array_annotations_difficult_situations_15_1 = array_annotations_difficult_situations_15[
                                                  :len(array_annotations_difficult_situations_15) // 2]
    array_annotations_difficult_situations_15_2 = array_annotations_difficult_situations_15[
                                                  len(array_annotations_difficult_situations_15) // 2:]
    array_annotations_difficult_situations_15_1 = array_annotations_difficult_situations_15_1[:-11]
    array_annotations_difficult_situations_15_2 = array_annotations_difficult_situations_15_2[:-11]

    images_difficult_situations_16_1 = images_difficult_situations_16[:len(images_difficult_situations_16) // 2]
    images_difficult_situations_16_2 = images_difficult_situations_16[len(images_difficult_situations_16) // 2:]
    images_difficult_situations_16_1 = images_difficult_situations_16_1[:-26]
    images_difficult_situations_16_2 = images_difficult_situations_16_2[:-26]
    array_annotations_difficult_situations_16_1 = array_annotations_difficult_situations_16[
                                                  :len(array_annotations_difficult_situations_16) // 2]
    array_annotations_difficult_situations_16_2 = array_annotations_difficult_situations_16[
                                                  len(array_annotations_difficult_situations_16) // 2:]
    array_annotations_difficult_situations_16_1 = array_annotations_difficult_situations_16_1[:-26]
    array_annotations_difficult_situations_16_2 = array_annotations_difficult_situations_16_2[:-26]

    images_difficult_situations_17_1 = images_difficult_situations_17[:len(images_difficult_situations_17) // 2]
    images_difficult_situations_17_2 = images_difficult_situations_17[len(images_difficult_situations_17) // 2:]
    images_difficult_situations_17_1 = images_difficult_situations_17_1[:-18]
    images_difficult_situations_17_2 = images_difficult_situations_17_2[:-18]
    array_annotations_difficult_situations_17_1 = array_annotations_difficult_situations_17[
                                                  :len(array_annotations_difficult_situations_17) // 2]
    array_annotations_difficult_situations_17_2 = array_annotations_difficult_situations_17[
                                                  len(array_annotations_difficult_situations_17) // 2:]
    array_annotations_difficult_situations_17_1 = array_annotations_difficult_situations_17_1[:-18]
    array_annotations_difficult_situations_17_2 = array_annotations_difficult_situations_17_2[:-18]

    images_difficult_situations_18_1 = images_difficult_situations_18[:len(images_difficult_situations_18) // 2]
    images_difficult_situations_18_2 = images_difficult_situations_18[len(images_difficult_situations_18) // 2:]
    images_difficult_situations_18_1 = images_difficult_situations_18_1[:-3]
    images_difficult_situations_18_2 = images_difficult_situations_18_2[:-3]
    array_annotations_difficult_situations_18_1 = array_annotations_difficult_situations_18[
                                                  :len(array_annotations_difficult_situations_18) // 2]
    array_annotations_difficult_situations_18_2 = array_annotations_difficult_situations_18[
                                                  len(array_annotations_difficult_situations_18) // 2:]
    array_annotations_difficult_situations_18_1 = array_annotations_difficult_situations_18_1[:-3]
    array_annotations_difficult_situations_18_2 = array_annotations_difficult_situations_18_2[:-3]

    images_difficult_situations_19_1 = images_difficult_situations_19[:len(images_difficult_situations_19) // 2]
    images_difficult_situations_19_2 = images_difficult_situations_19[len(images_difficult_situations_19) // 2:]
    array_annotations_difficult_situations_19_1 = array_annotations_difficult_situations_19[
                                                  :len(array_annotations_difficult_situations_19) // 2]
    array_annotations_difficult_situations_19_2 = array_annotations_difficult_situations_19[
                                                  len(array_annotations_difficult_situations_19) // 2:]

    images_difficult_situations_20_1 = images_difficult_situations_20[:len(images_difficult_situations_20) // 2]
    images_difficult_situations_20_2 = images_difficult_situations_20[len(images_difficult_situations_20) // 2:]
    array_annotations_difficult_situations_20_1 = array_annotations_difficult_situations_20[
                                                  :len(array_annotations_difficult_situations_20) // 2]
    array_annotations_difficult_situations_20_2 = array_annotations_difficult_situations_20[
                                                  len(array_annotations_difficult_situations_20) // 2:]

    images_difficult_situations_21_1 = images_difficult_situations_21[:len(images_difficult_situations_21) // 2]
    images_difficult_situations_21_2 = images_difficult_situations_21[len(images_difficult_situations_21) // 2:]
    images_difficult_situations_21_1 = images_difficult_situations_21_1[:-26]
    images_difficult_situations_21_2 = images_difficult_situations_21_2[:-26]
    array_annotations_difficult_situations_21_1 = array_annotations_difficult_situations_21[
                                                  :len(array_annotations_difficult_situations_21) // 2]
    array_annotations_difficult_situations_21_2 = array_annotations_difficult_situations_21[
                                                  len(array_annotations_difficult_situations_21) // 2:]
    array_annotations_difficult_situations_21_1 = array_annotations_difficult_situations_21_1[:-26]
    array_annotations_difficult_situations_21_2 = array_annotations_difficult_situations_21_2[:-26]

    images_difficult_situations_22_1 = images_difficult_situations_22[:len(images_difficult_situations_22) // 2]
    images_difficult_situations_22_2 = images_difficult_situations_22[len(images_difficult_situations_22) // 2:]
    images_difficult_situations_22_1 = images_difficult_situations_22_1[:-4]
    images_difficult_situations_22_2 = images_difficult_situations_22_2[:-4]
    array_annotations_difficult_situations_22_1 = array_annotations_difficult_situations_22[
                                                  :len(array_annotations_difficult_situations_22) // 2]
    array_annotations_difficult_situations_22_2 = array_annotations_difficult_situations_22[
                                                  len(array_annotations_difficult_situations_22) // 2:]
    array_annotations_difficult_situations_22_1 = array_annotations_difficult_situations_22_1[:-4]
    array_annotations_difficult_situations_22_2 = array_annotations_difficult_situations_22_2[:-4]

    images_difficult_situations_23_1 = images_difficult_situations_23[:len(images_difficult_situations_23) // 2]
    images_difficult_situations_23_2 = images_difficult_situations_23[len(images_difficult_situations_23) // 2:]
    images_difficult_situations_23_1 = images_difficult_situations_23_1[:-35]
    images_difficult_situations_23_2 = images_difficult_situations_23_2[:-35]
    array_annotations_difficult_situations_23_1 = array_annotations_difficult_situations_23[
                                                  :len(array_annotations_difficult_situations_23) // 2]
    array_annotations_difficult_situations_23_2 = array_annotations_difficult_situations_23[
                                                  len(array_annotations_difficult_situations_23) // 2:]
    array_annotations_difficult_situations_23_1 = array_annotations_difficult_situations_23_1[:-35]
    array_annotations_difficult_situations_23_2 = array_annotations_difficult_situations_23_2[:-35]

    images_difficult_situations_24_1 = images_difficult_situations_24[:len(images_difficult_situations_24) // 2]
    images_difficult_situations_24_2 = images_difficult_situations_24[len(images_difficult_situations_24) // 2:]
    images_difficult_situations_24_1 = images_difficult_situations_24_1[:-5]
    images_difficult_situations_24_2 = images_difficult_situations_24_2[:-5]
    array_annotations_difficult_situations_24_1 = array_annotations_difficult_situations_24[
                                                  :len(array_annotations_difficult_situations_24) // 2]
    array_annotations_difficult_situations_24_2 = array_annotations_difficult_situations_24[
                                                  len(array_annotations_difficult_situations_24) // 2:]
    array_annotations_difficult_situations_24_1 = array_annotations_difficult_situations_24_1[:-5]
    array_annotations_difficult_situations_24_2 = array_annotations_difficult_situations_24_2[:-5]

    images_difficult_situations_25_1 = images_difficult_situations_25[:len(images_difficult_situations_25) // 2]
    images_difficult_situations_25_2 = images_difficult_situations_25[len(images_difficult_situations_25) // 2:]
    images_difficult_situations_25_1 = images_difficult_situations_25_1[:-15]
    images_difficult_situations_25_2 = images_difficult_situations_25_2[:-15]
    array_annotations_difficult_situations_25_1 = array_annotations_difficult_situations_25[
                                                  :len(array_annotations_difficult_situations_25) // 2]
    array_annotations_difficult_situations_25_2 = array_annotations_difficult_situations_25[
                                                  len(array_annotations_difficult_situations_25) // 2:]
    array_annotations_difficult_situations_25_1 = array_annotations_difficult_situations_25_1[:-15]
    array_annotations_difficult_situations_25_2 = array_annotations_difficult_situations_25_2[:-15]

    images_difficult_situations_26_1 = images_difficult_situations_26[:len(images_difficult_situations_26) // 2]
    images_difficult_situations_26_2 = images_difficult_situations_26[len(images_difficult_situations_26) // 2:]
    images_difficult_situations_26_1 = images_difficult_situations_26_1[:-31]
    images_difficult_situations_26_2 = images_difficult_situations_26_2[:-31]
    array_annotations_difficult_situations_26_1 = array_annotations_difficult_situations_26[
                                                  :len(array_annotations_difficult_situations_26) // 2]
    array_annotations_difficult_situations_26_2 = array_annotations_difficult_situations_26[
                                                  len(array_annotations_difficult_situations_26) // 2:]
    array_annotations_difficult_situations_26_1 = array_annotations_difficult_situations_26_1[:-31]
    array_annotations_difficult_situations_26_2 = array_annotations_difficult_situations_26_2[:-31]

    images_difficult_situations_27_1 = images_difficult_situations_27[:len(images_difficult_situations_27) // 2]
    images_difficult_situations_27_2 = images_difficult_situations_27[len(images_difficult_situations_27) // 2:]
    images_difficult_situations_27_1 = images_difficult_situations_27_1[:-31]
    images_difficult_situations_27_2 = images_difficult_situations_27_2[:-31]
    array_annotations_difficult_situations_27_1 = array_annotations_difficult_situations_27[
                                                  :len(array_annotations_difficult_situations_27) // 2]
    array_annotations_difficult_situations_27_2 = array_annotations_difficult_situations_27[
                                                  len(array_annotations_difficult_situations_27) // 2:]
    array_annotations_difficult_situations_27_1 = array_annotations_difficult_situations_27_1[:-31]
    array_annotations_difficult_situations_27_2 = array_annotations_difficult_situations_27_2[:-31]

    array_x = [
        images_many_curves_1_1, images_many_curves_1_2, images_nurburgring_1_1, images_nurburgring_1_2,
        images_monaco_1_1, images_monaco_1_2,
        images_extended_simple_1_1, images_extended_simple_1_2, images_only_curves_1_1, images_only_curves_1_2,
        images_only_curves_2_1, images_only_curves_2_2,
        images_only_curves_3_1, images_only_curves_3_2, images_only_curves_4_1, images_only_curves_4_2,
        images_only_curves_5_1, images_only_curves_5_2,
        images_only_curves_6_1, images_only_curves_6_2, images_only_curves_7_1, images_only_curves_7_2,
        images_only_curves_8_1, images_only_curves_8_2,
        images_only_curves_9_1, images_only_curves_9_2, images_only_curves_10_1, images_only_curves_10_2,
        images_only_curves_11_1, images_only_curves_11_2,
        images_only_curves_12_1, images_only_curves_12_2, images_difficult_situations_1_1,
        images_difficult_situations_1_2, images_difficult_situations_2_1, images_difficult_situations_2_2,
        images_difficult_situations_3_1, images_difficult_situations_3_2, images_difficult_situations_4_1,
        images_difficult_situations_4_2,
        images_difficult_situations_5_1, images_difficult_situations_5_2, images_difficult_situations_6_1,
        images_difficult_situations_6_2,
        images_difficult_situations_7_1, images_difficult_situations_7_2, images_difficult_situations_8_1,
        images_difficult_situations_8_2,
        images_difficult_situations_9_1, images_difficult_situations_9_2,
        images_difficult_situations_10_1, images_difficult_situations_10_2, images_difficult_situations_11_1,
        images_difficult_situations_11_2,
        images_difficult_situations_12_1, images_difficult_situations_12_2, images_difficult_situations_13_1,
        images_difficult_situations_13_2,
        images_difficult_situations_14_1, images_difficult_situations_14_2, images_difficult_situations_15_1,
        images_difficult_situations_15_2,
        images_difficult_situations_16_1, images_difficult_situations_16_2, images_difficult_situations_17_1,
        images_difficult_situations_17_2,
        images_difficult_situations_18_1, images_difficult_situations_18_2, images_difficult_situations_19_1,
        images_difficult_situations_19_2,
        images_difficult_situations_20_1, images_difficult_situations_20_2, images_difficult_situations_21_1,
        images_difficult_situations_21_2,
        images_difficult_situations_22_1, images_difficult_situations_22_2, images_difficult_situations_23_1,
        images_difficult_situations_23_2,
        images_difficult_situations_24_1, images_difficult_situations_24_2, images_difficult_situations_25_1,
        images_difficult_situations_25_2,
        images_difficult_situations_26_1, images_difficult_situations_26_2, images_difficult_situations_27_1,
        images_difficult_situations_27_2
    ]

    array_y = [
        array_annotations_many_curves_1_1, array_annotations_many_curves_1_2, array_annotations_nurburgring_1_1,
        array_annotations_nurburgring_1_2, array_annotations_monaco_1_1, array_annotations_monaco_1_2,
        array_annotations_extended_simple_1_1, array_annotations_extended_simple_1_2, array_annotations_only_curves_1_1,
        array_annotations_only_curves_1_2, array_annotations_only_curves_2_1, array_annotations_only_curves_2_2,
        array_annotations_only_curves_3_1, array_annotations_only_curves_3_2, array_annotations_only_curves_4_1,
        array_annotations_only_curves_4_2, array_annotations_only_curves_5_1, array_annotations_only_curves_5_2,
        array_annotations_only_curves_6_1, array_annotations_only_curves_6_2, array_annotations_only_curves_7_1,
        array_annotations_only_curves_7_2, array_annotations_only_curves_8_1, array_annotations_only_curves_8_2,
        array_annotations_only_curves_9_1, array_annotations_only_curves_9_2, array_annotations_only_curves_10_1,
        array_annotations_only_curves_10_2, array_annotations_only_curves_11_1, array_annotations_only_curves_11_2,
        array_annotations_only_curves_12_1, array_annotations_only_curves_12_2,
        array_annotations_difficult_situations_1_1, array_annotations_difficult_situations_1_2,
        array_annotations_difficult_situations_2_1, array_annotations_difficult_situations_2_2,
        array_annotations_difficult_situations_3_1, array_annotations_difficult_situations_3_2,
        array_annotations_difficult_situations_4_1, array_annotations_difficult_situations_4_2,
        array_annotations_difficult_situations_5_1, array_annotations_difficult_situations_5_2,
        array_annotations_difficult_situations_6_1, array_annotations_difficult_situations_6_2,
        array_annotations_difficult_situations_7_1, array_annotations_difficult_situations_7_2,
        array_annotations_difficult_situations_8_1, array_annotations_difficult_situations_8_2,
        array_annotations_difficult_situations_9_1, array_annotations_difficult_situations_9_2,
        array_annotations_difficult_situations_10_1, array_annotations_difficult_situations_10_2,
        array_annotations_difficult_situations_11_1, array_annotations_difficult_situations_11_2,
        array_annotations_difficult_situations_12_1, array_annotations_difficult_situations_12_2,
        array_annotations_difficult_situations_13_1, array_annotations_difficult_situations_13_2,
        array_annotations_difficult_situations_14_1, array_annotations_difficult_situations_14_2,
        array_annotations_difficult_situations_15_1, array_annotations_difficult_situations_15_2,
        array_annotations_difficult_situations_16_1, array_annotations_difficult_situations_16_2,
        array_annotations_difficult_situations_17_1, array_annotations_difficult_situations_17_2,
        array_annotations_difficult_situations_18_1, array_annotations_difficult_situations_18_2,
        array_annotations_difficult_situations_19_1, array_annotations_difficult_situations_19_2,
        array_annotations_difficult_situations_20_1, array_annotations_difficult_situations_20_2,
        array_annotations_difficult_situations_21_1, array_annotations_difficult_situations_21_2,
        array_annotations_difficult_situations_22_1, array_annotations_difficult_situations_22_2,
        array_annotations_difficult_situations_23_1, array_annotations_difficult_situations_23_2,
        array_annotations_difficult_situations_24_1, array_annotations_difficult_situations_24_2,
        array_annotations_difficult_situations_25_1, array_annotations_difficult_situations_25_2,
        array_annotations_difficult_situations_26_1, array_annotations_difficult_situations_26_2,
        array_annotations_difficult_situations_27_1, array_annotations_difficult_situations_27_2,
    ]
    print(len(array_x))
    print(len(array_y))

    return array_x, array_y


def separate_dataset_into_sequences(array_x, array_y):
    new_array_x = []
    new_array_y = []

    for x, images_array in enumerate(array_x):
        mini_array_x = []
        mini_array_y = []
        for y, image in enumerate(images_array):
            if y + 9 < len(images_array):
                image_3d = np.array([array_x[x][y], array_x[x][y + 4], array_x[x][y + 9]])
                mini_array_x.append(image_3d)
                mini_array_y.append(array_y[x][y + 9])
        new_array_x.append(mini_array_x)
        new_array_y.append(mini_array_y)

    return new_array_x, new_array_y


def add_extreme_sequences(array_x, array_y):
    '''
    Look for extreme 50 frames sequences inside every big-sequence
    '''
    new_array_x_extreme = []
    new_array_y_extreme = []
    for x, big_imgs in enumerate(array_x):
        new_big_imgs = []
        new_big_anns = []
        for y, big_img in enumerate(big_imgs):
            big_ann = array_y[x][y]
            new_big_imgs.append(big_img)
            new_big_anns.append(big_ann)

            if big_ann[1] >= 0.55 or big_ann[1] <= 0.45:
                if big_ann[1] >= 0.8 or big_ann[1] <= 0.2:
                    for i in range(0, 30):
                        new_big_imgs.append(big_img)
                        new_big_anns.append(big_ann)
                elif big_ann[1] >= 0.75 or big_ann[1] <= 0.25:
                    for i in range(0, 15):
                        new_big_imgs.append(big_img)
                        new_big_anns.append(big_ann)
                elif big_ann[1] >= 0.6 or big_ann[1] <= 0.4:
                    for i in range(0, 10):
                        new_big_imgs.append(big_img)
                        new_big_anns.append(big_ann)
                else:
                    for i in range(0, 5):
                        new_big_imgs.append(big_img)
                        new_big_anns.append(big_ann)
        new_array_x_extreme.append(new_big_imgs)
        new_array_y_extreme.append(new_big_anns)

    new_array_x = new_array_x_extreme
    new_array_y = new_array_y_extreme

    shown_array_imgs = []
    shown_array_annotations = []
    random_sort = random.sample(range(0, 86), 86)

    for numb in random_sort:
        shown_array_imgs += new_array_x[numb]
        shown_array_annotations += new_array_y[numb]

    print(len(shown_array_imgs))
    print(len(shown_array_annotations))

    array_x = shown_array_imgs
    array_y = shown_array_annotations
    return array_x, array_y


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
    array_imgs, array_annotations = get_images_and_annotations(path_to_data, type_image, img_shape)
    array_x, array_y = separate_dataset_into_sequences(array_imgs, array_annotations)
    if data_type == 'extreme':
        array_x, array_y = add_extreme_sequences(array_x, array_y)
    images_train, annotations_train, images_validation, annotations_validation = separate_dataset_into_train_validation(
        array_x, array_y)

    return images_train, annotations_train, images_validation, annotations_validation
