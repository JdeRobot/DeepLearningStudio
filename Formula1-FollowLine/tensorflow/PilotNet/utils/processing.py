import glob
import cv2
import pandas

import numpy as np

from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split


def get_images(list_images, type_image, image_shape):
    image_shape = (image_shape[0], image_shape[1])
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


def add_extreme_data(images, array_annotations):
    for i in range(0, len(array_annotations)):
        if abs(array_annotations[i][1]) >= 0.3:
            if abs(array_annotations[i][1]) >= 4:
                num_iter = 50
            elif abs(array_annotations[i][1]) >= 3:
                num_iter = 35
            elif abs(array_annotations[i][1]) >= 2:
                num_iter = 25
            elif abs(array_annotations[i][1]) >= 1:
                num_iter = 15
            elif abs(array_annotations[i][1]) >= 0.5:
                num_iter = 5
            else:
                num_iter = 2
            for j in range(0, num_iter):
                array_annotations.append(array_annotations[i])
                images.append(images[i])

    return images, array_annotations


def get_images_and_annotations(path_to_data, type_image, img_shape, data_type):
    ######################################### 1 #########################################
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
    if data_type == 'extreme':
        images_many_curves_1, array_annotations_many_curves_1 = add_extreme_data(images_many_curves_1,
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

    ######################################### 2 #########################################
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
    if data_type == 'extreme':
        images_nurburgring_1, array_annotations_nurburgring_1 = add_extreme_data(images_nurburgring_1,
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
    if data_type == 'extreme':
        images_monaco_1, array_annotations_monaco_1 = add_extreme_data(images_monaco_1, array_annotations_monaco_1)

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
    if data_type == 'extreme':
        images_extended_simple_1, array_annotations_extended_simple_1 = add_extreme_data(images_extended_simple_1,
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
    if data_type == 'extreme':
        images_only_curves_1, array_annotations_only_curves_1 = add_extreme_data(images_only_curves_1,
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
    if data_type == 'extreme':
        images_only_curves_2, array_annotations_only_curves_2 = add_extreme_data(images_only_curves_2,
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
    if data_type == 'extreme':
        images_only_curves_3, array_annotations_only_curves_3 = add_extreme_data(images_only_curves_3,
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
    if data_type == 'extreme':
        images_only_curves_4, array_annotations_only_curves_4 = add_extreme_data(images_only_curves_4,
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
    if data_type == 'extreme':
        images_only_curves_5, array_annotations_only_curves_5 = add_extreme_data(images_only_curves_5,
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
    if data_type == 'extreme':
        images_only_curves_6, array_annotations_only_curves_6 = add_extreme_data(images_only_curves_6,
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
    only_curves_7_name_file = path_to_data + 'only_curves_01_04_2022/monaco_1/data.csv'
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
    if data_type == 'extreme':
        images_only_curves_7, array_annotations_only_curves_7 = add_extreme_data(images_only_curves_7,
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
    if data_type == 'extreme':
        images_only_curves_8, array_annotations_only_curves_8 = add_extreme_data(images_only_curves_8,
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
    if data_type == 'extreme':
        images_only_curves_9, array_annotations_only_curves_9 = add_extreme_data(images_only_curves_9,
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
    if data_type == 'extreme':
        images_only_curves_10, array_annotations_only_curves_10 = add_extreme_data(images_only_curves_10,
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
    if data_type == 'extreme':
        images_only_curves_11, array_annotations_only_curves_11 = add_extreme_data(images_only_curves_11,
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
    if data_type == 'extreme':
        images_only_curves_12, array_annotations_only_curves_12 = add_extreme_data(images_only_curves_12,
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
    if data_type == 'extreme':
        images_difficult_situations_1, array_annotations_difficult_situations_1 = add_extreme_data(
            images_difficult_situations_1, array_annotations_difficult_situations_1)

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
    if data_type == 'extreme':
        images_difficult_situations_2, array_annotations_difficult_situations_2 = add_extreme_data(
            images_difficult_situations_2, array_annotations_difficult_situations_2)

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
    if data_type == 'extreme':
        images_difficult_situations_3, array_annotations_difficult_situations_3 = add_extreme_data(
            images_difficult_situations_3, array_annotations_difficult_situations_3)

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
    if data_type == 'extreme':
        images_difficult_situations_4, array_annotations_difficult_situations_4 = add_extreme_data(
            images_difficult_situations_4, array_annotations_difficult_situations_4)

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
    if data_type == 'extreme':
        images_difficult_situations_5, array_annotations_difficult_situations_5 = add_extreme_data(
            images_difficult_situations_5, array_annotations_difficult_situations_5)

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
    if data_type == 'extreme':
        images_difficult_situations_6, array_annotations_difficult_situations_6 = add_extreme_data(
            images_difficult_situations_6, array_annotations_difficult_situations_6)

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
    if data_type == 'extreme':
        images_difficult_situations_7, array_annotations_difficult_situations_7 = add_extreme_data(
            images_difficult_situations_7, array_annotations_difficult_situations_7)

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
    if data_type == 'extreme':
        images_difficult_situations_8, array_annotations_difficult_situations_8 = add_extreme_data(
            images_difficult_situations_8, array_annotations_difficult_situations_8)

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
    images_difficult_situations_9, array_annotations_difficult_situations_9 = flip_images(
        images_difficult_situations_9, array_annotations_difficult_situations_9)
    if data_type == 'extreme':
        images_difficult_situations_9, array_annotations_difficult_situations_9 = add_extreme_data(
            images_difficult_situations_9, array_annotations_difficult_situations_9)

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
    if data_type == 'extreme':
        images_difficult_situations_10, array_annotations_difficult_situations_10 = add_extreme_data(
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
    if data_type == 'extreme':
        images_difficult_situations_11, array_annotations_difficult_situations_11 = add_extreme_data(
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
    if data_type == 'extreme':
        images_difficult_situations_12, array_annotations_difficult_situations_12 = add_extreme_data(
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
    if data_type == 'extreme':
        images_difficult_situations_13, array_annotations_difficult_situations_13 = add_extreme_data(
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
    if data_type == 'extreme':
        images_difficult_situations_14, array_annotations_difficult_situations_14 = add_extreme_data(
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

    array_imgs = images_many_curves_1 + \
        images_nurburgring_1 + \
        images_monaco_1 + \
        images_extended_simple_1 + \
        images_only_curves_1 + images_only_curves_2 + images_only_curves_3 + images_only_curves_4 + images_only_curves_5 + images_only_curves_6 + \
        images_only_curves_7 + images_only_curves_8 + images_only_curves_9 + images_only_curves_10 + images_only_curves_11 + images_only_curves_12 + \
        images_difficult_situations_1 + images_difficult_situations_2 + images_difficult_situations_3 + \
        images_difficult_situations_4 + images_difficult_situations_5 + images_difficult_situations_6 + \
        images_difficult_situations_7 + images_difficult_situations_8 + images_difficult_situations_9 + \
        images_difficult_situations_10 + images_difficult_situations_11 + images_difficult_situations_12 + \
        images_difficult_situations_13 + images_difficult_situations_14

    array_annotations = array_annotations_many_curves_1 + \
        array_annotations_nurburgring_1 + \
        array_annotations_monaco_1 + \
        array_annotations_extended_simple_1 + \
        array_annotations_only_curves_1 + array_annotations_only_curves_2 + array_annotations_only_curves_3 + array_annotations_only_curves_4 + array_annotations_only_curves_5 + array_annotations_only_curves_6 + \
        array_annotations_only_curves_7 + array_annotations_only_curves_8 + array_annotations_only_curves_9 + array_annotations_only_curves_10 + array_annotations_only_curves_11 + array_annotations_only_curves_12 + \
        array_annotations_difficult_situations_1 + array_annotations_difficult_situations_2 + array_annotations_difficult_situations_3 + \
        array_annotations_difficult_situations_4 + array_annotations_difficult_situations_5 + array_annotations_difficult_situations_6 + \
        array_annotations_difficult_situations_7 + array_annotations_difficult_situations_8 + array_annotations_difficult_situations_9 + \
        array_annotations_difficult_situations_10 + array_annotations_difficult_situations_11 + array_annotations_difficult_situations_12 + \
        array_annotations_difficult_situations_13 + array_annotations_difficult_situations_14

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


def get_images_and_annotations_val(path_to_data, type_image, img_shape, data_type):
    ######################################### VAL 1 #########################################
    simple_circuit_name_file = path_to_data + 'simple_circuit_01_04_2022_clockwise_1/data.csv'
    dir_simple_circuit_images = path_to_data + 'simple_circuit_01_04_2022_clockwise_1/'
    list_images_simple_circuit_1 = glob.glob(dir_simple_circuit_images + '*')
    new_list_images_simple_circuit_1 = []
    for image in list_images_simple_circuit_1:
        if image != path_to_data + 'simple_circuit_01_04_2022_clockwise_1/data.csv':
            new_list_images_simple_circuit_1.append(image)
    list_images_simple_circuit_1 = new_list_images_simple_circuit_1
    images_paths_simple_circuit = sorted(list_images_simple_circuit_1, key=lambda x: int(x.split('/')[-1].split('.png')[0]))

    array_annotations_simple_circuit = pandas.read_csv(simple_circuit_name_file)
    array_annotations_simple_circuit = parse_csv(array_annotations_simple_circuit)

    images_simple_circuit = get_images(images_paths_simple_circuit, type_image, img_shape)
    images_simple_circuit, array_annotations_simple_circuit = flip_images(images_simple_circuit,
                                                                        array_annotations_simple_circuit)
    if data_type == 'extreme':
        images_simple_circuit, array_annotations_simple_circuit = add_extreme_data(images_simple_circuit,
                                                                                 array_annotations_simple_circuit)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_simple_circuit:
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

    array_annotations_simple_circuit = normalized_annotations
    print("Loaded Simple Circuit!!")

#     ######################################### VAL 2 #########################################
#     montmelo_name_file = path_to_data + 'montmelo_12_05_2022_opencv_clockwise_1/data.csv'
#     dir_montmelo_images = path_to_data + 'montmelo_12_05_2022_opencv_clockwise_1/'
#     list_images_montmelo = glob.glob(dir_montmelo_images + '*')
#     new_list_images_montmelo = []
#     for image in list_images_montmelo:
#         if image != path_to_data + 'montmelo_12_05_2022_opencv_clockwise_1/data.csv':
#             new_list_images_montmelo.append(image)
#     list_images_montmelo = new_list_images_montmelo
#     images_paths_montmelo = sorted(list_images_montmelo, key=lambda x: int(x.split('/')[-1].split('.png')[0]))

#     array_annotations_montmelo = pandas.read_csv(montmelo_name_file)
#     array_annotations_montmelo = parse_csv(array_annotations_montmelo)

#     images_montmelo = get_images(images_paths_montmelo, type_image, img_shape)
#     images_montmelo, array_annotations_montmelo = flip_images(images_montmelo,
#                                                                         array_annotations_montmelo)
#     if data_type == 'extreme':
#         images_montmelo, array_annotations_montmelo = add_extreme_data(images_montmelo,
#                                                                                  array_annotations_montmelo)

#     array_annotations_v = []
#     array_annotations_w = []
#     for annotation in array_annotations_montmelo:
#         array_annotations_v.append(annotation[0])
#         array_annotations_w.append(annotation[1])

#     # START NORMALIZE DATA
#     array_annotations_v = np.stack(array_annotations_v, axis=0)
#     array_annotations_v = array_annotations_v.reshape(-1, 1)

#     array_annotations_w = np.stack(array_annotations_w, axis=0)
#     array_annotations_w = array_annotations_w.reshape(-1, 1)

#     normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
#     normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

#     normalized_annotations = []
#     for i in range(0, len(normalized_x)):
#         normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

#     array_annotations_montmelo = normalized_annotations
#     print("Loaded Montmelo!!")
# ######################################### VAL 3 #########################################
#     montreal_name_file = path_to_data + 'montreal_12_05_2022_opencv_clockwise_1/data.csv'
#     dir_montreal_images = path_to_data + 'montreal_12_05_2022_opencv_clockwise_1/'
#     list_images_montreal = glob.glob(dir_montreal_images + '*')
#     new_list_images_montreal = []
#     for image in list_images_montreal:
#         if image != path_to_data + 'montreal_12_05_2022_opencv_clockwise_1/data.csv':
#             new_list_images_montreal.append(image)
#     list_images_montreal = new_list_images_montreal
#     images_paths_montreal = sorted(list_images_montreal, key=lambda x: int(x.split('/')[-1].split('.png')[0]))

#     array_annotations_montreal = pandas.read_csv(montreal_name_file)
#     array_annotations_montreal = parse_csv(array_annotations_montreal)

#     images_montreal = get_images(images_paths_montreal, type_image, img_shape)
#     images_montreal, array_annotations_montreal = flip_images(images_montreal,
#                                                                         array_annotations_montreal)
#     if data_type == 'extreme':
#         images_montreal, array_annotations_montreal = add_extreme_data(images_montreal,
#                                                                                  array_annotations_montreal)

#     array_annotations_v = []
#     array_annotations_w = []
#     for annotation in array_annotations_montreal:
#         array_annotations_v.append(annotation[0])
#         array_annotations_w.append(annotation[1])

#     # START NORMALIZE DATA
#     array_annotations_v = np.stack(array_annotations_v, axis=0)
#     array_annotations_v = array_annotations_v.reshape(-1, 1)

#     array_annotations_w = np.stack(array_annotations_w, axis=0)
#     array_annotations_w = array_annotations_w.reshape(-1, 1)

#     normalized_x = np.interp(array_annotations_v, (6.5, 24), (0, 1))
#     normalized_y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))

#     normalized_annotations = []
#     for i in range(0, len(normalized_x)):
#         normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

#     array_annotations_montreal = normalized_annotations
#     print("Loaded Montreal!!")
#     #############################################################################

    array_imgs = images_simple_circuit #+ \
                    # images_montmelo + \
                    # images_montreal

    array_annotations = array_annotations_simple_circuit #+ \
                        # array_annotations_montmelo + \
                        # array_annotations_montreal

    return array_imgs, array_annotations


def process_dataset(path_to_data, type_image, data_type, img_shape, optimize_mode=False):

    if not optimize_mode:
        array_imgs, array_annotations = get_images_and_annotations(path_to_data, type_image, img_shape, data_type)
        images_train, annotations_train, images_validation, annotations_validation = separate_dataset_into_train_validation(
            array_imgs, array_annotations)
    else:
        # images_train, annotations_train = get_images_and_annotations(path_to_data, type_image, img_shape, data_type)
        images_train, annotations_train = [], []
        images_validation, annotations_validation = get_images_and_annotations_val(path_to_data, type_image, img_shape, data_type)
            

    return images_train, annotations_train, images_validation, annotations_validation
