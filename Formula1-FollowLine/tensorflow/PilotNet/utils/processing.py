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


def get_images_and_annotations(path_to_data, type_image, img_shape):
    print('---- many_curves_01_04_2022_clockwise_1 ----')
    many_curves_1_name_file = path_to_data + 'many_curves_01_04_2022_clockwise_1/data.csv'
    many_curves_1_file = open(many_curves_1_name_file, 'r')
    data_many_curves_1 = many_curves_1_file.read()
    many_curves_1_file.close()

    array_annotations_many_curves_1 = []
    DIR_many_curves_1_images = path_to_data + 'many_curves_01_04_2022_clockwise_1/'
    list_images_many_curves_1 = glob.glob(DIR_many_curves_1_images + '*')
    new_list_images_many_curves_1 = []
    for image in list_images_many_curves_1:
        if image != path_to_data + 'many_curves_01_04_2022_clockwise_1/data.csv':
            new_list_images_many_curves_1.append(image)
    list_images_many_curves_1 = new_list_images_many_curves_1
    images_paths_many_curves_1 = sorted(list_images_many_curves_1, key=lambda x: int(x.split('/')[6].split('.png')[0]))

    array_annotations_many_curves_1 = pandas.read_csv(many_curves_1_name_file)
    array_annotations_many_curves_1 = parse_csv(array_annotations_many_curves_1)

    images_many_curves_1 = get_images(images_paths_many_curves_1, 'cropped', img_shape)
    images_many_curves_1, array_annotations_many_curves_1 = flip_images(images_many_curves_1,
                                                                        array_annotations_many_curves_1)
    print(len(images_many_curves_1))
    print(type(images_many_curves_1))
    print(len(array_annotations_many_curves_1))
    images_many_curves_1, array_annotations_many_curves_1 = add_extreme_data(images_many_curves_1,
                                                                             array_annotations_many_curves_1)
    print(len(images_many_curves_1))
    print(type(images_many_curves_1))
    print(len(array_annotations_many_curves_1))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_many_curves_1 = normalized_annotations

    print(len(images_many_curves_1))
    print(type(images_many_curves_1))
    print(len(array_annotations_many_curves_1))

    ########################################################################################################################### 9 ###########################################################################################################################
    print('---- nurburgring_01_04_2022_clockwise_1 ----')
    nurburgring_1_name_file = path_to_data + 'nurburgring_01_04_2022_clockwise_1/data.csv'
    nurburgring_1_file = open(nurburgring_1_name_file, 'r')
    # data_nurburgring_1 = nurburgring_1_file.read()
    # nurburgring_1_file.close()

    array_annotations_nurburgring_1 = []
    DIR_nurburgring_1_images = path_to_data + 'nurburgring_01_04_2022_clockwise_1/'
    list_images_nurburgring_1 = glob.glob(DIR_nurburgring_1_images + '*')
    new_list_images_nurburgring_1 = []
    for image in list_images_nurburgring_1:
        if image != path_to_data + 'nurburgring_01_04_2022_clockwise_1/data.csv':
            new_list_images_nurburgring_1.append(image)
    list_images_nurburgring_1 = new_list_images_nurburgring_1
    images_paths_nurburgring_1 = sorted(list_images_nurburgring_1, key=lambda x: int(x.split('/')[6].split('.png')[0]))

    array_annotations_nurburgring_1 = pandas.read_csv(nurburgring_1_name_file)
    array_annotations_nurburgring_1 = parse_csv(array_annotations_nurburgring_1)

    images_nurburgring_1 = get_images(images_paths_nurburgring_1, 'cropped', img_shape)
    print(len(images_nurburgring_1))
    print(len(array_annotations_nurburgring_1))
    images_nurburgring_1, array_annotations_nurburgring_1 = flip_images(images_nurburgring_1,
                                                                        array_annotations_nurburgring_1)
    print(len(images_nurburgring_1))
    print(type(images_nurburgring_1))
    print(len(array_annotations_nurburgring_1))
    images_nurburgring_1, array_annotations_nurburgring_1 = add_extreme_data(images_nurburgring_1,
                                                                             array_annotations_nurburgring_1)
    print(len(images_nurburgring_1))
    print(type(images_nurburgring_1))
    print(len(array_annotations_nurburgring_1))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_nurburgring_1 = normalized_annotations

    print(len(images_nurburgring_1))
    print(type(images_nurburgring_1))
    print(len(array_annotations_nurburgring_1))

    ########################################################################################################################### 13 ###########################################################################################################################
    print('---- monaco_01_04_2022_clockwise_1 ----')
    monaco_1_name_file = path_to_data + 'monaco_01_04_2022_clockwise_1/data.csv'
    monaco_1_file = open(monaco_1_name_file, 'r')
    data_monaco_1 = monaco_1_file.read()
    monaco_1_file.close()

    array_annotations_monaco_1 = []
    DIR_monaco_1_images = path_to_data + 'monaco_01_04_2022_clockwise_1/'
    list_images_monaco_1 = glob.glob(DIR_monaco_1_images + '*')
    new_list_images_monaco_1 = []
    for image in list_images_monaco_1:
        if image != path_to_data + 'monaco_01_04_2022_clockwise_1/data.csv':
            new_list_images_monaco_1.append(image)
    list_images_monaco_1 = new_list_images_monaco_1
    images_paths_monaco_1 = sorted(list_images_monaco_1, key=lambda x: int(x.split('/')[6].split('.png')[0]))

    array_annotations_monaco_1 = pandas.read_csv(monaco_1_name_file)
    array_annotations_monaco_1 = parse_csv(array_annotations_monaco_1)

    images_monaco_1 = get_images(images_paths_monaco_1, 'cropped', img_shape)
    images_monaco_1, array_annotations_monaco_1 = flip_images(images_monaco_1, array_annotations_monaco_1)
    print(len(images_monaco_1))
    print(type(images_monaco_1))
    print(len(array_annotations_monaco_1))
    images_monaco_1, array_annotations_monaco_1 = add_extreme_data(images_monaco_1, array_annotations_monaco_1)
    print(len(images_monaco_1))
    print(type(images_monaco_1))
    print(len(array_annotations_monaco_1))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_monaco_1 = normalized_annotations

    print(len(images_monaco_1))
    print(type(images_monaco_1))
    print(len(array_annotations_monaco_1))

    ########################################################################################################################### 15 ###########################################################################################################################
    print('---- extended_simple_circuit_01_04_2022_clockwise_1 ----')
    extended_simple_1_name_file = path_to_data + 'extended_simple_circuit_01_04_2022_clockwise_1/data.csv'
    extended_simple_1_file = open(extended_simple_1_name_file, 'r')
    data_extended_simple_1 = extended_simple_1_file.read()
    extended_simple_1_file.close()

    array_annotations_extended_simple_1 = []
    DIR_extended_simple_1_images = path_to_data + 'extended_simple_circuit_01_04_2022_clockwise_1/'
    list_images_extended_simple_1 = glob.glob(DIR_extended_simple_1_images + '*')
    new_list_images_extended_simple_1 = []
    for image in list_images_extended_simple_1:
        if image != path_to_data + 'extended_simple_circuit_01_04_2022_clockwise_1/data.csv':
            new_list_images_extended_simple_1.append(image)
    list_images_extended_simple_1 = new_list_images_extended_simple_1
    images_paths_extended_simple_1 = sorted(list_images_extended_simple_1,
                                            key=lambda x: int(x.split('/')[6].split('.png')[0]))

    array_annotations_extended_simple_1 = pandas.read_csv(extended_simple_1_name_file)
    array_annotations_extended_simple_1 = parse_csv(array_annotations_extended_simple_1)

    images_extended_simple_1 = get_images(images_paths_extended_simple_1, 'cropped', img_shape)
    images_extended_simple_1, array_annotations_extended_simple_1 = flip_images(images_extended_simple_1,
                                                                                array_annotations_extended_simple_1)
    print(len(images_extended_simple_1))
    print(type(images_extended_simple_1))
    print(len(array_annotations_extended_simple_1))
    images_extended_simple_1, array_annotations_extended_simple_1 = add_extreme_data(images_extended_simple_1,
                                                                                     array_annotations_extended_simple_1)
    print(len(images_extended_simple_1))
    print(type(images_extended_simple_1))
    print(len(array_annotations_extended_simple_1))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_extended_simple_1 = normalized_annotations

    print(len(images_extended_simple_1))
    print(type(images_extended_simple_1))
    print(len(array_annotations_extended_simple_1))

    ##################################################################
    ############################# CURVES #############################
    ##################################################################
    ########################################################################################################################### 19 ###########################################################################################################################
    print('---- only_curves_01_04_2022/nurburgring_1 ----')
    only_curves_1_name_file = path_to_data + 'only_curves_01_04_2022/nurburgring_1/data.csv'
    only_curves_1_file = open(only_curves_1_name_file, 'r')
    data_only_curves_1 = only_curves_1_file.read()
    only_curves_1_file.close()

    array_annotations_only_curves_1 = []
    DIR_only_curves_1_images = path_to_data + 'only_curves_01_04_2022/nurburgring_1/'
    list_images_only_curves_1 = glob.glob(DIR_only_curves_1_images + '*')
    new_list_images_only_curves_1 = []
    for image in list_images_only_curves_1:
        if image != path_to_data + 'only_curves_01_04_2022/nurburgring_1/data.csv':
            new_list_images_only_curves_1.append(image)
    list_images_only_curves_1 = new_list_images_only_curves_1
    images_paths_only_curves_1 = sorted(list_images_only_curves_1, key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_1 = pandas.read_csv(only_curves_1_name_file)
    array_annotations_only_curves_1 = parse_csv(array_annotations_only_curves_1)

    images_only_curves_1 = get_images(images_paths_only_curves_1, 'cropped', img_shape)
    images_only_curves_1, array_annotations_only_curves_1 = flip_images(images_only_curves_1,
                                                                        array_annotations_only_curves_1)
    print(len(images_only_curves_1))
    print(type(images_only_curves_1))
    print(len(array_annotations_only_curves_1))
    images_only_curves_1, array_annotations_only_curves_1 = add_extreme_data(images_only_curves_1,
                                                                             array_annotations_only_curves_1)
    print(len(images_only_curves_1))
    print(type(images_only_curves_1))
    print(len(array_annotations_only_curves_1))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_only_curves_1 = normalized_annotations

    print(len(images_only_curves_1))
    print(type(images_only_curves_1))
    print(len(array_annotations_only_curves_1))

    ########################################################################################################################### 20 ###########################################################################################################################
    print('---- only_curves_01_04_2022/nurburgring_2 ----')
    only_curves_2_name_file = path_to_data + 'only_curves_01_04_2022/nurburgring_2/data.csv'
    only_curves_2_file = open(only_curves_2_name_file, 'r')
    data_only_curves_2 = only_curves_2_file.read()
    only_curves_2_file.close()

    array_annotations_only_curves_2 = []
    DIR_only_curves_2_images = path_to_data + 'only_curves_01_04_2022/nurburgring_2/'
    list_images_only_curves_2 = glob.glob(DIR_only_curves_2_images + '*')
    new_list_images_only_curves_2 = []
    for image in list_images_only_curves_2:
        if image != path_to_data + 'only_curves_01_04_2022/nurburgring_2/data.csv':
            new_list_images_only_curves_2.append(image)
    list_images_only_curves_2 = new_list_images_only_curves_2
    images_paths_only_curves_2 = sorted(list_images_only_curves_2, key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_2 = pandas.read_csv(only_curves_2_name_file)
    array_annotations_only_curves_2 = parse_csv(array_annotations_only_curves_2)

    images_only_curves_2 = get_images(images_paths_only_curves_2, 'cropped', img_shape)
    images_only_curves_2, array_annotations_only_curves_2 = flip_images(images_only_curves_2,
                                                                        array_annotations_only_curves_2)
    print(len(images_only_curves_2))
    print(type(images_only_curves_2))
    print(len(array_annotations_only_curves_2))
    images_only_curves_2, array_annotations_only_curves_2 = add_extreme_data(images_only_curves_2,
                                                                             array_annotations_only_curves_2)
    print(len(images_only_curves_2))
    print(type(images_only_curves_2))
    print(len(array_annotations_only_curves_2))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_only_curves_2 = normalized_annotations

    print(len(images_only_curves_2))
    print(type(images_only_curves_2))
    print(len(array_annotations_only_curves_2))

    ########################################################################################################################### 22 ###########################################################################################################################
    print('---- only_curves_01_04_2022/nurburgring_3 ----')
    only_curves_3_name_file = path_to_data + 'only_curves_01_04_2022/nurburgring_3/data.csv'
    only_curves_3_file = open(only_curves_3_name_file, 'r')
    data_only_curves_3 = only_curves_3_file.read()
    only_curves_3_file.close()

    array_annotations_only_curves_3 = []
    DIR_only_curves_3_images = path_to_data + 'only_curves_01_04_2022/nurburgring_3/'
    list_images_only_curves_3 = glob.glob(DIR_only_curves_3_images + '*')
    new_list_images_only_curves_3 = []
    for image in list_images_only_curves_3:
        if image != path_to_data + 'only_curves_01_04_2022/nurburgring_3/data.csv':
            new_list_images_only_curves_3.append(image)
    list_images_only_curves_3 = new_list_images_only_curves_3
    images_paths_only_curves_3 = sorted(list_images_only_curves_3, key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_3 = pandas.read_csv(only_curves_3_name_file)
    array_annotations_only_curves_3 = parse_csv(array_annotations_only_curves_3)

    images_only_curves_3 = get_images(images_paths_only_curves_3, 'cropped', img_shape)
    images_only_curves_3, array_annotations_only_curves_3 = flip_images(images_only_curves_3,
                                                                        array_annotations_only_curves_3)
    print(len(images_only_curves_3))
    print(type(images_only_curves_3))
    print(len(array_annotations_only_curves_3))
    images_only_curves_3, array_annotations_only_curves_3 = add_extreme_data(images_only_curves_3,
                                                                             array_annotations_only_curves_3)
    print(len(images_only_curves_3))
    print(type(images_only_curves_3))
    print(len(array_annotations_only_curves_3))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_only_curves_3 = normalized_annotations

    print(len(images_only_curves_3))
    print(type(images_only_curves_3))
    print(len(array_annotations_only_curves_3))

    ########################################################################################################################### 23 ###########################################################################################################################
    print('----only_curves_01_04_2022/nurburgring_4 ----')
    only_curves_4_name_file = path_to_data + 'only_curves_01_04_2022/nurburgring_4/data.csv'
    only_curves_4_file = open(only_curves_4_name_file, 'r')
    data_only_curves_4 = only_curves_4_file.read()
    only_curves_4_file.close()

    array_annotations_only_curves_4 = []
    DIR_only_curves_4_images = path_to_data + 'only_curves_01_04_2022/nurburgring_4/'
    list_images_only_curves_4 = glob.glob(DIR_only_curves_4_images + '*')
    new_list_images_only_curves_4 = []
    for image in list_images_only_curves_4:
        if image != path_to_data + 'only_curves_01_04_2022/nurburgring_4/data.csv':
            new_list_images_only_curves_4.append(image)
    list_images_only_curves_4 = new_list_images_only_curves_4
    images_paths_only_curves_4 = sorted(list_images_only_curves_4, key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_4 = pandas.read_csv(only_curves_4_name_file)
    array_annotations_only_curves_4 = parse_csv(array_annotations_only_curves_4)

    images_only_curves_4 = get_images(images_paths_only_curves_4, 'cropped', img_shape)
    images_only_curves_4, array_annotations_only_curves_4 = flip_images(images_only_curves_4,
                                                                        array_annotations_only_curves_4)
    print(len(images_only_curves_4))
    print(type(images_only_curves_4))
    print(len(array_annotations_only_curves_4))
    images_only_curves_4, array_annotations_only_curves_4 = add_extreme_data(images_only_curves_4,
                                                                             array_annotations_only_curves_4)
    print(len(images_only_curves_4))
    print(type(images_only_curves_4))
    print(len(array_annotations_only_curves_4))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_only_curves_4 = normalized_annotations

    print(len(images_only_curves_4))
    print(type(images_only_curves_4))
    print(len(array_annotations_only_curves_4))

    ########################################################################################################################### 24 ###########################################################################################################################
    print('---- only_curves_01_04_2022/nurburgring_5 ----')
    only_curves_5_name_file = path_to_data + 'only_curves_01_04_2022/nurburgring_5/data.csv'
    only_curves_5_file = open(only_curves_5_name_file, 'r')
    data_only_curves_5 = only_curves_5_file.read()
    only_curves_5_file.close()

    array_annotations_only_curves_5 = []
    DIR_only_curves_5_images = path_to_data + 'only_curves_01_04_2022/nurburgring_5/'
    list_images_only_curves_5 = glob.glob(DIR_only_curves_5_images + '*')
    new_list_images_only_curves_5 = []
    for image in list_images_only_curves_5:
        if image != path_to_data + 'only_curves_01_04_2022/nurburgring_5/data.csv':
            new_list_images_only_curves_5.append(image)
    list_images_only_curves_5 = new_list_images_only_curves_5
    images_paths_only_curves_5 = sorted(list_images_only_curves_5, key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_5 = pandas.read_csv(only_curves_5_name_file)
    array_annotations_only_curves_5 = parse_csv(array_annotations_only_curves_5)

    images_only_curves_5 = get_images(images_paths_only_curves_5, 'cropped', img_shape)
    images_only_curves_5, array_annotations_only_curves_5 = flip_images(images_only_curves_5,
                                                                        array_annotations_only_curves_5)
    print(len(images_only_curves_5))
    print(type(images_only_curves_5))
    print(len(array_annotations_only_curves_5))
    images_only_curves_5, array_annotations_only_curves_5 = add_extreme_data(images_only_curves_5,
                                                                             array_annotations_only_curves_5)
    print(len(images_only_curves_5))
    print(type(images_only_curves_5))
    print(len(array_annotations_only_curves_5))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_only_curves_5 = normalized_annotations

    print(len(images_only_curves_5))
    print(type(images_only_curves_5))
    print(len(array_annotations_only_curves_5))

    ########################################################################################################################### 25 ###########################################################################################################################
    print('---- only_curves_01_04_2022/nurburgring_6 ----')
    only_curves_6_name_file = path_to_data + 'only_curves_01_04_2022/nurburgring_6/data.csv'
    only_curves_6_file = open(only_curves_6_name_file, 'r')
    data_only_curves_6 = only_curves_6_file.read()
    only_curves_6_file.close()

    array_annotations_only_curves_6 = []
    DIR_only_curves_6_images = path_to_data + 'only_curves_01_04_2022/nurburgring_6/'
    list_images_only_curves_6 = glob.glob(DIR_only_curves_6_images + '*')
    new_list_images_only_curves_6 = []
    for image in list_images_only_curves_6:
        if image != path_to_data + 'only_curves_01_04_2022/nurburgring_6/data.csv':
            new_list_images_only_curves_6.append(image)
    list_images_only_curves_6 = new_list_images_only_curves_6
    images_paths_only_curves_6 = sorted(list_images_only_curves_6, key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_6 = pandas.read_csv(only_curves_6_name_file)
    array_annotations_only_curves_6 = parse_csv(array_annotations_only_curves_6)

    images_only_curves_6 = get_images(images_paths_only_curves_6, 'cropped', img_shape)
    images_only_curves_6, array_annotations_only_curves_6 = flip_images(images_only_curves_6,
                                                                        array_annotations_only_curves_6)
    print(len(images_only_curves_6))
    print(type(images_only_curves_6))
    print(len(array_annotations_only_curves_6))
    images_only_curves_6, array_annotations_only_curves_6 = add_extreme_data(images_only_curves_6,
                                                                             array_annotations_only_curves_6)
    print(len(images_only_curves_6))
    print(type(images_only_curves_6))
    print(len(array_annotations_only_curves_6))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_only_curves_6 = normalized_annotations

    print(len(images_only_curves_6))
    print(type(images_only_curves_6))
    print(len(array_annotations_only_curves_6))

    ########################################################################################################################### 26 ###########################################################################################################################
    print('---- only_curves_01_04_2022/monaco_1 ----')
    only_curves_7_name_file = path_to_data + 'only_curves_01_04_2022/monaco_1/data.csv'
    only_curves_7_file = open(only_curves_7_name_file, 'r')
    data_only_curves_7 = only_curves_7_file.read()
    only_curves_7_file.close()

    array_annotations_only_curves_7 = []
    DIR_only_curves_7_images = path_to_data + 'only_curves_01_04_2022/monaco_1/'
    list_images_only_curves_7 = glob.glob(DIR_only_curves_7_images + '*')
    new_list_images_only_curves_7 = []
    for image in list_images_only_curves_7:
        if image != path_to_data + 'only_curves_01_04_2022/monaco_1/data.csv':
            new_list_images_only_curves_7.append(image)
    list_images_only_curves_7 = new_list_images_only_curves_7
    images_paths_only_curves_7 = sorted(list_images_only_curves_7, key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_7 = pandas.read_csv(only_curves_7_name_file)
    array_annotations_only_curves_7 = parse_csv(array_annotations_only_curves_7)

    images_only_curves_7 = get_images(images_paths_only_curves_7, 'cropped', img_shape)
    images_only_curves_7, array_annotations_only_curves_7 = flip_images(images_only_curves_7,
                                                                        array_annotations_only_curves_7)
    print(len(images_only_curves_7))
    print(type(images_only_curves_7))
    print(len(array_annotations_only_curves_7))
    images_only_curves_7, array_annotations_only_curves_7 = add_extreme_data(images_only_curves_7,
                                                                             array_annotations_only_curves_7)
    print(len(images_only_curves_7))
    print(type(images_only_curves_7))
    print(len(array_annotations_only_curves_7))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_only_curves_7 = normalized_annotations

    print(len(images_only_curves_7))
    print(type(images_only_curves_7))
    print(len(array_annotations_only_curves_7))

    ########################################################################################################################### 27 ###########################################################################################################################
    print('---- only_curves_01_04_2022/monaco_2 ----')
    only_curves_8_name_file = path_to_data + 'only_curves_01_04_2022/monaco_2/data.csv'
    only_curves_8_file = open(only_curves_8_name_file, 'r')
    data_only_curves_8 = only_curves_8_file.read()
    only_curves_8_file.close()

    array_annotations_only_curves_8 = []
    DIR_only_curves_8_images = path_to_data + 'only_curves_01_04_2022/monaco_2/'
    list_images_only_curves_8 = glob.glob(DIR_only_curves_8_images + '*')
    new_list_images_only_curves_8 = []
    for image in list_images_only_curves_8:
        if image != path_to_data + 'only_curves_01_04_2022/monaco_2/data.csv':
            new_list_images_only_curves_8.append(image)
    list_images_only_curves_8 = new_list_images_only_curves_8
    images_paths_only_curves_8 = sorted(list_images_only_curves_8, key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_8 = pandas.read_csv(only_curves_8_name_file)
    array_annotations_only_curves_8 = parse_csv(array_annotations_only_curves_8)

    images_only_curves_8 = get_images(images_paths_only_curves_8, 'cropped', img_shape)
    images_only_curves_8, array_annotations_only_curves_8 = flip_images(images_only_curves_8,
                                                                        array_annotations_only_curves_8)
    print(len(images_only_curves_8))
    print(type(images_only_curves_8))
    print(len(array_annotations_only_curves_8))
    images_only_curves_8, array_annotations_only_curves_8 = add_extreme_data(images_only_curves_8,
                                                                             array_annotations_only_curves_8)
    print(len(images_only_curves_8))
    print(type(images_only_curves_8))
    print(len(array_annotations_only_curves_8))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_only_curves_8 = normalized_annotations

    print(len(images_only_curves_8))
    print(type(images_only_curves_8))
    print(len(array_annotations_only_curves_8))

    ########################################################################################################################### 28 ###########################################################################################################################
    print('---- only_curves_01_04_2022/monaco_3 ----')
    only_curves_9_name_file = path_to_data + 'only_curves_01_04_2022/monaco_3/data.csv'
    only_curves_9_file = open(only_curves_9_name_file, 'r')
    data_only_curves_9 = only_curves_9_file.read()
    only_curves_9_file.close()

    array_annotations_only_curves_9 = []
    DIR_only_curves_9_images = path_to_data + 'only_curves_01_04_2022/monaco_3/'
    list_images_only_curves_9 = glob.glob(DIR_only_curves_9_images + '*')
    new_list_images_only_curves_9 = []
    for image in list_images_only_curves_9:
        if image != path_to_data + 'only_curves_01_04_2022/monaco_3/data.csv':
            new_list_images_only_curves_9.append(image)
    list_images_only_curves_9 = new_list_images_only_curves_9
    images_paths_only_curves_9 = sorted(list_images_only_curves_9, key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_9 = pandas.read_csv(only_curves_9_name_file)
    array_annotations_only_curves_9 = parse_csv(array_annotations_only_curves_9)

    images_only_curves_9 = get_images(images_paths_only_curves_9, 'cropped', img_shape)
    images_only_curves_9, array_annotations_only_curves_9 = flip_images(images_only_curves_9,
                                                                        array_annotations_only_curves_9)
    print(len(images_only_curves_9))
    print(type(images_only_curves_9))
    print(len(array_annotations_only_curves_9))
    images_only_curves_9, array_annotations_only_curves_9 = add_extreme_data(images_only_curves_9,
                                                                             array_annotations_only_curves_9)
    print(len(images_only_curves_9))
    print(type(images_only_curves_9))
    print(len(array_annotations_only_curves_9))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_only_curves_9 = normalized_annotations

    print(len(images_only_curves_9))
    print(type(images_only_curves_9))
    print(len(array_annotations_only_curves_9))

    ########################################################################################################################### 29 ###########################################################################################################################
    print('---- only_curves_01_04_2022/monaco_4 ----')
    only_curves_10_name_file = path_to_data + 'only_curves_01_04_2022/monaco_4/data.csv'
    only_curves_10_file = open(only_curves_10_name_file, 'r')
    data_only_curves_10 = only_curves_10_file.read()
    only_curves_10_file.close()

    array_annotations_only_curves_10 = []
    DIR_only_curves_10_images = path_to_data + 'only_curves_01_04_2022/monaco_4/'
    list_images_only_curves_10 = glob.glob(DIR_only_curves_10_images + '*')
    new_list_images_only_curves_10 = []
    for image in list_images_only_curves_10:
        if image != path_to_data + 'only_curves_01_04_2022/monaco_4/data.csv':
            new_list_images_only_curves_10.append(image)
    list_images_only_curves_10 = new_list_images_only_curves_10
    images_paths_only_curves_10 = sorted(list_images_only_curves_10,
                                         key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_10 = pandas.read_csv(only_curves_10_name_file)
    array_annotations_only_curves_10 = parse_csv(array_annotations_only_curves_10)

    images_only_curves_10 = get_images(images_paths_only_curves_10, 'cropped', img_shape)
    images_only_curves_10, array_annotations_only_curves_10 = flip_images(images_only_curves_10,
                                                                          array_annotations_only_curves_10)
    print(len(images_only_curves_10))
    print(type(images_only_curves_10))
    print(len(array_annotations_only_curves_10))
    images_only_curves_10, array_annotations_only_curves_10 = add_extreme_data(images_only_curves_10,
                                                                               array_annotations_only_curves_10)
    print(len(images_only_curves_10))
    print(type(images_only_curves_10))
    print(len(array_annotations_only_curves_10))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_only_curves_10 = normalized_annotations

    print(len(images_only_curves_10))
    print(type(images_only_curves_10))
    print(len(array_annotations_only_curves_10))

    ########################################################################################################################### 20 ###########################################################################################################################
    print('---- only_curves_01_04_2022/many_curves_1 ----')
    only_curves_11_name_file = path_to_data + 'only_curves_01_04_2022/many_curves_1/data.csv'
    only_curves_11_file = open(only_curves_11_name_file, 'r')
    data_only_curves_11 = only_curves_11_file.read()
    only_curves_11_file.close()

    array_annotations_only_curves_11 = []
    DIR_only_curves_11_images = path_to_data + 'only_curves_01_04_2022/many_curves_1/'
    list_images_only_curves_11 = glob.glob(DIR_only_curves_11_images + '*')
    new_list_images_only_curves_11 = []
    for image in list_images_only_curves_11:
        if image != path_to_data + 'only_curves_01_04_2022/many_curves_1/data.csv':
            new_list_images_only_curves_11.append(image)
    list_images_only_curves_11 = new_list_images_only_curves_11
    images_paths_only_curves_11 = sorted(list_images_only_curves_11,
                                         key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_11 = pandas.read_csv(only_curves_11_name_file)
    array_annotations_only_curves_11 = parse_csv(array_annotations_only_curves_11)

    images_only_curves_11 = get_images(images_paths_only_curves_11, 'cropped', img_shape)
    images_only_curves_11, array_annotations_only_curves_11 = flip_images(images_only_curves_11,
                                                                          array_annotations_only_curves_11)
    print(len(images_only_curves_11))
    print(type(images_only_curves_11))
    print(len(array_annotations_only_curves_11))
    images_only_curves_11, array_annotations_only_curves_11 = add_extreme_data(images_only_curves_11,
                                                                               array_annotations_only_curves_11)
    print(len(images_only_curves_11))
    print(type(images_only_curves_11))
    print(len(array_annotations_only_curves_11))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_only_curves_11 = normalized_annotations

    print(len(images_only_curves_11))
    print(type(images_only_curves_11))
    print(len(array_annotations_only_curves_11))

    ########################################################################################################################### 20 ###########################################################################################################################
    print('---- only_curves_01_04_2022/many_curves_2 ----')
    only_curves_12_name_file = path_to_data + 'only_curves_01_04_2022/many_curves_2/data.csv'
    only_curves_12_file = open(only_curves_12_name_file, 'r')
    data_only_curves_12 = only_curves_12_file.read()
    only_curves_12_file.close()

    array_annotations_only_curves_12 = []
    DIR_only_curves_12_images = path_to_data + 'only_curves_01_04_2022/many_curves_2/'
    list_images_only_curves_12 = glob.glob(DIR_only_curves_12_images + '*')
    new_list_images_only_curves_12 = []
    for image in list_images_only_curves_12:
        if image != path_to_data + 'only_curves_01_04_2022/many_curves_2/data.csv':
            new_list_images_only_curves_12.append(image)
    list_images_only_curves_12 = new_list_images_only_curves_12
    images_paths_only_curves_12 = sorted(list_images_only_curves_12,
                                         key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_only_curves_12 = pandas.read_csv(only_curves_12_name_file)
    array_annotations_only_curves_12 = parse_csv(array_annotations_only_curves_12)

    images_only_curves_12 = get_images(images_paths_only_curves_12, 'cropped', img_shape)
    images_only_curves_12, array_annotations_only_curves_12 = flip_images(images_only_curves_12,
                                                                          array_annotations_only_curves_12)
    print(len(images_only_curves_12))
    print(type(images_only_curves_12))
    print(len(array_annotations_only_curves_12))
    images_only_curves_12, array_annotations_only_curves_12 = add_extreme_data(images_only_curves_12,
                                                                               array_annotations_only_curves_12)
    print(len(images_only_curves_12))
    print(type(images_only_curves_12))
    print(len(array_annotations_only_curves_12))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_only_curves_12 = normalized_annotations

    print(len(images_only_curves_12))
    print(type(images_only_curves_12))
    print(len(array_annotations_only_curves_12))

    ########################################################################################################################### 21 ###########################################################################################################################
    print('---- difficult_situations_01_04_2022/many_curves_1 ----')
    difficult_situations_1_name_file = path_to_data + 'difficult_situations_01_04_2022/many_curves_1/data.csv'
    difficult_situations_1_file = open(difficult_situations_1_name_file, 'r')
    data_difficult_situations_1 = difficult_situations_1_file.read()
    difficult_situations_1_file.close()

    array_annotations_difficult_situations_1 = []
    DIR_difficult_situations_1_images = path_to_data + 'difficult_situations_01_04_2022/many_curves_1/'
    list_images_difficult_situations_1 = glob.glob(DIR_difficult_situations_1_images + '*')
    new_list_images_difficult_situations_1 = []
    for image in list_images_difficult_situations_1:
        if image != path_to_data + 'difficult_situations_01_04_2022/many_curves_1/data.csv':
            new_list_images_difficult_situations_1.append(image)
    list_images_difficult_situations_1 = new_list_images_difficult_situations_1
    images_paths_difficult_situations_1 = sorted(list_images_difficult_situations_1,
                                                 key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_1 = pandas.read_csv(difficult_situations_1_name_file)
    array_annotations_difficult_situations_1 = parse_csv(array_annotations_difficult_situations_1)

    images_difficult_situations_1 = get_images(images_paths_difficult_situations_1, 'cropped', img_shape)
    images_difficult_situations_1, array_annotations_difficult_situations_1 = flip_images(images_difficult_situations_1,
                                                                                          array_annotations_difficult_situations_1)
    print(len(images_difficult_situations_1))
    print(type(images_difficult_situations_1))
    print(len(array_annotations_difficult_situations_1))
    images_difficult_situations_1, array_annotations_difficult_situations_1 = add_extreme_data(
        images_difficult_situations_1, array_annotations_difficult_situations_1)
    print(len(images_difficult_situations_1))
    print(type(images_difficult_situations_1))
    print(len(array_annotations_difficult_situations_1))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_difficult_situations_1 = normalized_annotations

    print(len(images_difficult_situations_1))
    print(type(images_difficult_situations_1))
    print(len(array_annotations_difficult_situations_1))

    ########################################################################################################################### 22 ###########################################################################################################################
    print('---- difficult_situations_01_04_2022/many_curves_2 ----')
    difficult_situations_2_name_file = path_to_data + 'difficult_situations_01_04_2022/many_curves_2/data.csv'
    difficult_situations_2_file = open(difficult_situations_2_name_file, 'r')
    data_difficult_situations_2 = difficult_situations_2_file.read()
    difficult_situations_2_file.close()

    array_annotations_difficult_situations_2 = []
    DIR_difficult_situations_2_images = path_to_data + 'difficult_situations_01_04_2022/many_curves_2/'
    list_images_difficult_situations_2 = glob.glob(DIR_difficult_situations_2_images + '*')
    new_list_images_difficult_situations_2 = []
    for image in list_images_difficult_situations_2:
        if image != path_to_data + 'difficult_situations_01_04_2022/many_curves_2/data.csv':
            new_list_images_difficult_situations_2.append(image)
    list_images_difficult_situations_2 = new_list_images_difficult_situations_2
    images_paths_difficult_situations_2 = sorted(list_images_difficult_situations_2,
                                                 key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_2 = pandas.read_csv(difficult_situations_2_name_file)
    array_annotations_difficult_situations_2 = parse_csv(array_annotations_difficult_situations_2)

    images_difficult_situations_2 = get_images(images_paths_difficult_situations_2, 'cropped', img_shape)
    images_difficult_situations_2, array_annotations_difficult_situations_2 = flip_images(images_difficult_situations_2,
                                                                                          array_annotations_difficult_situations_2)
    print(len(images_difficult_situations_2))
    print(type(images_difficult_situations_2))
    print(len(array_annotations_difficult_situations_2))
    images_difficult_situations_2, array_annotations_difficult_situations_2 = add_extreme_data(
        images_difficult_situations_2, array_annotations_difficult_situations_2)
    print(len(images_difficult_situations_2))
    print(type(images_difficult_situations_2))
    print(len(array_annotations_difficult_situations_2))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_difficult_situations_2 = normalized_annotations

    print(len(images_difficult_situations_2))
    print(type(images_difficult_situations_2))
    print(len(array_annotations_difficult_situations_2))

    ########################################################################################################################### 22 ###########################################################################################################################
    print('---- difficult_situations_01_04_2022/many_curves_3 ----')
    difficult_situations_3_name_file = path_to_data + 'difficult_situations_01_04_2022/many_curves_3/data.csv'
    difficult_situations_3_file = open(difficult_situations_3_name_file, 'r')
    data_difficult_situations_3 = difficult_situations_3_file.read()
    difficult_situations_3_file.close()

    array_annotations_difficult_situations_3 = []
    DIR_difficult_situations_3_images = path_to_data + 'difficult_situations_01_04_2022/many_curves_3/'
    list_images_difficult_situations_3 = glob.glob(DIR_difficult_situations_3_images + '*')
    new_list_images_difficult_situations_3 = []
    for image in list_images_difficult_situations_3:
        if image != path_to_data + 'difficult_situations_01_04_2022/many_curves_3/data.csv':
            new_list_images_difficult_situations_3.append(image)
    list_images_difficult_situations_3 = new_list_images_difficult_situations_3
    images_paths_difficult_situations_3 = sorted(list_images_difficult_situations_3,
                                                 key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_3 = pandas.read_csv(difficult_situations_3_name_file)
    array_annotations_difficult_situations_3 = parse_csv(array_annotations_difficult_situations_3)

    images_difficult_situations_3 = get_images(images_paths_difficult_situations_3, 'cropped', img_shape)
    images_difficult_situations_3, array_annotations_difficult_situations_3 = flip_images(images_difficult_situations_3,
                                                                                          array_annotations_difficult_situations_3)
    print(len(images_difficult_situations_3))
    print(type(images_difficult_situations_3))
    print(len(array_annotations_difficult_situations_3))
    images_difficult_situations_3, array_annotations_difficult_situations_3 = add_extreme_data(
        images_difficult_situations_3, array_annotations_difficult_situations_3)
    print(len(images_difficult_situations_3))
    print(type(images_difficult_situations_3))
    print(len(array_annotations_difficult_situations_3))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_difficult_situations_3 = normalized_annotations

    print(len(images_difficult_situations_3))
    print(type(images_difficult_situations_3))
    print(len(array_annotations_difficult_situations_3))

    ########################################################################################################################### 22 ###########################################################################################################################
    print('---- difficult_situations_01_04_2022/many_curves_4 ----')
    difficult_situations_4_name_file = path_to_data + 'difficult_situations_01_04_2022/many_curves_4/data.csv'
    difficult_situations_4_file = open(difficult_situations_4_name_file, 'r')
    data_difficult_situations_4 = difficult_situations_4_file.read()
    difficult_situations_4_file.close()

    array_annotations_difficult_situations_4 = []
    DIR_difficult_situations_4_images = path_to_data + 'difficult_situations_01_04_2022/many_curves_4/'
    list_images_difficult_situations_4 = glob.glob(DIR_difficult_situations_4_images + '*')
    new_list_images_difficult_situations_4 = []
    for image in list_images_difficult_situations_4:
        if image != path_to_data + 'difficult_situations_01_04_2022/many_curves_4/data.csv':
            new_list_images_difficult_situations_4.append(image)
    list_images_difficult_situations_4 = new_list_images_difficult_situations_4
    images_paths_difficult_situations_4 = sorted(list_images_difficult_situations_4,
                                                 key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_4 = pandas.read_csv(difficult_situations_4_name_file)
    array_annotations_difficult_situations_4 = parse_csv(array_annotations_difficult_situations_4)

    images_difficult_situations_4 = get_images(images_paths_difficult_situations_4, 'cropped', img_shape)
    images_difficult_situations_4, array_annotations_difficult_situations_4 = flip_images(images_difficult_situations_4,
                                                                                          array_annotations_difficult_situations_4)
    print(len(images_difficult_situations_4))
    print(type(images_difficult_situations_4))
    print(len(array_annotations_difficult_situations_4))
    images_difficult_situations_4, array_annotations_difficult_situations_4 = add_extreme_data(
        images_difficult_situations_4, array_annotations_difficult_situations_4)
    print(len(images_difficult_situations_4))
    print(type(images_difficult_situations_4))
    print(len(array_annotations_difficult_situations_4))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_difficult_situations_4 = normalized_annotations

    print(len(images_difficult_situations_4))
    print(type(images_difficult_situations_4))
    print(len(array_annotations_difficult_situations_4))

    ########################################################################################################################### 23 ###########################################################################################################################
    print('---- difficult_situations_01_04_2022/monaco_1 ----')
    difficult_situations_5_name_file = path_to_data + 'difficult_situations_01_04_2022/monaco_1/data.csv'
    difficult_situations_5_file = open(difficult_situations_5_name_file, 'r')
    data_difficult_situations_5 = difficult_situations_5_file.read()
    difficult_situations_5_file.close()

    array_annotations_difficult_situations_5 = []
    DIR_difficult_situations_5_images = path_to_data + 'difficult_situations_01_04_2022/monaco_1/'
    list_images_difficult_situations_5 = glob.glob(DIR_difficult_situations_5_images + '*')
    new_list_images_difficult_situations_5 = []
    for image in list_images_difficult_situations_5:
        if image != path_to_data + 'difficult_situations_01_04_2022/monaco_1/data.csv':
            new_list_images_difficult_situations_5.append(image)
    list_images_difficult_situations_5 = new_list_images_difficult_situations_5
    images_paths_difficult_situations_5 = sorted(list_images_difficult_situations_5,
                                                 key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_5 = pandas.read_csv(difficult_situations_5_name_file)
    array_annotations_difficult_situations_5 = parse_csv(array_annotations_difficult_situations_5)

    images_difficult_situations_5 = get_images(images_paths_difficult_situations_5, 'cropped', img_shape)
    images_difficult_situations_5, array_annotations_difficult_situations_5 = flip_images(images_difficult_situations_5,
                                                                                          array_annotations_difficult_situations_5)
    print(len(images_difficult_situations_5))
    print(type(images_difficult_situations_5))
    print(len(array_annotations_difficult_situations_5))
    images_difficult_situations_5, array_annotations_difficult_situations_5 = add_extreme_data(
        images_difficult_situations_5, array_annotations_difficult_situations_5)
    print(len(images_difficult_situations_5))
    print(type(images_difficult_situations_5))
    print(len(array_annotations_difficult_situations_5))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_difficult_situations_5 = normalized_annotations

    print(len(images_difficult_situations_5))
    print(type(images_difficult_situations_5))
    print(len(array_annotations_difficult_situations_5))

    ########################################################################################################################### 24 ###########################################################################################################################
    print('---- difficult_situations_01_04_2022/monaco_2 ----')
    difficult_situations_6_name_file = path_to_data + 'difficult_situations_01_04_2022/monaco_2/data.csv'
    difficult_situations_6_file = open(difficult_situations_6_name_file, 'r')
    data_difficult_situations_6 = difficult_situations_6_file.read()
    difficult_situations_6_file.close()

    array_annotations_difficult_situations_6 = []
    DIR_difficult_situations_6_images = path_to_data + 'difficult_situations_01_04_2022/monaco_2/'
    list_images_difficult_situations_6 = glob.glob(DIR_difficult_situations_6_images + '*')
    new_list_images_difficult_situations_6 = []
    for image in list_images_difficult_situations_6:
        if image != path_to_data + 'difficult_situations_01_04_2022/monaco_2/data.csv':
            new_list_images_difficult_situations_6.append(image)
    list_images_difficult_situations_6 = new_list_images_difficult_situations_6
    images_paths_difficult_situations_6 = sorted(list_images_difficult_situations_6,
                                                 key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_6 = pandas.read_csv(difficult_situations_6_name_file)
    array_annotations_difficult_situations_6 = parse_csv(array_annotations_difficult_situations_6)

    images_difficult_situations_6 = get_images(images_paths_difficult_situations_6, 'cropped', img_shape)
    images_difficult_situations_6, array_annotations_difficult_situations_6 = flip_images(images_difficult_situations_6,
                                                                                          array_annotations_difficult_situations_6)
    print(len(images_difficult_situations_6))
    print(type(images_difficult_situations_6))
    print(len(array_annotations_difficult_situations_6))
    images_difficult_situations_6, array_annotations_difficult_situations_6 = add_extreme_data(
        images_difficult_situations_6, array_annotations_difficult_situations_6)
    print(len(images_difficult_situations_6))
    print(type(images_difficult_situations_6))
    print(len(array_annotations_difficult_situations_6))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_difficult_situations_6 = normalized_annotations

    print(len(images_difficult_situations_6))
    print(type(images_difficult_situations_6))
    print(len(array_annotations_difficult_situations_6))

    ########################################################################################################################### 24 ###########################################################################################################################
    print('---- difficult_situations_01_04_2022/monaco_3 ----')
    difficult_situations_7_name_file = path_to_data + 'difficult_situations_01_04_2022/monaco_3/data.csv'
    difficult_situations_7_file = open(difficult_situations_7_name_file, 'r')
    data_difficult_situations_7 = difficult_situations_7_file.read()
    difficult_situations_7_file.close()

    array_annotations_difficult_situations_7 = []
    DIR_difficult_situations_7_images = path_to_data + 'difficult_situations_01_04_2022/monaco_3/'
    list_images_difficult_situations_7 = glob.glob(DIR_difficult_situations_7_images + '*')
    new_list_images_difficult_situations_7 = []
    for image in list_images_difficult_situations_7:
        if image != path_to_data + 'difficult_situations_01_04_2022/monaco_3/data.csv':
            new_list_images_difficult_situations_7.append(image)
    list_images_difficult_situations_7 = new_list_images_difficult_situations_7
    images_paths_difficult_situations_7 = sorted(list_images_difficult_situations_7,
                                                 key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_7 = pandas.read_csv(difficult_situations_7_name_file)
    array_annotations_difficult_situations_7 = parse_csv(array_annotations_difficult_situations_7)

    images_difficult_situations_7 = get_images(images_paths_difficult_situations_7, 'cropped', img_shape)
    images_difficult_situations_7, array_annotations_difficult_situations_7 = flip_images(images_difficult_situations_7,
                                                                                          array_annotations_difficult_situations_7)
    print(len(images_difficult_situations_7))
    print(type(images_difficult_situations_7))
    print(len(array_annotations_difficult_situations_7))
    images_difficult_situations_7, array_annotations_difficult_situations_7 = add_extreme_data(
        images_difficult_situations_7, array_annotations_difficult_situations_7)
    print(len(images_difficult_situations_7))
    print(type(images_difficult_situations_7))
    print(len(array_annotations_difficult_situations_7))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_difficult_situations_7 = normalized_annotations

    print(len(images_difficult_situations_7))
    print(type(images_difficult_situations_7))
    print(len(array_annotations_difficult_situations_7))

    ########################################################################################################################### 24 ###########################################################################################################################
    print('---- difficult_situations_01_04_2022/monaco_4 ----')
    difficult_situations_8_name_file = path_to_data + 'difficult_situations_01_04_2022/monaco_4/data.csv'
    difficult_situations_8_file = open(difficult_situations_8_name_file, 'r')
    data_difficult_situations_8 = difficult_situations_8_file.read()
    difficult_situations_8_file.close()

    array_annotations_difficult_situations_8 = []
    DIR_difficult_situations_8_images = path_to_data + 'difficult_situations_01_04_2022/monaco_4/'
    list_images_difficult_situations_8 = glob.glob(DIR_difficult_situations_8_images + '*')
    new_list_images_difficult_situations_8 = []
    for image in list_images_difficult_situations_8:
        if image != path_to_data + 'difficult_situations_01_04_2022/monaco_4/data.csv':
            new_list_images_difficult_situations_8.append(image)
    list_images_difficult_situations_8 = new_list_images_difficult_situations_8
    images_paths_difficult_situations_8 = sorted(list_images_difficult_situations_8,
                                                 key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_8 = pandas.read_csv(difficult_situations_8_name_file)
    array_annotations_difficult_situations_8 = parse_csv(array_annotations_difficult_situations_8)

    images_difficult_situations_8 = get_images(images_paths_difficult_situations_8, 'cropped', img_shape)
    images_difficult_situations_8, array_annotations_difficult_situations_8 = flip_images(images_difficult_situations_8,
                                                                                          array_annotations_difficult_situations_8)
    print(len(images_difficult_situations_8))
    print(type(images_difficult_situations_8))
    print(len(array_annotations_difficult_situations_8))
    images_difficult_situations_8, array_annotations_difficult_situations_8 = add_extreme_data(
        images_difficult_situations_8, array_annotations_difficult_situations_8)
    print(len(images_difficult_situations_8))
    print(type(images_difficult_situations_8))
    print(len(array_annotations_difficult_situations_8))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_difficult_situations_8 = normalized_annotations

    print(len(images_difficult_situations_8))
    print(type(images_difficult_situations_8))
    print(len(array_annotations_difficult_situations_8))

    ########################################################################################################################### 24 ###########################################################################################################################
    print('---- difficult_situations_01_04_2022/monaco_6 ----')
    difficult_situations_10_name_file = path_to_data + 'difficult_situations_01_04_2022/monaco_6/data.csv'
    difficult_situations_10_file = open(difficult_situations_10_name_file, 'r')
    data_difficult_situations_10 = difficult_situations_10_file.read()
    difficult_situations_10_file.close()

    array_annotations_difficult_situations_10 = []
    DIR_difficult_situations_10_images = path_to_data + 'difficult_situations_01_04_2022/monaco_6/'
    list_images_difficult_situations_10 = glob.glob(DIR_difficult_situations_10_images + '*')
    new_list_images_difficult_situations_10 = []
    for image in list_images_difficult_situations_10:
        if image != path_to_data + 'difficult_situations_01_04_2022/monaco_6/data.csv':
            new_list_images_difficult_situations_10.append(image)
    list_images_difficult_situations_10 = new_list_images_difficult_situations_10
    images_paths_difficult_situations_10 = sorted(list_images_difficult_situations_10,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_10 = pandas.read_csv(difficult_situations_10_name_file)
    array_annotations_difficult_situations_10 = parse_csv(array_annotations_difficult_situations_10)

    images_difficult_situations_10 = get_images(images_paths_difficult_situations_10, 'cropped', img_shape)
    images_difficult_situations_10, array_annotations_difficult_situations_10 = flip_images(
        images_difficult_situations_10, array_annotations_difficult_situations_10)
    print(len(images_difficult_situations_10))
    print(type(images_difficult_situations_10))
    print(len(array_annotations_difficult_situations_10))
    images_difficult_situations_10, array_annotations_difficult_situations_10 = add_extreme_data(
        images_difficult_situations_10, array_annotations_difficult_situations_10)
    print(len(images_difficult_situations_10))
    print(type(images_difficult_situations_10))
    print(len(array_annotations_difficult_situations_10))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_difficult_situations_10 = normalized_annotations

    print(len(images_difficult_situations_10))
    print(type(images_difficult_situations_10))
    print(len(array_annotations_difficult_situations_10))

    ########################################################################################################################### 24 ###########################################################################################################################
    print('---- difficult_situations_01_04_2022/monaco_7 ----')
    difficult_situations_11_name_file = path_to_data + 'difficult_situations_01_04_2022/monaco_7/data.csv'
    difficult_situations_11_file = open(difficult_situations_11_name_file, 'r')
    data_difficult_situations_11 = difficult_situations_11_file.read()
    difficult_situations_11_file.close()

    array_annotations_difficult_situations_11 = []
    DIR_difficult_situations_11_images = path_to_data + 'difficult_situations_01_04_2022/monaco_7/'
    list_images_difficult_situations_11 = glob.glob(DIR_difficult_situations_11_images + '*')
    new_list_images_difficult_situations_11 = []
    for image in list_images_difficult_situations_11:
        if image != path_to_data + 'difficult_situations_01_04_2022/monaco_7/data.csv':
            new_list_images_difficult_situations_11.append(image)
    list_images_difficult_situations_11 = new_list_images_difficult_situations_11
    images_paths_difficult_situations_11 = sorted(list_images_difficult_situations_11,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_11 = pandas.read_csv(difficult_situations_11_name_file)
    array_annotations_difficult_situations_11 = parse_csv(array_annotations_difficult_situations_11)

    images_difficult_situations_11 = get_images(images_paths_difficult_situations_11, 'cropped', img_shape)
    images_difficult_situations_11, array_annotations_difficult_situations_11 = flip_images(
        images_difficult_situations_11, array_annotations_difficult_situations_11)
    print(len(images_difficult_situations_11))
    print(type(images_difficult_situations_11))
    print(len(array_annotations_difficult_situations_11))
    images_difficult_situations_11, array_annotations_difficult_situations_11 = add_extreme_data(
        images_difficult_situations_11, array_annotations_difficult_situations_11)
    print(len(images_difficult_situations_11))
    print(type(images_difficult_situations_11))
    print(len(array_annotations_difficult_situations_11))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_difficult_situations_11 = normalized_annotations

    print(len(images_difficult_situations_11))
    print(type(images_difficult_situations_11))
    print(len(array_annotations_difficult_situations_11))

    ########################################################################################################################### 25 ###########################################################################################################################
    print('---- difficult_situations_01_04_2022/nurburgring_1 ----')
    difficult_situations_12_name_file = path_to_data + 'difficult_situations_01_04_2022/nurburgring_1/data.csv'
    difficult_situations_12_file = open(difficult_situations_12_name_file, 'r')
    data_difficult_situations_12 = difficult_situations_12_file.read()
    difficult_situations_12_file.close()

    array_annotations_difficult_situations_12 = []
    DIR_difficult_situations_12_images = path_to_data + 'difficult_situations_01_04_2022/nurburgring_1/'
    list_images_difficult_situations_12 = glob.glob(DIR_difficult_situations_12_images + '*')
    new_list_images_difficult_situations_12 = []
    for image in list_images_difficult_situations_12:
        if image != path_to_data + 'difficult_situations_01_04_2022/nurburgring_1/data.csv':
            new_list_images_difficult_situations_12.append(image)
    list_images_difficult_situations_12 = new_list_images_difficult_situations_12
    images_paths_difficult_situations_12 = sorted(list_images_difficult_situations_12,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_12 = pandas.read_csv(difficult_situations_12_name_file)
    array_annotations_difficult_situations_12 = parse_csv(array_annotations_difficult_situations_12)

    images_difficult_situations_12 = get_images(images_paths_difficult_situations_12, 'cropped', img_shape)
    images_difficult_situations_12, array_annotations_difficult_situations_12 = flip_images(
        images_difficult_situations_12, array_annotations_difficult_situations_12)
    print(len(images_difficult_situations_12))
    print(type(images_difficult_situations_12))
    print(len(array_annotations_difficult_situations_12))
    images_difficult_situations_12, array_annotations_difficult_situations_12 = add_extreme_data(
        images_difficult_situations_12, array_annotations_difficult_situations_12)
    print(len(images_difficult_situations_12))
    print(type(images_difficult_situations_12))
    print(len(array_annotations_difficult_situations_12))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_difficult_situations_12 = normalized_annotations

    print(len(images_difficult_situations_12))
    print(type(images_difficult_situations_12))
    print(len(array_annotations_difficult_situations_12))

    ########################################################################################################################### 26 ###########################################################################################################################
    print('---- difficult_situations_01_04_2022/nurburgring_2 ----')
    difficult_situations_13_name_file = path_to_data + 'difficult_situations_01_04_2022/nurburgring_2/data.csv'
    difficult_situations_13_file = open(difficult_situations_13_name_file, 'r')
    data_difficult_situations_13 = difficult_situations_13_file.read()
    difficult_situations_13_file.close()

    array_annotations_difficult_situations_13 = []
    DIR_difficult_situations_13_images = path_to_data + 'difficult_situations_01_04_2022/nurburgring_2/'
    list_images_difficult_situations_13 = glob.glob(DIR_difficult_situations_13_images + '*')
    new_list_images_difficult_situations_13 = []
    for image in list_images_difficult_situations_13:
        if image != path_to_data + 'difficult_situations_01_04_2022/nurburgring_2/data.csv':
            new_list_images_difficult_situations_13.append(image)
    list_images_difficult_situations_13 = new_list_images_difficult_situations_13
    images_paths_difficult_situations_13 = sorted(list_images_difficult_situations_13,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_13 = pandas.read_csv(difficult_situations_13_name_file)
    array_annotations_difficult_situations_13 = parse_csv(array_annotations_difficult_situations_13)

    images_difficult_situations_13 = get_images(images_paths_difficult_situations_13, 'cropped', img_shape)
    images_difficult_situations_13, array_annotations_difficult_situations_13 = flip_images(
        images_difficult_situations_13, array_annotations_difficult_situations_13)
    print(len(images_difficult_situations_13))
    print(type(images_difficult_situations_13))
    print(len(array_annotations_difficult_situations_13))
    images_difficult_situations_13, array_annotations_difficult_situations_13 = add_extreme_data(
        images_difficult_situations_13, array_annotations_difficult_situations_13)
    print(len(images_difficult_situations_13))
    print(type(images_difficult_situations_13))
    print(len(array_annotations_difficult_situations_13))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_difficult_situations_13 = normalized_annotations

    print(len(images_difficult_situations_13))
    print(type(images_difficult_situations_13))
    print(len(array_annotations_difficult_situations_13))

    ########################################################################################################################### 26 ###########################################################################################################################
    print('---- difficult_situations_01_04_2022/nurburgring_3 ----')
    difficult_situations_14_name_file = path_to_data + 'difficult_situations_01_04_2022/nurburgring_3/data.csv'
    difficult_situations_14_file = open(difficult_situations_14_name_file, 'r')
    data_difficult_situations_14 = difficult_situations_14_file.read()
    difficult_situations_14_file.close()

    array_annotations_difficult_situations_14 = []
    DIR_difficult_situations_14_images = path_to_data + 'difficult_situations_01_04_2022/nurburgring_3/'
    list_images_difficult_situations_14 = glob.glob(DIR_difficult_situations_14_images + '*')
    new_list_images_difficult_situations_14 = []
    for image in list_images_difficult_situations_14:
        if image != path_to_data + 'difficult_situations_01_04_2022/nurburgring_3/data.csv':
            new_list_images_difficult_situations_14.append(image)
    list_images_difficult_situations_14 = new_list_images_difficult_situations_14
    images_paths_difficult_situations_14 = sorted(list_images_difficult_situations_14,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_14 = pandas.read_csv(difficult_situations_14_name_file)
    array_annotations_difficult_situations_14 = parse_csv(array_annotations_difficult_situations_14)

    images_difficult_situations_14 = get_images(images_paths_difficult_situations_14, 'cropped', img_shape)
    images_difficult_situations_14, array_annotations_difficult_situations_14 = flip_images(
        images_difficult_situations_14, array_annotations_difficult_situations_14)
    print(len(images_difficult_situations_14))
    print(type(images_difficult_situations_14))
    print(len(array_annotations_difficult_situations_14))
    images_difficult_situations_14, array_annotations_difficult_situations_14 = add_extreme_data(
        images_difficult_situations_14, array_annotations_difficult_situations_14)
    print(len(images_difficult_situations_14))
    print(type(images_difficult_situations_14))
    print(len(array_annotations_difficult_situations_14))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_difficult_situations_14 = normalized_annotations

    print(len(images_difficult_situations_14))
    print(type(images_difficult_situations_14))
    print(len(array_annotations_difficult_situations_14))

    ########################################################################################################################### 26 ###########################################################################################################################
    print('---- difficult_situations_01_04_2022/nurburgring_4 ----')
    difficult_situations_15_name_file = path_to_data + 'difficult_situations_01_04_2022/nurburgring_4/data.csv'
    difficult_situations_15_file = open(difficult_situations_15_name_file, 'r')
    data_difficult_situations_15 = difficult_situations_15_file.read()
    difficult_situations_15_file.close()

    array_annotations_difficult_situations_15 = []
    DIR_difficult_situations_15_images = path_to_data + 'difficult_situations_01_04_2022/nurburgring_4/'
    list_images_difficult_situations_15 = glob.glob(DIR_difficult_situations_15_images + '*')
    new_list_images_difficult_situations_15 = []
    for image in list_images_difficult_situations_15:
        if image != path_to_data + 'difficult_situations_01_04_2022/nurburgring_4/data.csv':
            new_list_images_difficult_situations_15.append(image)
    list_images_difficult_situations_15 = new_list_images_difficult_situations_15
    images_paths_difficult_situations_15 = sorted(list_images_difficult_situations_15,
                                                  key=lambda x: int(x.split('/')[7].split('.png')[0]))

    array_annotations_difficult_situations_15 = pandas.read_csv(difficult_situations_15_name_file)
    array_annotations_difficult_situations_15 = parse_csv(array_annotations_difficult_situations_15)

    images_difficult_situations_15 = get_images(images_paths_difficult_situations_15, 'cropped', img_shape)
    images_difficult_situations_15, array_annotations_difficult_situations_15 = flip_images(
        images_difficult_situations_15, array_annotations_difficult_situations_15)
    print(len(images_difficult_situations_15))
    print(type(images_difficult_situations_15))
    print(len(array_annotations_difficult_situations_15))
    images_difficult_situations_15, array_annotations_difficult_situations_15 = add_extreme_data(
        images_difficult_situations_15, array_annotations_difficult_situations_15)
    print(len(images_difficult_situations_15))
    print(type(images_difficult_situations_15))
    print(len(array_annotations_difficult_situations_15))

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

    # normalized_X = np.interp(array_annotations_v, (-0.6, 13), (0, 1))
    normalized_X = np.interp(array_annotations_v, (6.5, 24), (0, 1))
    # normalized_Y = np.interp(array_annotations_w, (-3, 3), (0, 1))
    normalized_Y = np.interp(array_annotations_w, (-7.1, 7.1), (0, 1))
    # normalized_X = normalize_v(array_annotations_v)
    # normalized_Y = normalize(array_annotations_w)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    array_annotations_difficult_situations_15 = normalized_annotations

    print(len(images_difficult_situations_15))
    print(type(images_difficult_situations_15))
    print(len(array_annotations_difficult_situations_15))

    array_imgs = images_many_curves_1 + \
                 images_nurburgring_1 + \
                 images_monaco_1 + \
                 images_extended_simple_1 + \
                 images_only_curves_1 + images_only_curves_2 + images_only_curves_3 + images_only_curves_4 + images_only_curves_5 + images_only_curves_6 + \
                 images_only_curves_7 + images_only_curves_8 + images_only_curves_9 + images_only_curves_10 + images_only_curves_11 + images_only_curves_12 + \
                 images_difficult_situations_1 + images_difficult_situations_2 + images_difficult_situations_3 + \
                 images_difficult_situations_4 + images_difficult_situations_5 + images_difficult_situations_6 + \
                 images_difficult_situations_7 + images_difficult_situations_8 + \
                 images_difficult_situations_10 + images_difficult_situations_11 + images_difficult_situations_12 + \
                 images_difficult_situations_13 + images_difficult_situations_14 + images_difficult_situations_15

    array_annotations = array_annotations_many_curves_1 + \
                        array_annotations_nurburgring_1 + \
                        array_annotations_monaco_1 + \
                        array_annotations_extended_simple_1 + \
                        array_annotations_only_curves_1 + array_annotations_only_curves_2 + array_annotations_only_curves_3 + array_annotations_only_curves_4 + array_annotations_only_curves_5 + array_annotations_only_curves_6 + \
                        array_annotations_only_curves_7 + array_annotations_only_curves_8 + array_annotations_only_curves_9 + array_annotations_only_curves_10 + array_annotations_only_curves_11 + array_annotations_only_curves_12 + \
                        array_annotations_difficult_situations_1 + array_annotations_difficult_situations_2 + array_annotations_difficult_situations_3 + \
                        array_annotations_difficult_situations_4 + array_annotations_difficult_situations_5 + array_annotations_difficult_situations_6 + \
                        array_annotations_difficult_situations_7 + array_annotations_difficult_situations_8 + \
                        array_annotations_difficult_situations_10 + array_annotations_difficult_situations_11 + array_annotations_difficult_situations_12 + \
                        array_annotations_difficult_situations_13 + array_annotations_difficult_situations_14 + array_annotations_difficult_situations_15

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
    array_imgs, array_annotations = get_images_and_annotations(path_to_data, type_image, img_shape)
    if data_type == 'extreme':
        array_imgs, array_annotations = add_extreme_data(array_imgs, array_annotations)
    images_train, annotations_train, images_validation, annotations_validation = separate_dataset_into_train_validation(
        array_imgs, array_annotations)

    return images_train, annotations_train, images_validation, annotations_validation
