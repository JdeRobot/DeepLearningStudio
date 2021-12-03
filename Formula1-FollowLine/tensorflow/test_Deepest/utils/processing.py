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
    images_paths_complete = sorted(list_images_complete, key=lambda x: int(x.split('/')[6].split('.png')[0]))
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
    images_paths_curves = sorted(list_images_curves, key=lambda x: int(x.split('/')[6].split('.png')[0]))
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


def separate_dataset_into_sequences(array_imgs, array_annotations):
    # SEPARATE DATASET INTO SEQUENCES TO FIT BATCH SIZES
    # 1
    array_1_img = []
    array_1_ann = []
    # 2
    array_2_img = []
    array_2_ann = []
    # 3
    array_3_img = []
    array_3_ann = []
    # 4
    array_4_img = []
    array_4_ann = []
    # 5
    array_5_img = []
    array_5_ann = []
    # 6
    array_6_img = []
    array_6_ann = []
    # 7
    array_7_img = []
    array_7_ann = []
    # 8
    array_8_img = []
    array_8_ann = []
    # 9
    array_9_img = []
    array_9_ann = []
    # 10
    array_10_img = []
    array_10_ann = []
    # 11
    array_11_img = []
    array_11_ann = []
    # 12
    array_12_img = []
    array_12_ann = []
    # 13
    array_13_img = []
    array_13_ann = []
    # 14
    array_14_img = []
    array_14_ann = []
    # 15
    array_15_img = []
    array_15_ann = []
    # 16
    array_16_img = []
    array_16_ann = []
    # 17
    array_17_img = []
    array_17_ann = []
    # 18
    array_18_img = []
    array_18_ann = []
    # 19
    array_19_img = []
    array_19_ann = []
    # 20
    array_20_img = []
    array_20_ann = []
    # 21
    array_21_img = []
    array_21_ann = []
    # 22
    array_22_img = []
    array_22_ann = []
    # 23
    array_23_img = []
    array_23_ann = []
    # 24
    array_24_img = []
    array_24_ann = []
    # 25
    array_25_img = []
    array_25_ann = []
    # 26
    array_26_img = []
    array_26_ann = []
    # 27
    array_27_img = []
    array_27_ann = []
    # 28
    array_28_img = []
    array_28_ann = []
    # 29
    array_29_img = []
    array_29_ann = []
    # 30
    array_30_img = []
    array_30_ann = []
    # 31
    array_31_img = []
    array_31_ann = []
    # 32
    array_32_img = []
    array_32_ann = []
    # 33
    array_33_img = []
    array_33_ann = []

    for i in range(0, 3700):
        array_1_img.append(array_imgs[i])
        array_1_ann.append(array_annotations[i])
    for i in range(3745, 5045):
        array_2_img.append(array_imgs[i])
        array_2_ann.append(array_annotations[i])
    for i in range(5067, 9717):
        array_3_img.append(array_imgs[i])
        array_3_ann.append(array_annotations[i])
    for i in range(9721, 10371):
        array_4_img.append(array_imgs[i])
        array_4_ann.append(array_annotations[i])
    for i in range(10388, 10688):
        array_5_img.append(array_imgs[i])
        array_5_ann.append(array_annotations[i])
    for i in range(10696, 11246):
        array_6_img.append(array_imgs[i])
        array_6_ann.append(array_annotations[i])
    for i in range(11284, 11334):
        array_7_img.append(array_imgs[i])
        array_7_ann.append(array_annotations[i])
    for i in range(11355, 11455):
        array_8_img.append(array_imgs[i])
        array_8_ann.append(array_annotations[i])
    for i in range(11493, 11943):
        array_9_img.append(array_imgs[i])
        array_9_ann.append(array_annotations[i])
    for i in range(11981, 12581):
        array_10_img.append(array_imgs[i])
        array_10_ann.append(array_annotations[i])
    for i in range(12619, 13219):
        array_11_img.append(array_imgs[i])
        array_11_ann.append(array_annotations[i])
    for i in range(13232, 14082):
        array_12_img.append(array_imgs[i])
        array_12_ann.append(array_annotations[i])
    for i in range(14108, 15758):
        array_13_img.append(array_imgs[i])
        array_13_ann.append(array_annotations[i])
    # for i in range(15791, 17296):
    for i in range(15791, 17291):
        array_14_img.append(array_imgs[i])
        array_14_ann.append(array_annotations[i])
    for i in range(17341, 20491):
        array_15_img.append(array_imgs[i])
        array_15_ann.append(array_annotations[i])
    for i in range(20498, 22598):
        array_16_img.append(array_imgs[i])
        array_16_ann.append(array_annotations[i])
    for i in range(22609, 26309):
        array_17_img.append(array_imgs[i])
        array_17_ann.append(array_annotations[i])
    for i in range(26354, 27654):
        array_18_img.append(array_imgs[i])
        array_18_ann.append(array_annotations[i])
    for i in range(27676, 32326):
        array_19_img.append(array_imgs[i])
        array_19_ann.append(array_annotations[i])
    # for i in range(32330, 32960):
    for i in range(32330, 32930):
        array_20_img.append(array_imgs[i])
        array_20_ann.append(array_annotations[i])
    for i in range(32997, 33297):
        array_21_img.append(array_imgs[i])
        array_21_ann.append(array_annotations[i])
    for i in range(33305, 33855):
        array_22_img.append(array_imgs[i])
        array_22_ann.append(array_annotations[i])
    for i in range(33893, 33943):
        array_23_img.append(array_imgs[i])
        array_23_ann.append(array_annotations[i])
    for i in range(33964, 34064):
        array_24_img.append(array_imgs[i])
        array_24_ann.append(array_annotations[i])
    for i in range(34102, 34552):
        array_25_img.append(array_imgs[i])
        array_25_ann.append(array_annotations[i])
    for i in range(34590, 35190):
        array_26_img.append(array_imgs[i])
        array_26_ann.append(array_annotations[i])
    for i in range(35228, 35828):
        array_27_img.append(array_imgs[i])
        array_27_ann.append(array_annotations[i])
    for i in range(35841, 36691):
        array_28_img.append(array_imgs[i])
        array_28_ann.append(array_annotations[i])
    for i in range(36717, 38367):
        array_29_img.append(array_imgs[i])
        array_29_ann.append(array_annotations[i])
    for i in range(38400, 39300):
        array_30_img.append(array_imgs[i])
        array_30_ann.append(array_annotations[i])
    for i in range(39405, 39905):
        array_31_img.append(array_imgs[i])
        array_31_ann.append(array_annotations[i])
    for i in range(39950, 43100):
        array_32_img.append(array_imgs[i])
        array_32_ann.append(array_annotations[i])
    # for i in range(43107, 45202):
    for i in range(43107, 45157):
        array_33_img.append(array_imgs[i])
        array_33_ann.append(array_annotations[i])

    '''
    [21, 0, 12, 2, 10, 26, 25, 7, 29, 15, 30, 28, 27, 3, 4, 24, 32, 6, 19, 8, 20, 14, 17, 16, 31, 13, 9, 1, 5, 18, 22, 11, 23]
    '''

    array_x = []
    array_x.append(array_1_img)
    array_x.append(array_2_img)
    array_x.append(array_3_img)
    array_x.append(array_4_img)
    array_x.append(array_5_img)
    array_x.append(array_6_img)
    array_x.append(array_7_img)
    array_x.append(array_8_img)
    array_x.append(array_9_img)
    array_x.append(array_10_img)
    array_x.append(array_11_img)
    array_x.append(array_12_img)
    # array_x.append(array_13_img)
    # array_x.append(array_14_img)
    array_x.append(array_13_img)
    array_x.append(array_14_img)
    array_x.append(array_15_img)
    array_x.append(array_16_img)
    array_x.append(array_17_img)
    array_x.append(array_18_img)
    array_x.append(array_19_img)
    array_x.append(array_20_img)
    array_x.append(array_21_img)
    array_x.append(array_22_img)
    array_x.append(array_23_img)
    array_x.append(array_24_img)
    array_x.append(array_25_img)
    array_x.append(array_26_img)
    array_x.append(array_27_img)
    array_x.append(array_28_img)
    # array_x.append(array_29_img)
    # array_x.append(array_30_img)
    # array_x.append(array_31_img)
    array_x.append(array_29_img)
    array_x.append(array_30_img)
    array_x.append(array_31_img)
    array_x.append(array_32_img)
    array_x.append(array_33_img)

    array_y = []
    array_y.append(array_1_ann)
    array_y.append(array_2_ann)
    array_y.append(array_3_ann)
    array_y.append(array_4_ann)
    array_y.append(array_5_ann)
    array_y.append(array_6_ann)
    array_y.append(array_7_ann)
    array_y.append(array_8_ann)
    array_y.append(array_9_ann)
    array_y.append(array_10_ann)
    array_y.append(array_11_ann)
    array_y.append(array_12_ann)
    # array_y.append(array_13_ann)
    # array_y.append(array_14_ann)
    array_y.append(array_13_ann)
    array_y.append(array_14_ann)
    array_y.append(array_15_ann)
    array_y.append(array_16_ann)
    array_y.append(array_17_ann)
    array_y.append(array_18_ann)
    array_y.append(array_19_ann)
    array_y.append(array_20_ann)
    array_y.append(array_21_ann)
    array_y.append(array_22_ann)
    array_y.append(array_23_ann)
    array_y.append(array_24_ann)
    array_y.append(array_25_ann)
    array_y.append(array_26_ann)
    array_y.append(array_27_ann)
    array_y.append(array_28_ann)
    # array_y.append(array_29_ann)
    # array_y.append(array_30_ann)
    # array_y.append(array_31_ann)
    array_y.append(array_29_ann)
    array_y.append(array_30_ann)
    array_y.append(array_31_ann)
    array_y.append(array_32_ann)
    array_y.append(array_33_ann)

    print(len(array_x))
    print(len(array_y))

    return array_x, array_y


def add_extreme_sequences(array_x, array_y):
    '''
        Look for extreme 50 frames sequences inside every big-sequence
    '''
    new_array_x = []
    new_array_y = []
    for x, big_sequence_anns in enumerate(array_y):
        new_big_sequence_imgs = []
        new_big_sequence_anns = []
        for y in range(0, int(len(big_sequence_anns) / 50)):
            sequences_imgs = array_x[x][y * 50:(y * 50) + 50]
            sequences_anns = array_y[x][y * 50:(y * 50) + 50]
            new_big_sequence_imgs += sequences_imgs
            new_big_sequence_anns += sequences_anns
            for seq_number, seq in enumerate(sequences_anns):
                if seq[1] >= 0.7 or seq[1] <= 0.3:
                    if seq[1] >= 0.9 or seq[1] <= 0.1:
                        for i in range(0, 5):
                            new_big_sequence_imgs += sequences_imgs
                            new_big_sequence_anns += sequences_anns
                    elif seq[1] >= 0.8 or seq[1] <= 0.2:
                        for i in range(0, 3):
                            new_big_sequence_imgs += sequences_imgs
                            new_big_sequence_anns += sequences_anns
                    else:
                        for i in range(0, 2):
                            new_big_sequence_imgs += sequences_imgs
                            new_big_sequence_anns += sequences_anns
                if seq[0] <= 0.2:
                    for i in range(0, 5):
                        new_big_sequence_imgs += sequences_imgs
                        new_big_sequence_anns += sequences_anns
        new_array_x.append(new_big_sequence_imgs)
        new_array_y.append(new_big_sequence_anns)

    print('------')

    print(len(new_array_y[0]))

    shown_array_imgs = []
    shown_array_annotations = []

    random_sort = random.sample(range(0, 33), 33)
    print(len(random_sort))
    print(random_sort)

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
                                                                                                  shuffle=False)

    # Adapt the data
    images_train = np.stack(images_train, axis=0)
    annotations_train = np.stack(annotations_train, axis=0)
    images_validation = np.stack(images_validation, axis=0)
    annotations_validation = np.stack(annotations_validation, axis=0)

    print(annotations_train[0])
    print(annotations_train.shape)
    print(annotations_validation[0])
    print(annotations_validation.shape)
    print(images_train.shape)
    print(images_validation.shape)

    return images_train, annotations_train, images_validation, annotations_validation


def process_dataset(path_to_data, type_image, data_type, img_shape):
    array_imgs, array_annotations = get_images_and_annotations(path_to_data, type_image, img_shape)
    array_x, array_y = separate_dataset_into_sequences(array_imgs, array_annotations)
    if data_type == 'extreme':
        array_x, array_y = add_extreme_sequences(array_x, array_y)
    images_train, annotations_train, images_validation, annotations_validation = separate_dataset_into_train_validation(array_x, array_y)

    return images_train, annotations_train, images_validation, annotations_validation
