import glob
import cv2
import pandas

import numpy as np

from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split


def get_images(folder_prefix, list_images, image_shape):
    # Read the images
    array_imgs = []
    for name in list_images:
        try:
            img = cv2.imread(folder_prefix + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_shape)
            array_imgs.append(img)
        except Exception as ex:
            print(ex)
            print('ERROR in value')

    return array_imgs


def parse_csv(csv_data):
    array = []
    linear_speeds = csv_data['throttle'].tolist()
    angular_speeds = csv_data['steer'].tolist()
    brakes = csv_data['brake'].tolist()
    images_ids = csv_data['image_id'].tolist()
    velocity = csv_data['velocity'].tolist()
    timestamp = csv_data['timestamp'].tolist()
    for x, linear_speed in enumerate(linear_speeds):
        try:
            array.append((float(linear_speed), float(angular_speeds[x]), float(brakes[x]), float(velocity[x]),
                          float(timestamp[x])))
        except:
            print('ERROR in value')
    return images_ids, array


def get_images_and_annotations(path_to_data, type_image, img_shape, data_type):
    ######################################### 1 #########################################
    carla_dataset_name_file = path_to_data + 'carla_dataset_test_31_10_anticlockwise_town_01_previous_v/dataset.csv'
    array_annotations_carla_dataset_1 = pandas.read_csv(carla_dataset_name_file)
    images_ids, array_annotations_carla_dataset_1 = parse_csv(array_annotations_carla_dataset_1)

    images_carla_dataset_1 = get_images(path_to_data + 'carla_dataset_test_31_10_anticlockwise_town_01_previous_v/',
                                        images_ids, img_shape)

    array_annotations_v = []
    array_annotations_w = []
    array_annotations_b = []
    array_annotations_vel = []
    for annotation in array_annotations_carla_dataset_1:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])
        array_annotations_b.append(annotation[2])
        array_annotations_vel.append(annotation[3])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_Y = np.interp(array_annotations_w, (-1, 1), (0, 1))

    array_annotations_b = np.stack(array_annotations_b, axis=0)
    array_annotations_b = array_annotations_b.reshape(-1, 1)

    normalized_annotations = []
    for i in range(0, len(array_annotations_w)):
        normalized_annotations.append([array_annotations_v.item(i), normalized_Y.item(i), array_annotations_b.item(i)])

    array_annotations_carla_dataset_1 = normalized_annotations

    ######################################### 2 #########################################
    carla_dataset_name_file = path_to_data + 'carla_dataset_test_31_10_clockwise_town_01_previous_v/dataset.csv'
    array_annotations_carla_dataset_2 = pandas.read_csv(carla_dataset_name_file)
    images_ids, array_annotations_carla_dataset_2 = parse_csv(array_annotations_carla_dataset_2)

    images_carla_dataset_2 = get_images(path_to_data + 'carla_dataset_test_31_10_clockwise_town_01_previous_v/',
                                        images_ids, img_shape)

    array_annotations_v = []
    array_annotations_w = []
    array_annotations_b = []
    for annotation in array_annotations_carla_dataset_2:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])
        array_annotations_b.append(annotation[2])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_Y = np.interp(array_annotations_w, (-1, 1), (0, 1))

    array_annotations_b = np.stack(array_annotations_b, axis=0)
    array_annotations_b = array_annotations_b.reshape(-1, 1)

    normalized_annotations = []
    for i in range(0, len(array_annotations_w)):
        normalized_annotations.append([array_annotations_v.item(i), normalized_Y.item(i), array_annotations_b.item(i)])

    array_annotations_carla_dataset_2 = normalized_annotations

    ######################################### 3 #########################################
    carla_dataset_name_file = path_to_data + 'carla_dataset_test_04_11_clockwise_town_01_previous_v_extreme/dataset.csv'
    array_annotations_carla_dataset_3 = pandas.read_csv(carla_dataset_name_file)
    images_ids, array_annotations_carla_dataset_3 = parse_csv(array_annotations_carla_dataset_3)

    images_carla_dataset_3 = get_images(path_to_data + 'carla_dataset_test_04_11_clockwise_town_01_previous_v_extreme/',
                                        images_ids, img_shape)

    array_annotations_v = []
    array_annotations_w = []
    array_annotations_b = []
    for annotation in array_annotations_carla_dataset_3:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])
        array_annotations_b.append(annotation[2])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_Y = np.interp(array_annotations_w, (-1, 1), (0, 1))

    array_annotations_b = np.stack(array_annotations_b, axis=0)
    array_annotations_b = array_annotations_b.reshape(-1, 1)

    normalized_annotations = []
    for i in range(0, len(array_annotations_w)):
        normalized_annotations.append([array_annotations_v.item(i), normalized_Y.item(i), array_annotations_b.item(i)])

    array_annotations_carla_dataset_3 = normalized_annotations

    ######################################### 4 #########################################
    carla_dataset_name_file = path_to_data + 'carla_dataset_test_04_11_anticlockwise_town_03_previous_v/dataset.csv'
    array_annotations_carla_dataset_4 = pandas.read_csv(carla_dataset_name_file)
    images_ids, array_annotations_carla_dataset_4 = parse_csv(array_annotations_carla_dataset_4)

    images_carla_dataset_4 = get_images(path_to_data + 'carla_dataset_test_04_11_anticlockwise_town_03_previous_v/',
                                        images_ids, img_shape)

    array_annotations_v = []
    array_annotations_w = []
    array_annotations_b = []
    for annotation in array_annotations_carla_dataset_4:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])
        array_annotations_b.append(annotation[2])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_Y = np.interp(array_annotations_w, (-1, 1), (0, 1))

    array_annotations_b = np.stack(array_annotations_b, axis=0)
    array_annotations_b = array_annotations_b.reshape(-1, 1)

    normalized_annotations = []
    for i in range(0, len(array_annotations_w)):
        normalized_annotations.append([array_annotations_v.item(i), normalized_Y.item(i), array_annotations_b.item(i)])

    array_annotations_carla_dataset_4 = normalized_annotations

    ######################################### 5 #########################################

    carla_dataset_name_file = path_to_data + 'carla_dataset_test_04_11_clockwise_town_03_previous_v/dataset.csv'
    array_annotations_carla_dataset_5 = pandas.read_csv(carla_dataset_name_file)
    images_ids, array_annotations_carla_dataset_5 = parse_csv(array_annotations_carla_dataset_5)

    images_carla_dataset_5 = get_images(path_to_data + 'carla_dataset_test_04_11_clockwise_town_03_previous_v/',
                                        images_ids, img_shape)

    array_annotations_v = []
    array_annotations_w = []
    array_annotations_b = []
    for annotation in array_annotations_carla_dataset_5:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])
        array_annotations_b.append(annotation[2])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_Y = np.interp(array_annotations_w, (-1, 1), (0, 1))

    array_annotations_b = np.stack(array_annotations_b, axis=0)
    array_annotations_b = array_annotations_b.reshape(-1, 1)

    normalized_annotations = []
    for i in range(0, len(array_annotations_w)):
        normalized_annotations.append([array_annotations_v.item(i), normalized_Y.item(i), array_annotations_b.item(i)])

    array_annotations_carla_dataset_5 = normalized_annotations

    ######################################### 6 #########################################
    carla_dataset_name_file = path_to_data + 'carla_dataset_test_04_11_anticlockwise_town_05_previous_v/dataset.csv'
    array_annotations_carla_dataset_6 = pandas.read_csv(carla_dataset_name_file)
    images_ids, array_annotations_carla_dataset_6 = parse_csv(array_annotations_carla_dataset_6)

    images_carla_dataset_6 = get_images(path_to_data + 'carla_dataset_test_04_11_anticlockwise_town_05_previous_v/',
                                        images_ids, img_shape)

    array_annotations_v = []
    array_annotations_w = []
    array_annotations_b = []
    for annotation in array_annotations_carla_dataset_6:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])
        array_annotations_b.append(annotation[2])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_Y = np.interp(array_annotations_w, (-1, 1), (0, 1))

    array_annotations_b = np.stack(array_annotations_b, axis=0)
    array_annotations_b = array_annotations_b.reshape(-1, 1)

    normalized_annotations = []
    for i in range(0, len(array_annotations_w)):
        normalized_annotations.append([array_annotations_v.item(i), normalized_Y.item(i), array_annotations_b.item(i)])

    array_annotations_carla_dataset_6 = normalized_annotations

    ######################################### 7 #########################################
    carla_dataset_name_file = path_to_data + 'carla_dataset_test_04_11_clockwise_town_05_previous_v/dataset.csv'
    array_annotations_carla_dataset_7 = pandas.read_csv(carla_dataset_name_file)
    images_ids, array_annotations_carla_dataset_7 = parse_csv(array_annotations_carla_dataset_7)

    images_carla_dataset_7 = get_images(path_to_data + 'carla_dataset_test_04_11_clockwise_town_05_previous_v/',
                                        images_ids, img_shape)

    array_annotations_v = []
    array_annotations_w = []
    array_annotations_b = []
    for annotation in array_annotations_carla_dataset_7:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])
        array_annotations_b.append(annotation[2])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_Y = np.interp(array_annotations_w, (-1, 1), (0, 1))

    array_annotations_b = np.stack(array_annotations_b, axis=0)
    array_annotations_b = array_annotations_b.reshape(-1, 1)

    normalized_annotations = []
    for i in range(0, len(array_annotations_w)):
        normalized_annotations.append([array_annotations_v.item(i), normalized_Y.item(i), array_annotations_b.item(i)])

    array_annotations_carla_dataset_7 = normalized_annotations

    ######################################### 8 #########################################
    carla_dataset_name_file = path_to_data + 'carla_dataset_test_04_11_anticlockwise_town_07_previous_v/dataset.csv'
    array_annotations_carla_dataset_8 = pandas.read_csv(carla_dataset_name_file)
    images_ids, array_annotations_carla_dataset_8 = parse_csv(array_annotations_carla_dataset_8)

    images_carla_dataset_8 = get_images(path_to_data + 'carla_dataset_test_04_11_anticlockwise_town_07_previous_v/',
                                        images_ids, img_shape)

    array_annotations_v = []
    array_annotations_w = []
    array_annotations_b = []
    for annotation in array_annotations_carla_dataset_8:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])
        array_annotations_b.append(annotation[2])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_Y = np.interp(array_annotations_w, (-1, 1), (0, 1))

    array_annotations_b = np.stack(array_annotations_b, axis=0)
    array_annotations_b = array_annotations_b.reshape(-1, 1)

    normalized_annotations = []
    for i in range(0, len(array_annotations_w)):
        normalized_annotations.append([array_annotations_v.item(i), normalized_Y.item(i), array_annotations_b.item(i)])

    array_annotations_carla_dataset_8 = normalized_annotations

    ######################################### 9 #########################################
    carla_dataset_name_file = path_to_data + 'carla_dataset_test_04_11_clockwise_town_07_previous_v/dataset.csv'
    array_annotations_carla_dataset_9 = pandas.read_csv(carla_dataset_name_file)
    images_ids, array_annotations_carla_dataset_9 = parse_csv(array_annotations_carla_dataset_9)

    images_carla_dataset_9 = get_images(path_to_data + 'carla_dataset_test_04_11_clockwise_town_07_previous_v/',
                                        images_ids, img_shape)

    array_annotations_v = []
    array_annotations_w = []
    array_annotations_b = []
    for annotation in array_annotations_carla_dataset_9:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])
        array_annotations_b.append(annotation[2])

    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_Y = np.interp(array_annotations_w, (-1, 1), (0, 1))

    array_annotations_b = np.stack(array_annotations_b, axis=0)
    array_annotations_b = array_annotations_b.reshape(-1, 1)

    normalized_annotations = []
    for i in range(0, len(array_annotations_w)):
        normalized_annotations.append([array_annotations_v.item(i), normalized_Y.item(i), array_annotations_b.item(i)])

    array_annotations_carla_dataset_9 = normalized_annotations

    ###########

    new_images_carla_dataset_1 = images_carla_dataset_1[:-33]
    new_array_annotations_carla_dataset_1 = array_annotations_carla_dataset_1[:-33]

    new_images_carla_dataset_2 = images_carla_dataset_2[:-5]
    new_array_annotations_carla_dataset_2 = array_annotations_carla_dataset_2[:-5]

    new_images_carla_dataset_3 = images_carla_dataset_3[:-10]
    new_array_annotations_carla_dataset_3 = array_annotations_carla_dataset_3[:-10]

    new_images_carla_dataset_4 = images_carla_dataset_4[:-34]
    new_array_annotations_carla_dataset_4 = array_annotations_carla_dataset_4[:-34]

    new_images_carla_dataset_5 = images_carla_dataset_5[:-36]
    new_array_annotations_carla_dataset_5 = array_annotations_carla_dataset_5[:-36]

    new_images_carla_dataset_6 = images_carla_dataset_6[:-24]
    new_array_annotations_carla_dataset_6 = array_annotations_carla_dataset_6[:-24]

    new_images_carla_dataset_7 = images_carla_dataset_7[:-43]
    new_array_annotations_carla_dataset_7 = array_annotations_carla_dataset_7[:-43]

    new_images_carla_dataset_8 = images_carla_dataset_8[:-30]
    new_array_annotations_carla_dataset_8 = array_annotations_carla_dataset_8[:-30]

    new_images_carla_dataset_9 = images_carla_dataset_9[:-8]
    new_array_annotations_carla_dataset_9 = array_annotations_carla_dataset_9[:-8]

    array_imgs = [
        new_images_carla_dataset_1, new_images_carla_dataset_2, new_images_carla_dataset_3,
        new_images_carla_dataset_4, new_images_carla_dataset_5, new_images_carla_dataset_6,
        new_images_carla_dataset_7, new_images_carla_dataset_8, new_images_carla_dataset_9
    ]

    array_annotations = [
        new_array_annotations_carla_dataset_1, new_array_annotations_carla_dataset_2,
        new_array_annotations_carla_dataset_3,
        new_array_annotations_carla_dataset_4, new_array_annotations_carla_dataset_5,
        new_array_annotations_carla_dataset_6,
        new_array_annotations_carla_dataset_7, new_array_annotations_carla_dataset_8,
        new_array_annotations_carla_dataset_9
    ]

    return array_imgs, array_annotations


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
                if big_ann[1] >= 0.75 or big_ann[1] <= 0.25:
                    for i in range(0, 40):
                        new_big_imgs.append(big_img)
                        new_big_anns.append(big_ann)
                elif big_ann[1] >= 0.6 or big_ann[1] <= 0.4:
                    for i in range(0, 30):
                        new_big_imgs.append(big_img)
                        new_big_anns.append(big_ann)
                else:
                    for i in range(0, 15):
                        new_big_imgs.append(big_img)
                        new_big_anns.append(big_ann)
            if big_ann[2] >= 0.1:
                if abs(big_ann[2]) >= 0.3:
                    num_iter = 15
                elif abs(big_ann[2]) >= 0.2:
                    num_iter = 5
                else:
                    num_iter = 2
                for j in range(0, num_iter):
                    new_big_imgs.append(big_img)
                    new_big_anns.append(big_ann)
        new_array_x_extreme.append(new_big_imgs)
        new_array_y_extreme.append(new_big_anns)

    new_array_x = new_array_x_extreme
    new_array_y = new_array_y_extreme

    shown_array_imgs = []
    shown_array_annotations = []
    random_sort = random.sample(range(0, len(array_x)), len(array_x))

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
    array_imgs, array_annotations = get_images_and_annotations(path_to_data, type_image, img_shape, data_type)
    array_imgs, array_annotations = separate_dataset_into_sequences(array_imgs, array_annotations)
    if data_type == 'extreme':
        array_imgs, array_annotations = add_extreme_sequences(array_imgs, array_annotations)

    images_train, annotations_train, images_validation, annotations_validation = separate_dataset_into_train_validation(
        array_imgs, array_annotations)

    return images_train, annotations_train, images_validation, annotations_validation
