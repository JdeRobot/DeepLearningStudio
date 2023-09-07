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
            array.append((float(linear_speed), float(angular_speeds[x]), float(brakes[x]), float(velocity[x]), float(timestamp[x])))
        except:
            print('ERROR in value')
    return images_ids, array

def add_extreme_data(images, array_annotations):
    for i in range(0, len(array_annotations)):
        if abs(array_annotations[i][1]) >= 0.1:
            if abs(array_annotations[i][1]) >= 0.3:
                num_iter = 15
            elif abs(array_annotations[i][1]) >= 0.2:
                num_iter = 5
            else:
                num_iter = 2
            for j in range(0, num_iter):
                array_annotations.append(array_annotations[i])
                images.append(images[i])
        if abs(array_annotations[i][2]) >= 0.1:
            if abs(array_annotations[i][2]) >= 0.3:
                num_iter = 15
            elif abs(array_annotations[i][2]) >= 0.2:
                num_iter = 5
            else:
                num_iter = 2
            for j in range(0, num_iter):
                array_annotations.append(array_annotations[i])
                images.append(images[i])
    
    return images, array_annotations


def get_images_and_annotations(path_to_data, type_image, img_shape, data_type):
    ######################################### 1 #########################################
    carla_dataset_name_file = path_to_data + 'carla_dataset_test_31_10_anticlockwise_town_01_previous_v/dataset.csv'
    carla_dataset_file = open(carla_dataset_name_file, 'r')
    data_carla_dataset = carla_dataset_file.read()
    carla_dataset_file.close()

    array_annotations_carla_dataset_1 = []
    DIR_carla_dataset_images = path_to_data + 'carla_dataset_test_31_10_anticlockwise_town_01_previous_v/'
    list_images_carla_dataset = glob.glob(DIR_carla_dataset_images + '*')
    new_list_images_carla_dataset = []
    for image in list_images_carla_dataset:
        if image != path_to_data + 'carla_dataset_test_31_10_anticlockwise_town_01_previous_v/dataset.csv':
            new_list_images_carla_dataset.append(image)
    list_images_carla_dataset = new_list_images_carla_dataset

    images_paths_carla_dataset = sorted(list_images_carla_dataset, key=lambda x: int(x.split('/')[4].split('.png')[0]))

    array_annotations_carla_dataset_1 = pandas.read_csv(carla_dataset_name_file)
    images_ids, array_annotations_carla_dataset_1 = parse_csv(array_annotations_carla_dataset_1)

    images_carla_dataset_1 = get_images(path_to_data + 'carla_dataset_test_31_10_anticlockwise_town_01_previous_v/', images_ids, img_shape)
    images_carla_dataset_1, array_annotations_carla_dataset_1 = add_extreme_data(images_carla_dataset_1, array_annotations_carla_dataset_1)

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
    carla_dataset_file = open(carla_dataset_name_file, 'r')
    data_carla_dataset = carla_dataset_file.read()
    carla_dataset_file.close()

    array_annotations_carla_dataset_2 = []
    DIR_carla_dataset_images = path_to_data + 'carla_dataset_test_31_10_clockwise_town_01_previous_v/'
    list_images_carla_dataset = glob.glob(DIR_carla_dataset_images + '*')
    new_list_images_carla_dataset = []
    for image in list_images_carla_dataset:
        if image != path_to_data + 'carla_dataset_test_31_10_clockwise_town_01_previous_v/dataset.csv':
            new_list_images_carla_dataset.append(image)
    list_images_carla_dataset = new_list_images_carla_dataset

    images_paths_carla_dataset = sorted(list_images_carla_dataset, key=lambda x: int(x.split('/')[4].split('.png')[0]))

    array_annotations_carla_dataset_2 = pandas.read_csv(carla_dataset_name_file)
    images_ids, array_annotations_carla_dataset_2 = parse_csv(array_annotations_carla_dataset_2)

    images_carla_dataset_2 = get_images(path_to_data + 'carla_dataset_test_31_10_clockwise_town_01_previous_v/', images_ids, img_shape)
    images_carla_dataset_2, array_annotations_carla_dataset_2 = add_extreme_data(images_carla_dataset_2, array_annotations_carla_dataset_2)

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
    carla_dataset_file = open(carla_dataset_name_file, 'r')
    data_carla_dataset = carla_dataset_file.read()
    carla_dataset_file.close()

    array_annotations_carla_dataset_3 = []
    DIR_carla_dataset_images = path_to_data + 'carla_dataset_test_04_11_clockwise_town_01_previous_v_extreme/'
    list_images_carla_dataset = glob.glob(DIR_carla_dataset_images + '*')
    new_list_images_carla_dataset = []
    for image in list_images_carla_dataset:
        if image != path_to_data + 'carla_dataset_test_04_11_clockwise_town_01_previous_v_extreme/dataset.csv':
            new_list_images_carla_dataset.append(image)
    list_images_carla_dataset = new_list_images_carla_dataset

    images_paths_carla_dataset = sorted(list_images_carla_dataset, key=lambda x: int(x.split('/')[4].split('.png')[0]))

    array_annotations_carla_dataset_3 = pandas.read_csv(carla_dataset_name_file)
    images_ids, array_annotations_carla_dataset_3 = parse_csv(array_annotations_carla_dataset_3)

    images_carla_dataset_3 = get_images(path_to_data + 'carla_dataset_test_04_11_clockwise_town_01_previous_v_extreme/', images_ids, img_shape)
    images_carla_dataset_3, array_annotations_carla_dataset_3 = add_extreme_data(images_carla_dataset_3, array_annotations_carla_dataset_3)

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
    carla_dataset_file = open(carla_dataset_name_file, 'r')
    data_carla_dataset = carla_dataset_file.read()
    carla_dataset_file.close()

    array_annotations_carla_dataset_4 = []
    DIR_carla_dataset_images = path_to_data + 'carla_dataset_test_04_11_anticlockwise_town_03_previous_v/'
    list_images_carla_dataset = glob.glob(DIR_carla_dataset_images + '*')
    new_list_images_carla_dataset = []
    for image in list_images_carla_dataset:
        if image != path_to_data + 'carla_dataset_test_04_11_anticlockwise_town_03_previous_v/dataset.csv':
            new_list_images_carla_dataset.append(image)
    list_images_carla_dataset = new_list_images_carla_dataset

    images_paths_carla_dataset = sorted(list_images_carla_dataset, key=lambda x: int(x.split('/')[4].split('.png')[0]))

    array_annotations_carla_dataset_4 = pandas.read_csv(carla_dataset_name_file)
    images_ids, array_annotations_carla_dataset_4 = parse_csv(array_annotations_carla_dataset_4)

    images_carla_dataset_4 = get_images(path_to_data + 'carla_dataset_test_04_11_anticlockwise_town_03_previous_v/', images_ids, img_shape)
    images_carla_dataset_4, array_annotations_carla_dataset_4 = add_extreme_data(images_carla_dataset_4, array_annotations_carla_dataset_4)

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
    carla_dataset_file = open(carla_dataset_name_file, 'r')
    data_carla_dataset = carla_dataset_file.read()
    carla_dataset_file.close()

    array_annotations_carla_dataset_5 = []
    DIR_carla_dataset_images = path_to_data + 'carla_dataset_test_04_11_clockwise_town_03_previous_v/'
    list_images_carla_dataset = glob.glob(DIR_carla_dataset_images + '*')
    new_list_images_carla_dataset = []
    for image in list_images_carla_dataset:
        if image != path_to_data + 'carla_dataset_test_04_11_clockwise_town_03_previous_v/dataset.csv':
            new_list_images_carla_dataset.append(image)
    list_images_carla_dataset = new_list_images_carla_dataset

    images_paths_carla_dataset = sorted(list_images_carla_dataset, key=lambda x: int(x.split('/')[4].split('.png')[0]))

    array_annotations_carla_dataset_5 = pandas.read_csv(carla_dataset_name_file)
    images_ids, array_annotations_carla_dataset_5 = parse_csv(array_annotations_carla_dataset_5)

    images_carla_dataset_5 = get_images(path_to_data + 'carla_dataset_test_04_11_clockwise_town_03_previous_v/', images_ids, img_shape)
    images_carla_dataset_5, array_annotations_carla_dataset_5 = add_extreme_data(images_carla_dataset_5, array_annotations_carla_dataset_5)

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
    carla_dataset_file = open(carla_dataset_name_file, 'r')
    data_carla_dataset = carla_dataset_file.read()
    carla_dataset_file.close()

    array_annotations_carla_dataset_6 = []
    DIR_carla_dataset_images = path_to_data + 'carla_dataset_test_04_11_anticlockwise_town_05_previous_v/'
    list_images_carla_dataset = glob.glob(DIR_carla_dataset_images + '*')
    new_list_images_carla_dataset = []
    for image in list_images_carla_dataset:
        if image != path_to_data + 'carla_dataset_test_04_11_anticlockwise_town_05_previous_v/dataset.csv':
            new_list_images_carla_dataset.append(image)
    list_images_carla_dataset = new_list_images_carla_dataset

    images_paths_carla_dataset = sorted(list_images_carla_dataset, key=lambda x: int(x.split('/')[4].split('.png')[0]))

    array_annotations_carla_dataset_6 = pandas.read_csv(carla_dataset_name_file)
    images_ids, array_annotations_carla_dataset_6 = parse_csv(array_annotations_carla_dataset_6)

    images_carla_dataset_6 = get_images(path_to_data + 'carla_dataset_test_04_11_anticlockwise_town_05_previous_v/', images_ids, img_shape)
    images_carla_dataset_6, array_annotations_carla_dataset_6 = add_extreme_data(images_carla_dataset_6, array_annotations_carla_dataset_6)

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
    carla_dataset_file = open(carla_dataset_name_file, 'r')
    data_carla_dataset = carla_dataset_file.read()
    carla_dataset_file.close()

    array_annotations_carla_dataset_7 = []
    DIR_carla_dataset_images = path_to_data + 'carla_dataset_test_04_11_clockwise_town_05_previous_v/'
    list_images_carla_dataset = glob.glob(DIR_carla_dataset_images + '*')
    new_list_images_carla_dataset = []
    for image in list_images_carla_dataset:
        if image != path_to_data + 'carla_dataset_test_04_11_clockwise_town_05_previous_v/dataset.csv':
            new_list_images_carla_dataset.append(image)
    list_images_carla_dataset = new_list_images_carla_dataset

    images_paths_carla_dataset = sorted(list_images_carla_dataset, key=lambda x: int(x.split('/')[4].split('.png')[0]))

    array_annotations_carla_dataset_7 = pandas.read_csv(carla_dataset_name_file)
    images_ids, array_annotations_carla_dataset_7 = parse_csv(array_annotations_carla_dataset_7)

    images_carla_dataset_7 = get_images(path_to_data + 'carla_dataset_test_04_11_clockwise_town_05_previous_v/', images_ids, img_shape)
    images_carla_dataset_7, array_annotations_carla_dataset_7 = add_extreme_data(images_carla_dataset_7, array_annotations_carla_dataset_7)

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
    carla_dataset_file = open(carla_dataset_name_file, 'r')
    data_carla_dataset = carla_dataset_file.read()
    carla_dataset_file.close()

    array_annotations_carla_dataset_8 = []
    DIR_carla_dataset_images = path_to_data + 'carla_dataset_test_04_11_anticlockwise_town_07_previous_v/'
    list_images_carla_dataset = glob.glob(DIR_carla_dataset_images + '*')
    new_list_images_carla_dataset = []
    for image in list_images_carla_dataset:
        if image != path_to_data + 'carla_dataset_test_04_11_anticlockwise_town_07_previous_v/dataset.csv':
            new_list_images_carla_dataset.append(image)
    list_images_carla_dataset = new_list_images_carla_dataset

    images_paths_carla_dataset = sorted(list_images_carla_dataset, key=lambda x: int(x.split('/')[4].split('.png')[0]))

    array_annotations_carla_dataset_8 = pandas.read_csv(carla_dataset_name_file)
    images_ids, array_annotations_carla_dataset_8 = parse_csv(array_annotations_carla_dataset_8)

    images_carla_dataset_8 = get_images(path_to_data + 'carla_dataset_test_04_11_anticlockwise_town_07_previous_v/', images_ids, img_shape)
    images_carla_dataset_8, array_annotations_carla_dataset_8 = add_extreme_data(images_carla_dataset_8, array_annotations_carla_dataset_8)

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
    carla_dataset_file = open(carla_dataset_name_file, 'r')
    data_carla_dataset = carla_dataset_file.read()
    carla_dataset_file.close()

    array_annotations_carla_dataset_9 = []
    DIR_carla_dataset_images = path_to_data + 'carla_dataset_test_04_11_clockwise_town_07_previous_v/'
    list_images_carla_dataset = glob.glob(DIR_carla_dataset_images + '*')
    new_list_images_carla_dataset = []
    for image in list_images_carla_dataset:
        if image != path_to_data + 'carla_dataset_test_04_11_clockwise_town_07_previous_v/dataset.csv':
            new_list_images_carla_dataset.append(image)
    list_images_carla_dataset = new_list_images_carla_dataset

    images_paths_carla_dataset = sorted(list_images_carla_dataset, key=lambda x: int(x.split('/')[4].split('.png')[0]))

    array_annotations_carla_dataset_9 = pandas.read_csv(carla_dataset_name_file)
    images_ids, array_annotations_carla_dataset_9 = parse_csv(array_annotations_carla_dataset_9)

    images_carla_dataset_9 = get_images(path_to_data + 'carla_dataset_test_04_11_clockwise_town_07_previous_v/', images_ids, img_shape)
    images_carla_dataset_9, array_annotations_carla_dataset_9 = add_extreme_data(images_carla_dataset_9, array_annotations_carla_dataset_9)

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

    ######################################### 10 #########################################
    carla_dataset_name_file = path_to_data + 'carla_dataset_16_11_clockwise_town_01_extreme_2/dataset.csv'
    carla_dataset_file = open(carla_dataset_name_file, 'r')
    data_carla_dataset = carla_dataset_file.read()
    carla_dataset_file.close()

    array_annotations_carla_dataset_10 = []
    DIR_carla_dataset_images = path_to_data + 'carla_dataset_16_11_clockwise_town_01_extreme_2/'
    list_images_carla_dataset = glob.glob(DIR_carla_dataset_images + '*')
    new_list_images_carla_dataset = []
    for image in list_images_carla_dataset:
        if image != path_to_data + 'carla_dataset_16_11_clockwise_town_01_extreme_2/dataset.csv':
            new_list_images_carla_dataset.append(image)
    list_images_carla_dataset = new_list_images_carla_dataset

    images_paths_carla_dataset = sorted(list_images_carla_dataset, key=lambda x: int(x.split('/')[4].split('.png')[0]))

    array_annotations_carla_dataset_10 = pandas.read_csv(carla_dataset_name_file)
    images_ids, array_annotations_carla_dataset_10 = parse_csv(array_annotations_carla_dataset_10)

    images_carla_dataset_10 = get_images(path_to_data + 'carla_dataset_16_11_clockwise_town_01_extreme_2/',
                                        images_ids, img_shape)
    images_carla_dataset_10, array_annotations_carla_dataset_10 = add_extreme_data(images_carla_dataset_10,
                                                                                 array_annotations_carla_dataset_10)

    array_annotations_v = []
    array_annotations_w = []
    array_annotations_b = []
    for annotation in array_annotations_carla_dataset_10:
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

    array_annotations_carla_dataset_10 = normalized_annotations

    ######################################### 11 #########################################
    carla_dataset_name_file = path_to_data + 'carla_dataset_24_07_anticlockwise_town_01_extreme/dataset.csv'
    carla_dataset_file = open(carla_dataset_name_file, 'r')
    data_carla_dataset = carla_dataset_file.read()
    carla_dataset_file.close()

    array_annotations_carla_dataset_11 = []
    DIR_carla_dataset_images = path_to_data + 'carla_dataset_24_07_anticlockwise_town_01_extreme/'
    list_images_carla_dataset = glob.glob(DIR_carla_dataset_images + '*')
    new_list_images_carla_dataset = []
    for image in list_images_carla_dataset:
        if image != path_to_data + 'carla_dataset_24_07_anticlockwise_town_01_extreme/dataset.csv':
            new_list_images_carla_dataset.append(image)
    list_images_carla_dataset = new_list_images_carla_dataset

    images_paths_carla_dataset = sorted(list_images_carla_dataset, key=lambda x: int(x.split('/')[4].split('.png')[0]))

    array_annotations_carla_dataset_11 = pandas.read_csv(carla_dataset_name_file)
    images_ids, array_annotations_carla_dataset_11 = parse_csv(array_annotations_carla_dataset_11)

    images_carla_dataset_11 = get_images(path_to_data + 'carla_dataset_24_07_anticlockwise_town_01_extreme/',
                                         images_ids, img_shape)
    images_carla_dataset_11, array_annotations_carla_dataset_11 = add_extreme_data(images_carla_dataset_11,
                                                                                   array_annotations_carla_dataset_11)

    array_annotations_v = []
    array_annotations_w = []
    array_annotations_b = []
    for annotation in array_annotations_carla_dataset_11:
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

    array_annotations_carla_dataset_11 = normalized_annotations

    ###########
    
    array_imgs = images_carla_dataset_1 + images_carla_dataset_2 + images_carla_dataset_3 + images_carla_dataset_4 + \
        images_carla_dataset_5 + images_carla_dataset_6 + images_carla_dataset_7 + images_carla_dataset_8 + images_carla_dataset_9 + \
        images_carla_dataset_10 + images_carla_dataset_11
    array_annotations = array_annotations_carla_dataset_1 + array_annotations_carla_dataset_2 + array_annotations_carla_dataset_3 + \
        array_annotations_carla_dataset_4 + array_annotations_carla_dataset_5 + array_annotations_carla_dataset_6 + \
        array_annotations_carla_dataset_7 + array_annotations_carla_dataset_8 + array_annotations_carla_dataset_9 + \
        array_annotations_carla_dataset_10 + array_annotations_carla_dataset_11

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

def process_dataset(path_to_data, type_image, data_type, img_shape, optimize_mode=False):

    if not optimize_mode:
        array_imgs, array_annotations = get_images_and_annotations(path_to_data, type_image, img_shape, data_type)
        images_train, annotations_train, images_validation, annotations_validation = separate_dataset_into_train_validation(
            array_imgs, array_annotations)
    else:
        images_train, annotations_train = get_images_and_annotations(path_to_data, type_image, img_shape, data_type)
        images_validation, annotations_validation = images_train, annotations_train

    return images_train, annotations_train, images_validation, annotations_validation
