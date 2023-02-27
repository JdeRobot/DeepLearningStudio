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
            #print(folder_prefix + name)
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
    #print(csv_data)
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

    images_paths_carla_dataset = sorted(list_images_carla_dataset, key=lambda x: int(x.split('/')[6].split('.png')[0]))

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

    ###########
    
    array_imgs = images_carla_dataset_1

    array_annotations = array_annotations_carla_dataset_1
    
    return array_imgs, array_annotations

'''
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
'''
'''
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

    ######################################### VAL 2 #########################################
    montmelo_name_file = path_to_data + 'montmelo_12_05_2022_opencv_clockwise_1/data.csv'
    dir_montmelo_images = path_to_data + 'montmelo_12_05_2022_opencv_clockwise_1/'
    list_images_montmelo = glob.glob(dir_montmelo_images + '*')
    new_list_images_montmelo = []
    for image in list_images_montmelo:
        if image != path_to_data + 'montmelo_12_05_2022_opencv_clockwise_1/data.csv':
            new_list_images_montmelo.append(image)
    list_images_montmelo = new_list_images_montmelo
    images_paths_montmelo = sorted(list_images_montmelo, key=lambda x: int(x.split('/')[-1].split('.png')[0]))

    array_annotations_montmelo = pandas.read_csv(montmelo_name_file)
    array_annotations_montmelo = parse_csv(array_annotations_montmelo)

    images_montmelo = get_images(images_paths_montmelo, type_image, img_shape)
    images_montmelo, array_annotations_montmelo = flip_images(images_montmelo,
                                                                        array_annotations_montmelo)
    if data_type == 'extreme':
        images_montmelo, array_annotations_montmelo = add_extreme_data(images_montmelo,
                                                                                 array_annotations_montmelo)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_montmelo:
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

    array_annotations_montmelo = normalized_annotations
    print("Loaded Montmelo!!")
######################################### VAL 3 #########################################
    montreal_name_file = path_to_data + 'montreal_12_05_2022_opencv_clockwise_1/data.csv'
    dir_montreal_images = path_to_data + 'montreal_12_05_2022_opencv_clockwise_1/'
    list_images_montreal = glob.glob(dir_montreal_images + '*')
    new_list_images_montreal = []
    for image in list_images_montreal:
        if image != path_to_data + 'montreal_12_05_2022_opencv_clockwise_1/data.csv':
            new_list_images_montreal.append(image)
    list_images_montreal = new_list_images_montreal
    images_paths_montreal = sorted(list_images_montreal, key=lambda x: int(x.split('/')[-1].split('.png')[0]))

    array_annotations_montreal = pandas.read_csv(montreal_name_file)
    array_annotations_montreal = parse_csv(array_annotations_montreal)

    images_montreal = get_images(images_paths_montreal, type_image, img_shape)
    images_montreal, array_annotations_montreal = flip_images(images_montreal,
                                                                        array_annotations_montreal)
    if data_type == 'extreme':
        images_montreal, array_annotations_montreal = add_extreme_data(images_montreal,
                                                                                 array_annotations_montreal)

    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations_montreal:
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

    array_annotations_montreal = normalized_annotations
    print("Loaded Montreal!!")
    #############################################################################

    array_imgs = images_simple_circuit + \
                    images_montmelo + \
                    images_montreal

    array_annotations = array_annotations_simple_circuit + \
                        array_annotations_montmelo + \
                        array_annotations_montreal

    return array_imgs, array_annotations
    '''

def process_dataset(path_to_data, type_image, data_type, img_shape, optimize_mode=False):

    if not optimize_mode:
        array_imgs, array_annotations = get_images_and_annotations(path_to_data, type_image, img_shape, data_type)
        images_train, annotations_train, images_validation, annotations_validation = separate_dataset_into_train_validation(
            array_imgs, array_annotations)
    else:
        images_train, annotations_train = get_images_and_annotations(path_to_data, type_image, img_shape, data_type)
        # images_train, annotations_train = [], []
        images_validation, annotations_validation = get_images_and_annotations(path_to_data, type_image, img_shape, data_type)
            

    return images_train, annotations_train, images_validation, annotations_validation