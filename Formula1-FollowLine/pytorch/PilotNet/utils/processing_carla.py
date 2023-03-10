import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
import csv


def load_data(folder):
    name_folder = folder #+ '/' #+ '/Images/'
    name_file = folder + 'dataset.csv'
    file = open(name_file, 'r')
    reader = csv.DictReader(file)
    data = []
    images = []
    for row in reader: # reading all values
        images.append(name_folder + row['image_id'])
        data.append((float(row['throttle']), float(row['steer']), float(row['brake']), float(row['velocity']), float(row['timestamp'])))
    file.close()
    return images, data

def get_images(list_images, type_image, array_imgs):
    # Read the images
    for name in tqdm(list_images):
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (200,66))
        array_imgs.append(img)

    print('--------')
    print(array_imgs[0].shape)
    print('--------')
    return array_imgs

def parse_csv(data, array):
    print(data)
    # Process csv
    for v, w in data:
        array.append((float(v), float(w)))

    return array

def preprocess_data(array, imgs, data_type):
    # Data augmentation
    # Take the image and just flip it and negate the measurement
    
    #flip_imgs = []
    #array_flip = []
    #for i in tqdm(range(len(imgs))):
    #    flip_imgs.append(cv2.flip(imgs[i], 1))
    #    array_flip.append((array[i][0], -array[i][1]))
    #new_array = array + array_flip
    #new_array_imgs = imgs + flip_imgs

    #new_array = array
    #new_array_imgs = imgs

    #print('**')
    #print(len(new_array))
    #print(len(new_array_imgs))
    #print('**')

    array_annotations = array
    array_imgs = imgs
    
    print('**')
    print(len(array_annotations))
    print(len(array_imgs))
    print('**')
    

    if data_type == 'extreme':
        '''
        extreme_case_1_img = []
        extreme_case_2_img = []
        extreme_case_1_array = []
        extreme_case_2_array = []

        for i in tqdm(range(len(new_array_imgs))):
            print(i)
            print(new_array[i])
            if abs(new_array[i][1]) > 2:
                extreme_case_2_img.append(new_array_imgs[i])
                extreme_case_2_array.append(new_array[i])
            elif abs(new_array[i][1]) > 1:
                extreme_case_1_img.append(new_array_imgs[i])
                extreme_case_1_array.append(new_array[i])

        new_array += extreme_case_1_array*5 + extreme_case_2_array*10
        new_array_imgs += extreme_case_1_img*5 + extreme_case_2_img*10
        '''

        for i in range(0, len(array_annotations)):
            if abs(array_annotations[i][1]) >= 0.1:
                if abs(array_annotations[i][1]) >= 0.3:
                    num_iter = 15
                    #num_iter = 10
                elif abs(array_annotations[i][1]) >= 0.2:
                    num_iter = 5
                else:
                    num_iter = 2
                for j in range(0, num_iter):
                    array_annotations.append(array_annotations[i])
                    array_imgs.append(array_imgs[i])
            if abs(array_annotations[i][2]) >= 0.1:
                if abs(array_annotations[i][2]) >= 0.3:
                    num_iter = 15
                    #num_iter = 10
                elif abs(array_annotations[i][2]) >= 0.2:
                    num_iter = 5
                else:
                    num_iter = 2
                for j in range(0, num_iter):
                    array_annotations.append(array_annotations[i])
                    array_imgs.append(array_imgs[i])


    print('**')
    print(len(array_annotations))
    print(len(array_imgs))
    print('**')

    array_annotations = normalize_annotations(array_annotations)

    return array_annotations, array_imgs


def normalize_annotations(array_annotations):
    array_annotations_v = []
    array_annotations_w = []
    array_annotations_b = []
    for annotation in array_annotations:
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

    return normalized_annotations

'''
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

    normalized_X = normalize(array_annotations_v, min=6.5, max=24)
    normalized_Y = normalize(array_annotations_w, min=-7.1, max=7.1)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    return normalized_annotations
'''

def normalize(x, min, max):
    x = np.asarray(x)
    return (x - min) / (max - min)

def check_path(path):
    if not os.path.exists(path):
        print(f"{path} not exist")
        os.makedirs(path)
        print(f"Create {path} success")
