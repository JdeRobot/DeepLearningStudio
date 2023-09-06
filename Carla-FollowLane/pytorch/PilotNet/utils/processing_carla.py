import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
import csv


def load_data(folder):
    name_folder = folder
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
        img = cv2.resize(img, (66,200))
        array_imgs.append(img)

    return array_imgs

def parse_csv(data, array):
    # Process csv
    for v, w in data:
        array.append((float(v), float(w)))

    return array

def preprocess_data(array, imgs, data_type):
    # Data augmentation

    array_annotations = array
    array_imgs = imgs  

    if data_type == 'extreme':
        for i in range(0, len(array_annotations)):
            if abs(array_annotations[i][1]) >= 0.1:
                if abs(array_annotations[i][1]) >= 0.3:
                    num_iter = 20
                    #num_iter = 15
                    #num_iter = 10
                elif abs(array_annotations[i][1]) >= 0.2:
                    num_iter = 10
                    #num_iter = 5
                else:
                    num_iter = 5
                    #num_iter = 2
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

def check_path(path):
    if not os.path.exists(path):
        print(f"{path} not exist")
        os.makedirs(path)
        print(f"Create {path} success")
