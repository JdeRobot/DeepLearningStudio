from tqdm import tqdm
import numpy as np
import os
import os.path
import sys
import random
import math
import cv2
import gc
import json
import glob

class datasource(object):
    def __init__(self, images, speed):
        self.images = images
        self.speed = speed

def preprocess(images, speeds, type_image):
    images_out = [] #final result
    speeds_out = [] #final result
    #Resize input images
    for i in tqdm(range(len(images))):
        img = cv2.imread(images[i])
        if 'cropped' in type_image:
            img = img[120:240,0:320]
            img = cv2.resize(img, (int(224), int(224)))
        else:
            target_height = int(224)
            target_width = int(target_height * img.shape[1]/img.shape[0])
            img_resized = cv2.resize(img, (target_width, target_height))
            padding_left = int((224 - target_width)/2)
            padding_right = 224 - target_width - padding_left
            img = cv2.copyMakeBorder(img_resized.copy(),0,0,padding_left,padding_right,cv2.BORDER_CONSTANT,value=[0, 0, 0])
        X = img
        if X is not None:
            X = cv2.resize(X, (224, 224))
            X = np.transpose(X,(2,0,1))
            X = np.squeeze(X)
            X = np.transpose(X, (1,2,0))
            if 'stacked' in type_image:
                X = np.expand_dims(X, axis=0)
            images_out.append(X)
            speeds_out.append(speeds[i])
    del X, i
    gc.collect()
    return images_out, speeds_out

def get_data(dataset, type_image):
    speed = []
    images = []
    list_images = glob.glob(dataset + '/Images/*')
    list_images = sorted(list_images, key=lambda x: int(x.split('/')[-1].split('.png')[0].split('image')[-1]))
    data_file = open(dataset + '/data.json', 'r')
    data_list = json.load(data_file)
    data_list = sorted(data_list, key=lambda x: x["iter"])
    data_file.close()
    
    for data in data_list:
        p0 = float(data['v'])
        p1 = float(data['w'])
        p2 = float(data['vz'])
        speed.append((p0,p1,p2))
        images.append(list_images[data['iter']])
    images_out, speeds_out = preprocess(images, speed, type_image)
    return datasource(images_out, speeds_out)

def getTrainSource(dataset_train, type_image):
    datasource_train = get_data(dataset_train, type_image)

    images_train = []
    speed_train = []

    for i in range(len(datasource_train.images)):
        images_train.append(datasource_train.images[i])
        speed_train.append(datasource_train.speed[i])

    return datasource(images_train, speed_train)

def getTestSource(dataset_test):
    datasource_test = get_data(dataset_test)
    
    images_test = []
    speed_test = []

    for i in range(len(datasource_test.images)):
        images_test.append(datasource_test.images[i])
        speed_test.append(datasource_test.speed[i])

    return datasource(images_test, speed_test)

def preprocess_data(array, imgs, data_type):
    # Data augmentation
    # Take the image and just flip it and negate the measurement
    flip_imgs = []
    array_flip = []
    for i in tqdm(range(len(imgs))):
        flip_imgs.append(cv2.flip(imgs[i], 1))
        array_flip.append((array[i][0], -array[i][1], array[i][2]))
    new_array = array + array_flip
    new_array_imgs = imgs + flip_imgs

    if data_type == 'extreme':
        extreme_case_1_img = []
        extreme_case_2_img = []
        extreme_case_1_array = []
        extreme_case_2_array = []

        for i in tqdm(range(len(new_array_imgs))):
            if abs(new_array[i][1]) > 2:
                extreme_case_2_img.append(new_array_imgs[i])
                extreme_case_2_array.append(new_array[i])
            elif abs(new_array[i][1]) > 1:
                extreme_case_1_img.append(new_array_imgs[i])
                extreme_case_1_array.append(new_array[i])

        new_array += extreme_case_1_array*5 + extreme_case_2_array*10
        new_array_imgs += extreme_case_1_img*5 + extreme_case_2_img*10

    return new_array, new_array_imgs

def check_path(path):
    if not os.path.exists(path):
        print(f"{path} not exist")
        os.makedirs(path)
        print(f"Create {path} success")