from tqdm import tqdm
import numpy as np
import os
import os.path
import sys
import random
import math
import cv2
import gc

class datasource(object):
	def __init__(self, images, speed):
		self.images = images
		self.speed = speed

def preprocess(images, speeds):
	images_out = [] #final result
	speeds_out = [] #final result
	#Resize input images
	for i in tqdm(range(len(images))):
		X = cv2.imread(images[i])
		if X is not None:
			X = cv2.resize(X, (224, 224))
			X = np.transpose(X,(2,0,1))
			X = np.squeeze(X)
			X = np.transpose(X, (1,2,0))
			Y = np.expand_dims(X, axis=0)
			images_out.append(Y)
			speeds_out.append(speeds[i])
	del X, i
	gc.collect()
	return images_out, speeds_out

def get_data(dataset):
	speed = []
	images = []
	
	with open(dataset+'data.txt') as f:
		next(f)  # skip the header line
		for line in f:
			fname,p0,p1,p2,p3 = line.split()
			p0 = float(p0)
			p1 = float(p1)
			p2 = float(p2)
			p3 = float(p3)
			speed.append((p0,p1,p2,p3))
			images.append(dataset+fname)
	images_out, speeds_out = preprocess(images, speed)
	return datasource(images_out, speeds_out)

def getTrainSource(dataset_train):
	datasource_train = get_data(dataset_train)

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

def check_path(path):
    if not os.path.exists(path):
        print(f"{path} not exist")
        os.makedirs(path)
        print(f"Create {path} success")