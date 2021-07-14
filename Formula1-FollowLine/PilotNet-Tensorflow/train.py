############################################################## 2 READ DATASET ##############################################################

import glob
import os
import cv2

import numpy as np

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

image_shape=(200, 66)

def get_images(list_images, type_image):
    # Read the images
    array_imgs = []
    for name in list_images:
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #if type_image == 'cropped':
        #    img = img[240:480, 0:640]
        #img = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)))
        img = cv2.resize(img, image_shape)
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
'''
def add_extreme_data(images, array_annotations):
    for i in range(0, len(array_annotations)):
        if abs(array_annotations[i][1]) >= 1:
            if abs(array_annotations[i][1]) >= 2:
                #num_iter = 10
                #num_iter = 15
                num_iter = 20
            else:
                #num_iter = 5
                #num_iter = 10
                num_iter = 15
            for j in range(0, num_iter):
                array_annotations.append(array_annotations[i])
                images.append(images[i])
        if float(array_annotations[i][0]) <= 2:
            #for j in range(0, 1):
            #for j in range(0, 5):
            for j in range(0, 10):
                array_annotations.append(array_annotations[i])
                images.append(images[i])
                
    return images, array_annotations
'''
def normalize(x):
    x = np.asarray(x)
    return (x - x.min()) / (np.ptp(x))

print('---- train ----')
train_name_file = '../../../complete_dataset/Train/train.json'
file_train = open(train_name_file, 'r')
data_train = file_train.read()
file_train.close()

array_annotations_train = []
DIR_train_images = '../../../complete_dataset/Train/Images/'
list_images_train = glob.glob(DIR_train_images + '*')
images_paths_train = sorted(list_images_train, key=lambda x: int(x.split('/')[6].split('.png')[0]))
#print(images_paths_train)
array_annotations_train = parse_json(data_train)
print(type(images_paths_train))
print(len(images_paths_train))
print(images_paths_train[0])
print(type(array_annotations_train))
print(array_annotations_train[0])
print(len(array_annotations_train))

images_train = get_images(images_paths_train, 'cropped')
print(len(images_train))
print(type(images_train))
images_train, array_annotations_train = flip_images(images_train, array_annotations_train)
print(len(images_train))
print(type(images_train))
print(len(array_annotations_train))
#images_train, array_annotations_train = add_extreme_data(images_train, array_annotations_train)
#print(len(images_train))
#print(type(images_train))
#print(len(array_annotations_train))

array_annotations_v = []
array_annotations_w = []
for annotation in array_annotations_train:
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
    
array_annotations_train = normalized_annotations

print(len(images_train))
print(type(images_train))
print(len(array_annotations_train))

print('---- val ----')
test_name_file = '../../../complete_dataset/Test/test.json'
file_test = open(test_name_file, 'r')
data_test = file_test.read()
file_test.close()

DIR_test_images = '../../../complete_dataset/Test/Images/'
list_images_val = glob.glob(DIR_test_images + '*')
images_paths_val = sorted(list_images_val, key=lambda x: int(x.split('/')[6].split('.png')[0]))
array_annotations_val = parse_json(data_test)
print(type(images_paths_val))
print(len(images_paths_val))
print(images_paths_val[0])
print(type(array_annotations_val))
print(array_annotations_val[0])
print(len(array_annotations_val))

images_val = get_images(images_paths_val, 'cropped')
print(len(images_val))
print(type(images_val))
images_val, array_annotations_val = flip_images(images_val, array_annotations_val)
print(len(images_val))
print(type(images_val))
print(len(array_annotations_val))
#images_val, array_annotations_val = add_extreme_data(images_val, array_annotations_val)
#print(len(images_val))
#print(type(images_val))
#print(len(array_annotations_val))


array_annotations_v = []
array_annotations_w = []
for annotation in array_annotations_val:
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
    
array_annotations_val = normalized_annotations

print(len(images_val))
print(type(images_val))
print(len(array_annotations_val))


############################################################## 7 TRAIN ##############################################################

import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
import datetime

timestr = time.strftime("%Y%m%d-%H%M%S")
print(timestr)
#img_shape = (60, 160, 3)
#img_shape = (66, 200, 3)
img_shape = (200, 66, 3)

hparams = {
    'train_batch_size': 50, 
    'val_batch_size': 50,
    'batch_size': 50,
    'n_epochs': 100, 
    'checkpoint_dir': '../logs_test/'
}

print(hparams)


from utils.pilotnet import pilotnet_model

model_name = 'pilotnet_model'
model = pilotnet_model(img_shape)
model_filename = timestr + '_pilotnet_model_100_all_n_extreme_3_albumentations_no_crop'
model_file = model_filename + '.h5'

from utils.dataset import DatasetSequence
from utils.dataset import AUGMENTATIONS_TRAIN, AUGMENTATIONS_TEST
# Training data
train_gen = DatasetSequence(images_train, array_annotations_train, hparams['batch_size'], augmentations=AUGMENTATIONS_TRAIN)

# Validation data
valid_gen = DatasetSequence(images_val, array_annotations_val, hparams['batch_size'], augmentations=AUGMENTATIONS_TEST)
#valid_gen = DatasetSequence(images_val, array_annotations_val, hparams['batch_size'], augmentations=AUGMENTATIONS_TRAIN)


# Define callbacks
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
earlystopping=EarlyStopping(monitor="mae", patience=30, verbose=1, mode='auto')
# Create a callback that saves the model's weights
checkpoint_path = model_filename + '_cp.h5'
cp_callback = ModelCheckpoint(filepath=checkpoint_path, monitor='mse', save_best_only=True, verbose=1)
csv_logger = CSVLogger(model_filename + '.csv', append=True)

# Print layers
print(model)
model.build(img_shape)
print(model.summary())
# Training
model.fit(
    train_gen,
    epochs=hparams['n_epochs'],
    verbose=2,
    validation_data=valid_gen,
    #workers=2, use_multiprocessing=False,
    callbacks=[tensorboard_callback, earlystopping, cp_callback, csv_logger])

# Save the model
model.save(model_file)


# Evaluate the model
score = model.evaluate_generator(valid_gen, verbose=0)

print('Evaluating')
print('Test loss: ', score[0])
print('Test mean squared error: ', score[1])
print('Test mean absolute error: ', score[2])


# SAVE METADATA
from tensorflow.python.keras.saving import hdf5_format
import h5py

model_path = model_file
# Save model
with h5py.File(model_path, mode='w') as f:
    hdf5_format.save_model_to_hdf5(model, f)
    f.attrs['experiment_name'] = ''
    f.attrs['experiment_description'] = ''
    f.attrs['batch_size'] = hparams['train_batch_size']
    f.attrs['nb_epoch'] = hparams['n_epochs']
    f.attrs['model'] = model_name
    f.attrs['img_shape'] = img_shape
    f.attrs['normalized_dataset'] = True
    f.attrs['sequences_dataset'] = True
    f.attrs['gpu_trained'] = True
    f.attrs['data_augmentation'] = True
    f.attrs['extreme_data'] = False
    f.attrs['split_test_train'] = 0.30
    f.attrs['instances_number'] = len(array_annotations_train)
    f.attrs['loss'] = score[0]
    f.attrs['mse'] = score[1]
    f.attrs['mae'] = score[2]
    f.attrs['csv_path'] = model_filename + '.csv'
