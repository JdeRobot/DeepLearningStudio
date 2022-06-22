import os
import argparse
import datetime
import time
import h5py

import numpy as np

from utils.dataset import get_augmentations, DatasetSequence
from utils.processing import process_dataset
from utils.pilotnet_x3 import pilotnet_x3
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
from tensorflow.python.keras.saving import hdf5_format


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", action='append', help="Directory to find Data")
    parser.add_argument("--preprocess", action='append', default=None,
                        help="preprocessing information: choose from crop/nocrop and normal/extreme")
    parser.add_argument("--data_augs", type=int, default=0, help="Data Augmentations: 0=No / 1=Normal / 2=Normal+Weather changes")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of Epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for Policy Net")
    parser.add_argument("--img_shape", type=str, default=(200, 66, 3), help="Image shape")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    path_to_data = args.data_dir[0]
    preprocess = args.preprocess
    data_augs = args.data_augs
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    img_shape = tuple(map(int, args.img_shape.split(',')))

    if 'no_crop' in preprocess:
        type_image = 'no_crop'
    else:
        type_image = 'crop'

    if 'extreme' in preprocess:
        data_type = 'extreme'
    else:
        data_type = 'no_extreme'

    images_train, array_annotations_train, images_val, array_annotations_val = process_dataset(path_to_data, type_image,
                                                                                               data_type, img_shape)

    # Train
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print(timestr)

    hparams = {
        'batch_size': batch_size,
        'n_epochs': num_epochs,
        'checkpoint_dir': '../logs_test/'
    }

    print(hparams)

    model_name = 'pilotnet_x3'
    model = pilotnet_x3(img_shape, learning_rate)
    model_filename = timestr + '_pilotnet_x3_new_dataset_opencv_10fps_anticlowise_more_extreme_cases_support_300_epochs'
    model_file = model_filename + '.h5'

    AUGMENTATIONS_TRAIN, AUGMENTATIONS_TEST = get_augmentations(data_augs)

    # Training data
    train_gen = DatasetSequence(images_train, array_annotations_train, hparams['batch_size'],
                                augmentations=AUGMENTATIONS_TRAIN)

    # Validation data
    valid_gen = DatasetSequence(images_val, array_annotations_val, hparams['batch_size'],
                                augmentations=AUGMENTATIONS_TEST)

    # Define callbacks
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    earlystopping = EarlyStopping(monitor="mae", patience=30, verbose=1, mode='auto')
    # Create a callback that saves the model's weights
    checkpoint_path = model_filename + '_cp.h5'
    cp_callback = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
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
        # workers=2, use_multiprocessing=False,
        callbacks=[tensorboard_callback, earlystopping, cp_callback, csv_logger])

    # Save model
    model.save(model_file)

    # Evaluate model
    score = model.evaluate(valid_gen, verbose=0)

    print('Evaluating')
    print('Test loss: ', score[0])
    print('Test mean squared error: ', score[1])
    print('Test mean absolute error: ', score[2])

    model_path = model_file
    # Save model metadata
    with h5py.File(model_path, mode='w') as f:
        hdf5_format.save_model_to_hdf5(model, f)
        f.attrs['experiment_name'] = ''
        f.attrs['experiment_description'] = ''
        f.attrs['batch_size'] = hparams['batch_size']
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
