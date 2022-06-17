import time
import datetime
import argparse
import h5py

from tensorflow.python.keras.saving import hdf5_format
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
from utils.processing import process_dataset
from utils.deepest_lstm_tinypilotnet import deepest_lstm_tinypilotnet_model
from utils.dataset import get_augmentations, DatasetSequence


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", action='append', help="Directory to find dataset")
    parser.add_argument("--preprocess", action='append', default=None, help="preprocessing information: choose from crop/nocrop and normal/extreme")
    parser.add_argument("--data_augs", action='append', type=bool, default=None, help="Data Augmentations True/False")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
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
    img_shape = tuple(map(int, args.img_shape.split(',')))

    if 'no_crop' in preprocess:
        type_image = 'no_crop'
    else:
        type_image = 'cropped'

    if 'extreme' in preprocess:
        data_type = 'extreme'
    else:
        data_type = 'no_extreme'

    images_train, annotations_train, images_validation, annotations_validation = process_dataset(path_to_data, type_image, data_type, img_shape)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    print(timestr)
    print(images_train.shape)
    print(annotations_train.shape)
    print(images_validation.shape)
    print(annotations_validation.shape)

    hparams = {
        'train_batch_size': batch_size,
        'val_batch_size': batch_size,
        'batch_size': batch_size,
        'n_epochs': num_epochs,
        'checkpoint_dir': '../logs_test/'
    }

    print(hparams)

    model_name = 'deepest_lstm_tinypilotnet'
    model = deepest_lstm_tinypilotnet_model(img_shape)
    model_filename = timestr + '_deepest_lstm_tinypilotnet_300_all_crop_no_seq_unique_albu_extreme_seq'
    model_file = model_filename + '.h5'

    AUGMENTATIONS_TRAIN, AUGMENTATIONS_TEST = get_augmentations(data_augs)

    # Training data
    train_gen = DatasetSequence(images_train, annotations_train, hparams['batch_size'],
                                augmentations=AUGMENTATIONS_TRAIN)

    # Validation data
    valid_gen = DatasetSequence(images_validation, annotations_validation, hparams['batch_size'],
                                augmentations=AUGMENTATIONS_TEST)

    # Define callbacks
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    earlystopping = EarlyStopping(monitor="mae", patience=30, verbose=1, mode='auto')
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
        callbacks=[tensorboard_callback, earlystopping, cp_callback, csv_logger])

    # Save the model
    model.save(model_file)

    # Evaluate the model
    score = model.evaluate_generator(valid_gen, verbose=0)

    print('Evaluating')
    print('Test loss: ', score[0])
    print('Test mean squared error: ', score[1])
    print('Test mean absolute error: ', score[2])

    model_path = model_file
    # Save model
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
        f.attrs['instances_number'] = len(annotations_train)
        f.attrs['loss'] = score[0]
        f.attrs['mse'] = score[1]
        f.attrs['mae'] = score[2]
        f.attrs['csv_path'] = model_filename + '.csv'
