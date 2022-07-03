# Re-run after Kernel restart
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
# from tensorflow.python.saved_model import tag_constants
# from tensorflow.python.compiler.tensorrt import trt_convert as trt
# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import pathlib
import argparse
from utils.dataset import get_augmentations, DatasetSequence
from utils.processing import process_dataset
from tqdm import tqdm


def measure_inference_time(tflite_model, images_val):
    # measure average inference time
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    inf_time = []
    r_idx = np.random.randint(0, len(images_val), 1000)
    for i in tqdm(r_idx):
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        test_image = np.expand_dims(images_val[i], axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_image)

        start_t = time.time()
        # Run inference.
        interpreter.invoke()
        # pred = tflite_model.predict(img, verbose=0)
        inf_time.append(time.time() - start_t)
        # Post-processing
        output = interpreter.tensor(output_index)
        
    return np.mean(inf_time)

def measure_mse(tflite_model, images_val, valid_set, batch_size):
    # measure average inference time
    # put to interpreter for inference
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    # resize input and output tensor to handle batch size
    interpreter.resize_tensor_input(input_index, (batch_size, *images_val[0].shape))
    interpreter.resize_tensor_input(output_index, (batch_size, 2))
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    metric = 0.0
    for idx in tqdm(range((len(valid_set)-1))):
        test_images, test_labels = valid_set[idx]
        # Pre-processing
        interpreter.set_tensor(input_index, test_images)
        # Run inference.
        interpreter.invoke()
        # Post-processing
        output = interpreter.get_tensor(output_index)
        metric += np.mean(tf.keras.losses.mse(test_labels, output).numpy())

    return metric/(len(valid_set)-1)


def evaluate_model(model_path, tflite_model, valid_set, images_val, batch_size):
    '''
    Calculate accuracy, model size and inference time for the given model.
    Args:
        model_path: path to saved tflite model
        tflite_model: converted model instance (to tflite)
        valid_set: dataset to do test for accuracy
    return:
        accuracy, model_size, inf_time
    '''
    model_size = os.path.getsize(model_path) / float(2**20)
    
    mse = measure_mse(tflite_model, images_val, valid_set, batch_size)

    inf_time = measure_inference_time(tflite_model, images_val)

    return model_size, mse, inf_time



def convert_baseline(model_path, model_name, tflite_models_dir, valid_set, images_val, batch_size):
    # convertering original model to tflite
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_model_file = tflite_models_dir/f"{model_name}_model.tflite"
    tflite_model_file.write_bytes(tflite_model) # save model
    model_size, mse, inf_time = evaluate_model(tflite_model_file, tflite_model, valid_set, images_val, batch_size)
    print("********** Baseline stats **********")
    print("Model size (MB):", model_size)
    print("MSE:", mse)
    print("Inference time (s):", inf_time)
    return model_size, mse, inf_time


def dynamic_range_quantization(model_path, model_name, tflite_models_dir, valid_set, images_val, batch_size):
    # Post-training dynamic range quantization
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_model_quant_file = tflite_models_dir/f"{model_name}_model_quant.tflite"
    tflite_model_quant_file.write_bytes(tflite_model) # save model
    model_size, mse, inf_time = evaluate_model(tflite_model_quant_file, tflite_model, valid_set, images_val, batch_size)
    print("********** Dynamic range Q stats **********")
    print("Model size (MB):", model_size)
    print("MSE:", mse)
    print("Inference time (s):", inf_time)
    return model_size, mse, inf_time

def load_data(args):

    img_shape = tuple(map(int, args.img_shape.split(',')))

    if 'no_crop' in args.preprocess:
        type_image = 'no_crop'
    else:
        type_image = 'crop'

    if 'extreme' in args.preprocess:
        data_type = 'extreme'
    else:
        data_type = 'no_extreme'
    ##!! All dataset alloted to val/test
    images_train, annotations_train, images_val, annotations_val = process_dataset(args.data_dir, type_image,
                                                                                    data_type, img_shape, optimize_mode=True)
    AUGMENTATIONS_TRAIN, AUGMENTATIONS_TEST = get_augmentations(args.data_augs)
    # Training data
    train_gen = DatasetSequence(images_train, annotations_train, args.batch_size,
                                augmentations=AUGMENTATIONS_TRAIN)

    # Validation data
    valid_gen = DatasetSequence(images_val, annotations_val, args.batch_size,
                                augmentations=AUGMENTATIONS_TEST)

    return train_gen, valid_gen, images_val

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", action='append', help="Directory to find Data")
    parser.add_argument("--preprocess", action='append', default=None,
                        help="preprocessing information: choose from crop/nocrop and normal/extreme")
    parser.add_argument("--data_augs", type=int, default=0, help="Data Augmentations: 0=No / 1=Normal / 2=Normal+Weather changes")
    parser.add_argument("--img_shape", type=str, default=(200, 66, 3), help="Image shape")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument('--model_path', type=str, default='trained_models/pilotnet.h5', help="Path to directory containing pre-trained models")
    parser.add_argument('--model_name', default='pilotnet', help="Name of model" )
    # parser.add_argument('--res_path', default='Result_Model_3.csv', help="Path(+filename) to store the results" )
    parser.add_argument('--eval_base', type=bool, default=False, help="If set to True, it will calculate accuracy, size and inference time for original model.")
    parser.add_argument("--tech", action='append', default=[], help="Techniques to apply for model compression. Options are: \n"+
                               "'dynamic_quan', 'float16_quan', 'full_int_quan', and 'all' .")
    
    args = parser.parse_args()
    return args

# def tensorrt():
    # model = ResNet50()
    # model.save('resnet50_saved_model') 
    # model_path = 'resnet50_saved_model'
    # print('Converting to TF-TRT FP32...')
    # conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP32,
    #                                                             max_workspace_size_bytes=8000000000)

    # converter = trt.TrtGraphConverterV2(input_saved_model_dir= model_path,
    #                                     conversion_params=conversion_params)
    # converter.convert()
    # converter.save(output_saved_model_dir='pilotnet_TFTRT_FP32')
    # print('Done Converting to TF-TRT FP32')


if __name__ == '__main__':

    args = parse_args()
    # directory to save optimized models
    tflite_models_dir = pathlib.Path("tflite_models/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    # load datasets
    train_set, valid_set, images_val = load_data(args)

    results = []

    if args.eval_base:
        res = convert_baseline(args.model_path, args.model_name, tflite_models_dir, valid_set, images_val, args.batch_size) 
        results.append(("Baseline",) + res)
    if "dynamic_quan" in args.tech or 'all' in args.tech : 
        res = dynamic_range_quantization(args.model_path, args.model_name, tflite_models_dir, valid_set, images_val, args.batch_size)
        results.append(("Dynamic Range Q",) + res)

    df = pd.DataFrame(results)
    df.columns = ["Method", "Model size (MB)", "MSE", "Inference time (s)"]
    df.to_csv("model_evaluation.csv", index=False)