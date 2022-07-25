import argparse
import time
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import pathlib
import argparse
from PilotNet.utils.dataset import get_augmentations, DatasetSequence
from PilotNet.utils.processing import process_dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import copy
import os




def converter_and_save(path, precision, save_path, args, calibration_input_fn=None):
    """Loads a saved model using a TF-TRT converter, and returns the converter
    """

    params = copy.deepcopy(trt.DEFAULT_TRT_CONVERSION_PARAMS)
    if precision == 'int8':
        precision_mode = trt.TrtPrecisionMode.INT8
    elif precision == 'fp16':
        precision_mode = trt.TrtPrecisionMode.FP16
    else:
        precision_mode = trt.TrtPrecisionMode.FP32

    params = params._replace(
        precision_mode=precision_mode,
        max_workspace_size_bytes=8 * (10**9),  # bytes -> 10^9 = Gb
        maximum_cached_engines=100,
        minimum_segment_size=3,
        allow_build_at_runtime=True
    )

    import pprint
    print("%" * 85)
    pprint.pprint(params)
    print("%" * 85)

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=str(path),
        conversion_params=params,
        # use_dynamic_shape=True, dynamic_shape_profile_strategy='Optimal',
    )

    if precision == 'int8':
        converter.convert(calibration_input_fn=calibration_input_fn)
    else:
        converter.convert() # convert to tensorrt
    print("Conversion complete!!")

    tftrt_model_file = save_path/f"{args.model_name}_tftrt_{precision}"
    converter.save(str(tftrt_model_file))
    print("TF-TRT model saved!")

    return tftrt_model_file


def measure_inference_time(model, images_val):
    # measure average inference time
    # preparing model for inference
    infer = model.signatures['serving_default']
    output_tensorname = list(infer.structured_outputs.keys())[0]

    inf_time = []

    # warm-up
    r_idx = np.random.randint(0, len(images_val), 50)
    for i in r_idx:
        test_image = np.expand_dims(images_val[i], axis=0)
        test_image = tf.convert_to_tensor(test_image, dtype=tf.float32)
        output = infer(test_image)[output_tensorname]            

    # inference
    r_idx = np.random.randint(0, len(images_val), 1000)
    for i in tqdm(r_idx):
        # Pre-processing: add batch dimension and convert to 'dtype' to match with
        # the model's input data format.
        # Check if the input type is quantized, then rescale input data to uint8
        test_image = np.expand_dims(images_val[i], axis=0)
        # test_image = tf.constant(test_image)
        test_image = tf.convert_to_tensor(test_image, dtype=tf.float32)

        start_t = time.time()
        # Run inference.
        output = infer(test_image)[output_tensorname]        
        inf_time.append(time.time() - start_t)
        
    return np.mean(inf_time)


def measure_mse(model, valid_set):
    # put to interpreter for inference

    # preparing model for inference
    infer = model.signatures['serving_default']
    output_tensorname = list(infer.structured_outputs.keys())[0]
    
    metric = 0.0
    for idx in tqdm(range((len(valid_set)-1))):
        test_images, test_labels = valid_set[idx]
        
        # Pre-processing
        test_images = tf.convert_to_tensor(test_images, dtype=tf.float32)
        # Run inference
        output = infer(test_images)[output_tensorname]        
        # Post-processing
        metric += np.mean(tf.keras.losses.mse(test_labels, output).numpy())

    return metric/(len(valid_set)-1)



def evaluate_model(tftrt_model_file, valid_set, images_val):
    '''
    Calculate accuracy, model size and inference time for the given model.
    Args:
        tftrt_model_file: path to saved tflite model
        valid_set: dataset to do test for accuracy
    return:
        accuracy, model_size, inf_time
    '''
    print("Model evaluation started ......")
    # load model
    tftrt_model = tf.saved_model.load(str(tftrt_model_file))
    # model size
    model_size = os.path.getsize(tftrt_model_file) / float(2**20)
    # model perf
    mse = measure_mse(tftrt_model, valid_set)
    # model inf time
    inf_time = measure_inference_time(tftrt_model, images_val)

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
    
    images_train, annotations_train, images_val, annotations_val = process_dataset(args.data_dir[0], type_image,
                                                                                    data_type, img_shape, optimize_mode=True)
    AUGMENTATIONS_TRAIN, AUGMENTATIONS_TEST = get_augmentations(args.data_augs)
    # Training data
    train_gen = DatasetSequence(images_train, annotations_train, args.batch_size,
                                augmentations=AUGMENTATIONS_TRAIN)

    # Validation data
    valid_gen = DatasetSequence(images_val, annotations_val, args.batch_size,
                                augmentations=AUGMENTATIONS_TEST)

    return train_gen, valid_gen, np.array(images_train), np.array(annotations_train), np.array(images_val), np.array(annotations_val)


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
    parser.add_argument('--precision', action='append', type=str, default=[], choices=['int8', 'fp16', 'fp32', 'all'], help='Precisions to apply for model optimization')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    # directory to save optimized models
    tftrt_models = pathlib.Path("tf_trt_models/")
    tftrt_models.mkdir(exist_ok=True, parents=True)

    # save model in `SavedModel` tensorflow format from keras
    if '.h5' in args.model_path:
        print("Saving Keras into TF SavedModel format ....\n")
        model = tf.keras.models.load_model(args.model_path)
        args.model_path = tftrt_models/f"{args.model_name}_saved_model"
        model.save(args.model_path)

    # load datasets
    train_set, valid_set, images_train, annotations_train, images_val, annotations_val = load_data(args)

    def calibration_input_fn():
        for img, label in valid_set:
            yield (img,)

    results = []

    if args.eval_base:
        res = evaluate_model(args.model_path, valid_set, images_val)
        results.append( ("Baseline",) + res)
        print(res)

    if "fp32" in args.precision or 'all' in args.precision:
        # convert and save to TensorRT
        tftrt_model_file = converter_and_save(args.model_path, "fp32", tftrt_models, args)
        # evaluation model
        res = evaluate_model(tftrt_model_file, valid_set, images_val)
        results.append( ("Precision fp32",) + res)
        print(res)

    if "fp16" in args.precision or 'all' in args.precision:
        # convert and save to TensorRT
        tftrt_model_file = converter_and_save(args.model_path, "fp16", tftrt_models, args)
        # evaluation model
        res = evaluate_model(tftrt_model_file, valid_set, images_val)
        results.append( ("Precision fp16",) + res)
        print(res)

    if "int8" in args.precision or 'all' in args.precision:
        # convert and save to TensorRT
        tftrt_model_file = converter_and_save(args.model_path, "int8", tftrt_models, args, calibration_input_fn=calibration_input_fn)
        # evaluation model
        res = evaluate_model(tftrt_model_file, valid_set, images_val)
        results.append( ("Precision int8",) + res)
        print(res)


    df = pd.DataFrame(results)
    df.columns = ["Method", "Model size (MB)", "MSE", "Inference time (s)"] 
    df.to_csv("model_evaluation.csv", index=False)