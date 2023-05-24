# Re-run after Kernel restart
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import datetime
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
#from utils.dataset import get_augmentations, DatasetSequence
from utils.carla_dataset import get_augmentations, DatasetSequence
#from utils.processing import process_dataset
from utils.processing_carla_tf_lite import process_dataset
from tqdm import tqdm
import tensorflow_model_optimization as tfmot
from tensorflow.keras.optimizers import Adam

def measure_inference_time(tflite_model, images_val):
    # measure average inference time
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    inf_time = []
    r_idx = np.random.randint(0, len(images_val), 1000)
    for i in tqdm(r_idx):
        # Pre-processing: add batch dimension and convert to 'dtype' to match with
        # the model's input data format.
        # Check if the input type is quantized, then rescale input data to uint8
        test_image = np.expand_dims(images_val[i], axis=0)
        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            test_image = test_image / input_scale + input_zero_point
        
        interpreter.set_tensor(input_details["index"], test_image.astype(input_details["dtype"]))

        start_t = time.time()
        # Run inference.
        interpreter.invoke()
        # pred = tflite_model.predict(img, verbose=0)
        inf_time.append(time.time() - start_t)
        # Post-processing
        output = interpreter.get_tensor(output_details["index"])
        
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
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    metric = 0.0
    for idx in tqdm(range((len(valid_set)-1))):
        test_images, test_labels = valid_set[idx]
        
        # Pre-processing
        # Check if the input type is quantized, then rescale input data to uint8
        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            test_images = test_images / input_scale + input_zero_point
        
        interpreter.set_tensor(input_details["index"], test_images.astype(input_details["dtype"]))
        # Run inference.
        interpreter.invoke()
        # Post-processing
        output = interpreter.get_tensor(output_details["index"])
        if output_details['dtype'] == np.uint8:
            output_scale, input_zero_point = output_details["quantization"]
            output = output.astype(np.float32)
            output = (output - input_zero_point) * output_scale
            test_labels = test_labels.astype(np.float32)
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
    print(model.summary())
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
    print()
    print("********* Start Dynamic range Quantization ***********")
    # Post-training dynamic range quantization
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_model_quant_file = tflite_models_dir/f"{model_name}_dynamic_quant.tflite"
    tflite_model_quant_file.write_bytes(tflite_model) # save model
    
    model_size, mse, inf_time = evaluate_model(tflite_model_quant_file, tflite_model, valid_set, images_val, batch_size)
    print("********** Dynamic range Q stats **********")
    print("Model size (MB):", model_size)
    print("MSE:", mse)
    print("Inference time (s):", inf_time)
    return model_size, mse, inf_time


def integer_only_quantization(model_path, model_name, tflite_models_dir, valid_set, images_val, batch_size):
    print()
    print("********* Start Integer Quantization ***********")
    def representative_data_gen():
        for input_value in tf.data.Dataset.from_tensor_slices(np.array(images_val, dtype=np.float32)).batch(1).take(10000):
            yield [input_value]

    # Post-training integer only quantization
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()

    tflite_model_quant_file = tflite_models_dir/f"{model_name}_int_quant.tflite"
    tflite_model_quant_file.write_bytes(tflite_model) # save model
    
    model_size, mse, inf_time = evaluate_model(tflite_model_quant_file, tflite_model, valid_set, images_val, batch_size)
    print("********** Integer only Q stats **********")
    print("Model size (MB):", model_size)
    print("MSE:", mse)
    print("Inference time (s):", inf_time)
    return model_size, mse, inf_time


def integer_float_quantization(model_path, model_name, tflite_models_dir, valid_set, images_val, batch_size):
    print()
    print("********* Start Integer (float fallback) Quantization ***********")
    print(images_val.shape)
    print('*************')
    def representative_data_gen():
        for input_value in tf.data.Dataset.from_tensor_slices(np.array(images_val, dtype=np.float32)).batch(1).take(10000):
            yield [input_value]

    # Post-training integer only quantization
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    input_type = interpreter.get_input_details()[0]['dtype']
    print('input: ', input_type)
    output_type = interpreter.get_output_details()[0]['dtype']
    print('output: ', output_type)

    tflite_model_quant_file = tflite_models_dir/f"{model_name}_intflt_quant.tflite"
    tflite_model_quant_file.write_bytes(tflite_model) # save model
    
    model_size, mse, inf_time = evaluate_model(tflite_model_quant_file, tflite_model, valid_set, images_val, batch_size)
    print("********** Integer (float fallback) Q stats **********")
    print("Model size (MB):", model_size)
    print("MSE:", mse)
    print("Inference time (s):", inf_time)
    return model_size, mse, inf_time


def float16_quantization(model_path, model_name, tflite_models_dir, valid_set, images_val, batch_size):
    print()
    print("********* Start Float16 Quantization ***********")
    # Post-training dynamic range quantization
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    tflite_model_quant_file = tflite_models_dir/f"{model_name}_float16_quant.tflite"
    tflite_model_quant_file.write_bytes(tflite_model) # save model
    
    model_size, mse, inf_time = evaluate_model(tflite_model_quant_file, tflite_model, valid_set, images_val, batch_size)
    print("********** Float16 Q stats **********")
    print("Model size (MB):", model_size)
    print("MSE:", mse)
    print("Inference time (s):", inf_time)
    return model_size, mse, inf_time


def quantization_aware_train(model_path, model_name, tflite_models_dir, valid_set, images_val, args, images_train, annotations_train):
    print()
    print("********* Start Quantization Aware Training ***********")

    # https://www.tensorflow.org/model_optimization/guide/quantization/training_example#clone_and_fine-tune_pre-trained_model_with_quantization_aware_training
    # https://www.tensorflow.org/model_optimization/guide/quantization/training_comprehensive_guide.md

    model = tf.keras.models.load_model(model_path) # load original model
    # quantize_model = tfmot.quantization.keras.quantize_model
    # q_aware_model = quantize_model(model)

    ### Quantize model - dense and conv2d (batchnorm not support)
    # Helper function uses `quantize_annotate_layer` to annotate that only the specified layers be Q 
    def apply_quantization_to_layers(layer):
        if isinstance(layer, tf.keras.layers.Dense):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        if isinstance(layer, tf.keras.layers.Conv2D):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        return layer
    # Use `tf.keras.models.clone_model` to apply `apply_quantization_to_dense` 
    # to the layers of the model.
    annotated_model = tf.keras.models.clone_model(
        model,
        clone_function=apply_quantization_to_layers,
    )
    # Now that the Dense layers are annotated,
    # `quantize_apply` actually makes the model quantization aware.
    q_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    # `quantize_model` requires a recompile.
    q_aware_model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss="mse", metrics=['mse', 'mae'])
    q_aware_model.summary() # every layer has `quant` prefix
    # use subset of data to train; here 1%
    #ridx = np.random.randint(0, len(images_train), int(len(images_train)*0.01))
    ridx = np.random.randint(0, len(images_train), int(len(images_train)*0.1))
    images_train, annotations_train = images_train[ridx], annotations_train[ridx]
    # fine-tune pre-trained model with quantization aware training
    #q_aware_model.fit(images_train, annotations_train, batch_size=args.batch_size, epochs=2, validation_split=0.1)
    q_aware_model.fit(images_train, annotations_train, batch_size=args.batch_size, epochs=20, validation_split=0.1)
    
    # Create quantized model for TFLite backend - quantized model with int8 weights and uint8 activations
    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_model_quant_file = tflite_models_dir/f"{model_name}_quant_aware.tflite"
    tflite_model_quant_file.write_bytes(tflite_model) # save model
    
    model_size, mse, inf_time = evaluate_model(tflite_model_quant_file, tflite_model, valid_set, images_val, args.batch_size)
    print("********** Quantization Aware Training stats **********")
    print("Model size (MB):", model_size)
    print("MSE:", mse)
    print("Inference time (s):", inf_time)
    return model_size, mse, inf_time


def weight_pruning(model_path, model_name, tflite_models_dir, valid_set, images_val, args, images_train, annotations_train, apply_quan=False):
    print()
    print("********* Start (random sparse) Weight pruning ***********")

    model = tf.keras.models.load_model(model_path) # load original model
    
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    # Compute end step to finish pruning after 2 epochs.
    epochs = 2
    validation_split = 0.1 # 10% of training set will be used for validation set. 
    num_images = len(images_train) * (1 - validation_split)
    end_step = np.ceil(num_images / args.batch_size).astype(np.int32) * epochs
    # Define model for pruning.
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.30,
                                                                final_sparsity=0.40,
                                                                begin_step=0,
                                                                end_step=end_step)
    }
    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    # requires a recompile.
    model_for_pruning.compile(optimizer=Adam(learning_rate=args.learning_rate), loss="mse", metrics=['mse', 'mae'])
    model_for_pruning.summary() # every layer has `quant` prefix
    
    log_dir = "logs/fit_prune/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # The logs show the progression of sparsity on a per-layer basis.
    # tensorboard --logdir={logdir}
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir),
    ]
    # fine-tune pre-trained model with pruning
    model_for_pruning.fit(images_train, annotations_train, batch_size=args.batch_size, epochs=epochs, 
                        validation_split=validation_split, callbacks=callbacks)
    # removes every tf.Variable that pruning only needs during training, 
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)


    # apply post-training quantization to the pruned model for additional benefits.
    if apply_quan:
        converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        tflite_model_file = tflite_models_dir/f"{model_name}_pruned_quan.tflite"
        tflite_model_file.write_bytes(tflite_model) # save model
        mssg = "********** Weight Pruning + Quantization stats **********" 
    else: # convert only pruned model
        converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
        tflite_model = converter.convert()
        tflite_model_file = tflite_models_dir/f"{model_name}_pruned.tflite"
        tflite_model_file.write_bytes(tflite_model) # save model
        mssg = "********** Weight Pruning stats **********"

    model_size, mse, inf_time = evaluate_model(tflite_model_file, tflite_model, valid_set, images_val, args.batch_size)
    print(mssg)
    print("Model size (MB):", model_size)
    print("MSE:", mse)
    print("Inference time (s):", inf_time)
    return model_size, mse, inf_time


def CQAT(model_path, model_name, tflite_models_dir, valid_set, images_val, args, images_train, annotations_train):
    '''
        Cluster preserving quantization aware training (CQAT)
    '''
    print()
    print("********* Start Cluster preserving Quantization Aware Training ***********")

    model = tf.keras.models.load_model(model_path) # load original model

    ## Clustering
    cluster_weights = tfmot.clustering.keras.cluster_weights
    CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

    clustering_params = {
    'number_of_clusters': 8,
    'cluster_centroids_init': CentroidInitialization.KMEANS_PLUS_PLUS,
    'cluster_per_channel': True,
    }
    clustered_model = cluster_weights(model, **clustering_params)

    # Use smaller learning rate for fine-tuning
    clustered_model.compile(optimizer=Adam(learning_rate=args.learning_rate/100), loss="mse", metrics=['mse', 'mae'])
    clustered_model.summary()
    clustered_model.fit(images_train, annotations_train, batch_size=args.batch_size, epochs=3, validation_split=0.1)
    
    stripped_clustered_model = tfmot.clustering.keras.strip_clustering(clustered_model)

    # CQAT
    ### Quantize model - dense and conv2d (batchnorm not support)
    # Helper function uses `quantize_annotate_layer` to annotate that only the specified layers be Q 
    def apply_quantization_to_layers(layer):
        if isinstance(layer, tf.keras.layers.Dense):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        if isinstance(layer, tf.keras.layers.Conv2D):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        return layer
    # Use `tf.keras.models.clone_model` to apply `apply_quantization_to_dense` 
    # to the layers of the model.
    annotated_model = tf.keras.models.clone_model(
        stripped_clustered_model,
        clone_function=apply_quantization_to_layers,
    )
    # Now that the Dense layers are annotated,
    # `quantize_apply` actually makes the model quantization aware.
    cqat_model = tfmot.quantization.keras.quantize_apply(annotated_model,
                tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme())

    cqat_model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss="mse", metrics=['mse', 'mae'])
    print('Train cqat model:')
    cqat_model.fit(images_train, annotations_train, batch_size=args.batch_size, epochs=1, validation_split=0.1)
    
    # Create quantized model for TFLite backend - quantized model with int8 weights and uint8 activations
    converter = tf.lite.TFLiteConverter.from_keras_model(cqat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_model_cqat_file = tflite_models_dir/f"{model_name}_cqat_model.tflite"
    tflite_model_cqat_file.write_bytes(tflite_model) # save model
    
    model_size, mse, inf_time = evaluate_model(tflite_model_cqat_file, tflite_model, valid_set, images_val, args.batch_size)
    print("********** Cluster preserving Quantization Aware Training stats **********")
    print("Model size (MB):", model_size)
    print("MSE:", mse)
    print("Inference time (s):", inf_time)
    return model_size, mse, inf_time


def PQAT(model_path, model_name, tflite_models_dir, valid_set, images_val, args, images_train, annotations_train):
    '''
        Pruning preserving quantization aware training (PQAT)
    '''
    print()
    print("********* Start Pruning preserving Quantization Aware Training ***********")

    model = tf.keras.models.load_model(model_path) # load original model

    ## Pruning
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, begin_step=0, frequency=100)
    }
    callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep()
    ]

    pruned_model = prune_low_magnitude(model, **pruning_params)

    # Use smaller learning rate for fine-tuning
    pruned_model.compile(optimizer=Adam(learning_rate=args.learning_rate/100), loss="mse", metrics=['mse', 'mae'])
    pruned_model.summary()
    pruned_model.fit(images_train, annotations_train, batch_size=args.batch_size, epochs=5, validation_split=0.1, callbacks=callbacks)
    
    stripped_pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

    # PQAT
    # quant_aware_annotate_model = tfmot.quantization.keras.quantize_annotate_model(
    #             stripped_pruned_model)
    # pqat_model = tfmot.quantization.keras.quantize_apply(
    #           quant_aware_annotate_model,
    #           tfmot.experimental.combine.Default8BitPrunePreserveQuantizeScheme())

    ### Quantize model - dense and conv2d (batchnorm not support)
    # Helper function uses `quantize_annotate_layer` to annotate that only the specified layers be Q 
    def apply_quantization_to_layers(layer):
        if isinstance(layer, tf.keras.layers.Dense):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        if isinstance(layer, tf.keras.layers.Conv2D):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        return layer
    # Use `tf.keras.models.clone_model` to apply `apply_quantization_to_dense` 
    # to the layers of the model.
    annotated_model = tf.keras.models.clone_model(
        stripped_pruned_model,
        clone_function=apply_quantization_to_layers,
    )
    # Now that the Dense layers are annotated,
    # `quantize_apply` actually makes the model quantization aware.
    pqat_model = tfmot.quantization.keras.quantize_apply(annotated_model,
              tfmot.experimental.combine.Default8BitPrunePreserveQuantizeScheme())
    
    pqat_model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss="mse", metrics=['mse', 'mae'])
    print('Train pqat model:')
    pqat_model.fit(images_train, annotations_train, batch_size=args.batch_size, epochs=3, validation_split=0.1)

    # Create quantized model for TFLite backend - quantized model with int8 weights and uint8 activations
    converter = tf.lite.TFLiteConverter.from_keras_model(pqat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_model_pqat_file = tflite_models_dir/f"{model_name}_pqat_model.tflite"
    tflite_model_pqat_file.write_bytes(tflite_model) # save model
    
    model_size, mse, inf_time = evaluate_model(tflite_model_pqat_file, tflite_model, valid_set, images_val, args.batch_size)
    print("********** Pruning preserving Quantization Aware Training stats **********")
    print("Model size (MB):", model_size)
    print("MSE:", mse)
    print("Inference time (s):", inf_time)
    return model_size, mse, inf_time


def PCQAT(model_path, model_name, tflite_models_dir, valid_set, images_val, args, images_train, annotations_train):
    '''
        Sparsity and cluster preserving quantization aware training (PCQAT)
    '''
    print()
    print("********* Start Sparsity and cluster preserving quantization aware training ***********")

    model = tf.keras.models.load_model(model_path) # load original model

    ## Pruning
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, begin_step=0, frequency=100)
    }
    callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep()
    ]

    pruned_model = prune_low_magnitude(model, **pruning_params)

    # Use smaller learning rate for fine-tuning
    pruned_model.compile(optimizer=Adam(learning_rate=args.learning_rate/100), loss="mse", metrics=['mse', 'mae'])
    pruned_model.fit(images_train, annotations_train, batch_size=args.batch_size, epochs=5, validation_split=0.1, callbacks=callbacks)
    
    stripped_pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

    ## sparsity preserving clustering
    from tensorflow_model_optimization.python.core.clustering.keras.experimental import (
    cluster,
    )

    cluster_weights = tfmot.clustering.keras.cluster_weights
    CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
    cluster_weights = cluster.cluster_weights

    clustering_params = {
    'number_of_clusters': 8,
    'cluster_centroids_init': CentroidInitialization.KMEANS_PLUS_PLUS,
    'preserve_sparsity': True
    }

    sparsity_clustered_model = cluster_weights(stripped_pruned_model, **clustering_params)
    # training
    sparsity_clustered_model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss="mse", metrics=['mse', 'mae'])
    sparsity_clustered_model.fit(images_train, annotations_train, batch_size=args.batch_size, epochs=5, validation_split=0.1)
    
    stripped_clustered_model = tfmot.clustering.keras.strip_clustering(sparsity_clustered_model)

    ## PCQAT
    # quant_aware_annotate_model = tfmot.quantization.keras.quantize_annotate_model(
    #             stripped_clustered_model)
    # pcqat_model = tfmot.quantization.keras.quantize_apply(
    #             quant_aware_annotate_model,
    #             tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme(preserve_sparsity=True))

    ### Quantize model - dense and conv2d (batchnorm not support)
    # Helper function uses `quantize_annotate_layer` to annotate that only the specified layers be Q 
    def apply_quantization_to_layers(layer):
        if isinstance(layer, tf.keras.layers.Dense):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        if isinstance(layer, tf.keras.layers.Conv2D):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        return layer
    # Use `tf.keras.models.clone_model` to apply `apply_quantization_to_dense` 
    # to the layers of the model.
    annotated_model = tf.keras.models.clone_model(
        stripped_clustered_model,
        clone_function=apply_quantization_to_layers,
    )
    # Now that the Dense layers are annotated,
    # `quantize_apply` actually makes the model quantization aware.
    pcqat_model = tfmot.quantization.keras.quantize_apply(annotated_model,
                tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme(preserve_sparsity=True))

    pcqat_model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss="mse", metrics=['mse', 'mae'])
    print('Train pcqat model:')
    pcqat_model.fit(images_train, annotations_train, batch_size=args.batch_size, epochs=3, validation_split=0.1)
    
    # Create quantized model for TFLite backend - quantized model with int8 weights and uint8 activations
    converter = tf.lite.TFLiteConverter.from_keras_model(pcqat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_model_pcqat_file = tflite_models_dir/f"{model_name}_pcqat_model.tflite"
    tflite_model_pcqat_file.write_bytes(tflite_model) # save model
    
    model_size, mse, inf_time = evaluate_model(tflite_model_pcqat_file, tflite_model, valid_set, images_val, args.batch_size)
    print("********** Sparsity and cluster preserving quantization aware training (PCQAT) stats **********")
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
    
    images_train, annotations_train, images_val, annotations_val = process_dataset(args.data_dir[0], type_image,
                                                                                    data_type, img_shape, optimize_mode=True)
    AUGMENTATIONS_TRAIN, AUGMENTATIONS_TEST = get_augmentations(args.data_augs)
    # Training data
    train_gen = DatasetSequence(images_train, annotations_train, args.batch_size,
                                augmentations=AUGMENTATIONS_TRAIN)

    # Validation data
    valid_gen = DatasetSequence(images_val, annotations_val, args.batch_size,
                                augmentations=AUGMENTATIONS_TEST)

    
    new_images_train = []
    for image in images_train:
        augmented_image = AUGMENTATIONS_TEST(image=image)['image']
        new_images_train.append(augmented_image)
    images_train = new_images_train

    new_images_val = []
    for image in images_val:
        augmented_image = AUGMENTATIONS_TEST(image=image)['image']
        new_images_val.append(augmented_image)
    images_val = new_images_val
    

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
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    # parser.add_argument('--res_path', default='Result_Model_3.csv', help="Path(+filename) to store the results" )
    parser.add_argument('--eval_base', type=bool, default=False, help="If set to True, it will calculate accuracy, size and inference time for original model.")
    parser.add_argument("--tech", action='append', default=[], help="Techniques to apply for model compression. Options are: \n"+
                               "'dynamic_quan', 'int_quan', 'int_flt_quan', 'float16_quan', 'quan_aware', 'prune', 'prune_quan', 'clust_qat', 'prune_qat', 'prune_clust_qat' and 'all' .")
    
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
    train_set, valid_set, images_train, annotations_train, images_val, annotations_val = load_data(args)

    results = []

    if args.eval_base:
        res = convert_baseline(args.model_path, args.model_name, tflite_models_dir, valid_set, images_val, args.batch_size) 
        results.append(("Baseline",) + res)
    if "dynamic_quan" in args.tech or 'all' in args.tech : 
        res = dynamic_range_quantization(args.model_path, args.model_name, tflite_models_dir, valid_set, images_val, args.batch_size)
        results.append(("Dynamic Range Q",) + res)
    if "int_quan" in args.tech or 'all' in args.tech : 
        res = integer_only_quantization(args.model_path, args.model_name, tflite_models_dir, valid_set, images_val, args.batch_size)
        results.append(("Integer only Q",) + res)
    if "int_flt_quan" in args.tech or 'all' in args.tech : 
        res = integer_float_quantization(args.model_path, args.model_name, tflite_models_dir, valid_set, images_val, args.batch_size)
        results.append(("Integer (float fallback) Q",) + res)
    if "float16_quan" in args.tech or 'all' in args.tech : 
        res = float16_quantization(args.model_path, args.model_name, tflite_models_dir, valid_set, images_val, args.batch_size)
        results.append(("Float16 Q",) + res)
    if "quan_aware" in args.tech or 'all' in args.tech : 
        res = quantization_aware_train(args.model_path, args.model_name, tflite_models_dir, valid_set, 
                                    images_val, args, images_train, annotations_train)
        results.append(("Q aware training",) + res)
    if "prune" in args.tech or 'all' in args.tech : 
        res = weight_pruning(args.model_path, args.model_name, tflite_models_dir, valid_set, 
                            images_val, args, images_train, annotations_train, apply_quan=False)
        results.append(("Weight pruning",) + res)
    if "prune_quan" in args.tech or 'all' in args.tech : 
        res = weight_pruning(args.model_path, args.model_name, tflite_models_dir, valid_set, 
                            images_val, args, images_train, annotations_train, apply_quan=True)
        results.append(("Weight pruning + Q",) + res)
    if "clust_qat" in args.tech or 'all' in args.tech : 
        res = CQAT(args.model_path, args.model_name, tflite_models_dir, valid_set, 
                                    images_val, args, images_train, annotations_train)
        results.append(("CQAT",) + res)
    if "prune_qat" in args.tech or 'all' in args.tech : 
        res = PQAT(args.model_path, args.model_name, tflite_models_dir, valid_set, 
                                    images_val, args, images_train, annotations_train)
        results.append(("PQAT",) + res)
    if "prune_clust_qat" in args.tech or 'all' in args.tech : 
        res = PCQAT(args.model_path, args.model_name, tflite_models_dir, valid_set, 
                                    images_val, args, images_train, annotations_train)
        results.append(("PCQAT",) + res)

        
    df = pd.DataFrame(results)
    df.columns = ["Method", "Model size (MB)", "MSE", "Inference time (s)"]
    df.to_csv("model_evaluation.csv", index=False)
