import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import os
from utils.processing import *
from utils.pilot_net_dataset import PilotNetDataset
from utils.pilotnet import PilotNet
from utils.transform_helper import createTransform
import argparse
import json
import numpy as np
from copy import deepcopy
import pandas as pd


# def measure_inference_time(tflite_model, images_val):
#     # measure average inference time
#     interpreter = tf.lite.Interpreter(model_content=tflite_model)
#     interpreter.allocate_tensors()
#     input_details = interpreter.get_input_details()[0]
#     output_details = interpreter.get_output_details()[0]
    
#     inf_time = []
#     r_idx = np.random.randint(0, len(images_val), 1000)
#     for i in tqdm(r_idx):
#         # Pre-processing: add batch dimension and convert to 'dtype' to match with
#         # the model's input data format.
#         # Check if the input type is quantized, then rescale input data to uint8
#         test_image = np.expand_dims(images_val[i], axis=0)
#         if input_details['dtype'] == np.uint8:
#             input_scale, input_zero_point = input_details["quantization"]
#             test_image = test_image / input_scale + input_zero_point
        
#         interpreter.set_tensor(input_details["index"], test_image.astype(input_details["dtype"]))

#         start_t = time.time()
#         # Run inference.
#         interpreter.invoke()
#         # pred = tflite_model.predict(img, verbose=0)
#         inf_time.append(time.time() - start_t)
#         # Post-processing
#         output = interpreter.get_tensor(output_details["index"])
        
#     return np.mean(inf_time)

def measure_mse(quant_model, val_loader):
    # measure average inference time
    criterion = nn.MSELoss()

    quant_model.eval()
    with torch.no_grad():
        total_loss = 0
        for images, labels in val_loader:
            images = FLOAT(images).to(device)
            labels = FLOAT(labels.float()).to(device)
            outputs =  quant_model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
        
        MSE = total_loss/len(val_loader)
        print('Average MSE of the model on the test images: {} %'.format( MSE))

    return MSE

def evaluate_model(qmodel_path, quant_model, val_loader):
    '''
    Calculate accuracy, model size and inference time for the given model.
    Args:
        model_path: path to saved tflite model
        tflite_model: converted model instance (to tflite)
        valid_set: dataset to do test for accuracy
    return:
        accuracy, model_size, inf_time
    '''
    model_size = os.path.getsize(qmodel_path) / float(2**20)

    mse = measure_mse(quant_model, val_loader)
    
    # inf_time = measure_inference_time(quant_model, val_loader)
    inf_time = -1
    return model_size, mse, inf_time


def dynamic_quantization(model, model_save_dir, val_loader):
    print()
    print("********* Start Dynamic range Quantization ***********")
    # Post-training dynamic range quantization

    quant_model = quantize_dynamic(
                    model=model, qconfig_spec={nn.Linear}, dtype=torch.qint8, inplace=False #can add nn.LSTM in qconfig_spec for LSTM layers
                    )

    qmodel_path = model_save_dir + '/dynamic_quan.ckpt'
    torch.save(quant_model.state_dict(), qmodel_path)
    
    model_size, mse, inf_time = evaluate_model(qmodel_path, quant_model, val_loader)
    print("********** Dynamic range Q stats **********")
    print("Model size (MB):", model_size)
    print("MSE:", mse)
    print("Inference time (s):", inf_time)
    return model_size, mse, inf_time


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", action='append', help="Directory to find Data")
    parser.add_argument("--val_dir", action='append', help="Directory to find Data")
    parser.add_argument("--preprocess", action='append', default=None, help="preprocessing information: choose from crop/nocrop and normal/extreme")
    parser.add_argument("--base_dir", type=str, default='optimized_models', help="Directory to save everything")
    parser.add_argument("--model_dir", type=str, default='trained_models/', help="Directory to trained models")
    parser.add_argument("--data_augs", action='append', type=str, default=None, help="Data Augmentations")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of Epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Policy Net")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument('--eval_base', type=bool, default=False, help="If set to True, it will calculate accuracy, size and inference time for original model.")
    parser.add_argument("--tech", action='append', default=[], help="Techniques to apply for model compression. Options are: \n"+
                               "'dynamic_quan', 'int_quan', 'int_flt_quan', 'float16_quan', 'quan_aware', 'prune', 'prune_quan', 'clust_qat', 'prune_qat', 'prune_clust_qat' and 'all' .")
    

    args = parser.parse_args()
    return args

if __name__=="__main__":

    args = parse_args()

    exp_setup = vars(args)

    # Base Directory
    # path_to_data = args.data_dir
    base_dir = './experiments/'+ args.base_dir + '/'
    model_save_dir = base_dir + 'trained_models'
    log_dir = base_dir + 'log'

    check_path(base_dir)
    check_path(log_dir)
    check_path(model_save_dir)

    with open(base_dir+'args.json', 'w') as fp:
        json.dump(exp_setup, fp)

    # Hyperparameters
    augmentations = args.data_augs
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    # Device Selection (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FLOAT = torch.FloatTensor

    # Tensorboard Initialization
    writer = SummaryWriter(log_dir)

    # Define data transformations
    transformations_train = createTransform(augmentations)
    transformations_val = createTransform(['None']) # only need Normalize()
    # Load data
    train_set = PilotNetDataset(args.train_dir, transformations_train, preprocessing=args.preprocess)
    val_set = PilotNetDataset(args.val_dir, transformations_val, preprocessing=args.preprocess)
    # create DataLoader for batching
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Load Model
    pilotModel = PilotNet(val_set.image_shape, val_set.num_labels).to(device)
    pilotModel.load_state_dict(torch.load(args.model_dir))
    
    results = []
    if "dynamic_quan" in args.tech or 'all' in args.tech : 
        res = dynamic_quantization(pilotModel, model_save_dir, val_loader)
        results.append(("Dynamic Range Q",) + res)
    
    df = pd.DataFrame(results)
    df.columns = ["Method", "Model size (MB)", "MSE", "Inference time (s)"]
    df.to_csv("model_evaluation.csv", index=False)