import os
import time
import torch
import torch_tensorrt
import torchvision

import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms

from utils.processing_carla import *
from utils.pilot_net_dataset import PilotNetDataset, PilotNetDatasetTest
from utils.transform_helper import createTransform
from utils.pilotnet import PilotNet

FLOAT = torch.FloatTensor

image_shape = np.array([200,66, 3])
device = 'cuda'
model_dir = '/docker-tensorrt/pilot_net_model_best_123.pth'

pilotModel = PilotNet(image_shape, 3).eval().to(device)
pilotModel.load_state_dict(torch.load(model_dir))

inputs = [torch.rand(1, 3, 200, 66)] # Input should be a tensor

print(inputs[0].shape)

augmentations = 'all'

path_to_data = [
    '/docker-tensorrt/carla_dataset_previous_v/carla_dataset_test_31_10_anticlockwise_town_01_previous_v/',
    '/docker-tensorrt/carla_dataset_previous_v/carla_dataset_test_31_10_clockwise_town_01_previous_v/',
    '/docker-tensorrt/carla_dataset_previous_v/carla_dataset_test_04_11_clockwise_town_01_previous_v_extreme/',
    '/docker-tensorrt/carla_dataset_previous_v/carla_dataset_test_04_11_clockwise_town_01_previous_v_extreme/',
    '/docker-tensorrt/carla_dataset_previous_v/carla_dataset_test_04_11_anticlockwise_town_03_previous_v/',
    '/docker-tensorrt/carla_dataset_previous_v/carla_dataset_test_04_11_clockwise_town_03_previous_v/',
    '/docker-tensorrt/carla_dataset_previous_v/carla_dataset_test_04_11_anticlockwise_town_05_previous_v/',
    '/docker-tensorrt/carla_dataset_previous_v/carla_dataset_test_04_11_clockwise_town_05_previous_v/',
    '/docker-tensorrt/carla_dataset_previous_v/carla_dataset_test_04_11_anticlockwise_town_07_previous_v/',
    '/docker-tensorrt/carla_dataset_previous_v/carla_dataset_test_04_11_clockwise_town_07_previous_v/',  
    ]
val_split = 0.3
shuffle_dataset = True 
random_seed = 123
batch_size = 1

# Define data transformations
transformations = createTransform(augmentations)
# Load data
dataset = PilotNetDatasetTest(path_to_data, transformations, preprocessing=['extreme'])

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(val_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_split = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
test_sampler = SubsetRandomSampler(val_split)
testing_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

###################################################

calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
    testing_dataloader,
    cache_file="./calibration.cache",
    use_cache=False,
    algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
    device=torch.device(device),
)

trt_mod = torch_tensorrt.compile(pilotModel, inputs=[torch_tensorrt.Input((1, 3, 200, 66))],
                                    enabled_precisions={torch.half},
                                    calibrator=calibrator,
                                    #device={
                                    #     "device_type": torch_tensorrt.DeviceType.GPU,
                                    #     "gpu_id": 0,
                                    #     "dla_core": 0,
                                    #     "allow_gpu_fallback": False,
                                    #     "disable_tf32": False
                                    # })
)

data = iter(testing_dataloader)
images, _ = next(data)
trt_mod = torch.jit.trace(pilotModel, images.to("cuda"))
torch.jit.save(trt_mod, 'trt_mod.jit.pt')
#torch.jit.save(trt_mod, 'trt_mod.pth')

def measure_inference_time(model, val_set):
    # measure average inference time
    
    # GPU warm-up
    r_idx = np.random.randint(0, len(val_set), 50)
    for i in r_idx:
        image, _ = val_set[i]
        image = torch.unsqueeze(image, 0).to(device)
        _ = model(image) 
    
    # actual inference call
    inf_time = []
    r_idx = np.random.randint(0, len(val_set), 1000)
    for i in tqdm(r_idx):
        # preprocessing
        image, _ = val_set[i]
        image = torch.unsqueeze(image, 0).to(device)
        # Run inference.
        start_t = time.time()
        _ = model(image) 
        inf_time.append(time.time() - start_t)
        
    return np.mean(inf_time)

def measure_mse(model, val_loader):
    criterion = nn.MSELoss()

    model.eval()
    with torch.no_grad():
        total_loss = 0
        for images, labels in tqdm(val_loader):
            images = FLOAT(images).to(device)
            labels = FLOAT(labels.float()).to(device)
            outputs =  model(images).clone().detach().to(dtype=torch.float16)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

        MSE = total_loss/len(val_loader)

    return MSE

def evaluate_model(model_path, opt_model, val_set, val_loader):
    '''
    Calculate accuracy, model size and inference time for the given model.
    Args:
        model_path: path to saved quantized model
        opt_model: converted model instance
        val_set: dataset to use for inference benchmarking
        val_loader: Dataset loader for accuracy test
    return:
        accuracy, model_size, inf_time
    '''
    model_size = os.path.getsize(model_path) / float(2**20)

    mse = measure_mse(opt_model, val_loader)
    
    inf_time = measure_inference_time(opt_model,  val_set)

    return model_size, mse, inf_time

model_size, mse, inf_time = evaluate_model('trt_mod.jit.pt', trt_mod, dataset, testing_dataloader)

print("Model size (MB):", model_size)
print("MSE:", mse)
print("Inference time (s):", inf_time)
