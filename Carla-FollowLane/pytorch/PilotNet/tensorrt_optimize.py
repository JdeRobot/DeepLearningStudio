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

trt_mod = torch_tensorrt.compile(pilotModel, 
                                    #inputs=[torch_tensorrt.Input((1, 3, 200, 66), dtype=torch.half)],
                                    inputs=[torch_tensorrt.Input((1, 3, 200, 66), dtype=torch.float32)],
                                    #enabled_precisions={torch.half},
                                    enabled_precisions={torch.float32},
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
traced_model = torch.jit.trace(pilotModel, images.to("cuda"))
torch.jit.save(traced_model, 'trt_mod.jit.pt')

def benchmark(model, input_shape=(1024, 1, 224, 224), dtype='fp32', nwarmup=50, nruns=10000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype=='fp16':
        input_data = input_data.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                print('Iteration %d/%d, ave batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.size())
    print("Output features size:", features.size())
    print('Average batch time: %.2f ms'%(np.mean(timings)*1000))


benchmark(trt_mod, input_shape=(1, 3, 200, 66), nruns=100)
#benchmark(trt_mod, input_shape=(1, 3, 200, 66), dtype='fp16', nruns=100)

def measure_inference_time(model, val_set, dtype):
    # measure average inference time
    
    # GPU warm-up
    r_idx = np.random.randint(0, len(val_set), 50)
    for i in r_idx:
        image, _ = val_set[i]
        image = torch.unsqueeze(image, 0).to(device)
        if dtype=='fp16':
            image = image.half()
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
        if dtype=='fp16':
            image = image.half()
        _ = model(image) 
        inf_time.append(time.time() - start_t)
        
    return np.mean(inf_time)

def measure_mse(model, val_loader, dtype):
    criterion = nn.MSELoss()

    model.eval()
    with torch.no_grad():
        total_loss = 0
        for images, labels in tqdm(val_loader):
            images = FLOAT(images).to(device)
            labels = FLOAT(labels.float()).to(device)
            if dtype=='fp16':
                images = images.half()
            outputs =  model(images).clone().detach().to(dtype=torch.float16)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

        MSE = total_loss/len(val_loader)

    return MSE

def evaluate_model(model_path, opt_model, val_set, val_loader, dtype='fp32'):
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

    mse = measure_mse(opt_model, val_loader, dtype)
    
    inf_time = measure_inference_time(opt_model,  val_set, dtype)

    return model_size, mse, inf_time

#model_size, mse, inf_time = evaluate_model('trt_mod.jit.pt', trt_mod, dataset, testing_dataloader)
model_size, mse, inf_time = evaluate_model('trt_mod.jit.pt', trt_mod, dataset, testing_dataloader)
#model_size, mse, inf_time = evaluate_model('trt_mod.jit.pt', trt_mod, dataset, testing_dataloader, dtype='fp16')


print("Model size (MB):", model_size)
print("MSE:", mse)
print("Inference time (s):", inf_time)