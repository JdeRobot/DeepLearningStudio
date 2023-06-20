import os
import time
import torch
import torch_tensorrt
import torchvision
import argparse

import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms

from utils.processing_carla import *
from utils.pilot_net_dataset import PilotNetDataset, PilotNetDatasetTest
from utils.transform_helper import createTransform
from utils.pilotnet import PilotNet

FLOAT = torch.FloatTensor

def benchmark(model, input_shape=(1024, 1, 224, 224), dtype='float32', nwarmup=50, nruns=10000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype=='half':
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



def measure_inference_time(model, val_set, dtype):
    # measure average inference time
    
    # GPU warm-up
    r_idx = np.random.randint(0, len(val_set), 50)
    for i in r_idx:
        image, _ = val_set[i]
        image = torch.unsqueeze(image, 0).to(device)
        if dtype=='half':
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
        if dtype=='half':
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
            if dtype=='half':
                images = images.half()
            outputs =  model(images).clone().detach().to(dtype=torch.float16)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

        MSE = total_loss/len(val_loader)

    return MSE

def evaluate_model(model_path, opt_model, val_set, val_loader, dtype='float32'):
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



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", action='append', help="Directory to find Train Data")
    parser.add_argument("--preprocess", action='append', default=None, help="preprocessing information: choose from crop/nocrop and normal/extreme")
    parser.add_argument("--data_augs", action='append', type=str, default=None, help="Data Augmentations")
    parser.add_argument("--shuffle", type=bool, default=False, help="Shuffle dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--seed", type=int, default=123, help="Seed for reproducing")
    parser.add_argument("--input_shape", type=str, default=(200, 66, 3), help="Image shape")
    parser.add_argument("--model_dir", type=str, help="Directory to find model")
    parser.add_argument("--device", type=str, default="cuda", help="Device for training")
    parser.add_argument("--val_split", type=float, default=0.2, help="Train test Split")
    parser.add_argument("--model_name", type=str, default="PilotNet", help="Model name")
    parser.add_argument("--calibration_type", type=str, default="float32", help="Calitration type float32/half")

    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = parse_args()

    image_shape = np.array(tuple(map(int, args.input_shape.split(','))))
    device = args.device
    model_dir = args.model_dir

    pilotModel = PilotNet(image_shape, 3).eval().to(device)
    pilotModel.load_state_dict(torch.load(model_dir))

    augmentations = args.data_augs
    path_to_data = args.data_dir
    val_split = args.val_split
    shuffle_dataset = args.shuffle
    random_seed = args.seed
    batch_size = args.batch_size


    # Define data transformations
    transformations = createTransform(augmentations)
    # Load data
    dataset = PilotNetDatasetTest(path_to_data, transformations, preprocessing=args.preprocess)

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


    calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
        testing_dataloader,
        cache_file="./calibration.cache",
        use_cache=False,
        algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
        device=torch.device(device),
    )

    if args.calibration_type == 'float32':
        trt_mod = torch_tensorrt.compile(pilotModel, 
                                            inputs=[torch_tensorrt.Input((batch_size, image_shape[2], image_shape[0], image_shape[1]), dtype=torch.float32)],
                                            enabled_precisions={torch.float32},
                                            calibrator=calibrator,
        )
    else:
        trt_mod = torch_tensorrt.compile(pilotModel, 
                                            inputs=[torch_tensorrt.Input((batch_size, image_shape[2], image_shape[0], image_shape[1]), dtype=torch.half)],
                                            enabled_precisions={torch.half},
                                            calibrator=calibrator,
        )

    data = iter(testing_dataloader)
    images, _ = next(data)
    traced_model = torch.jit.trace(pilotModel, images.to("cuda"))
    torch.jit.save(traced_model, args.model_name + '_trt_mod.jit.pt')


    benchmark(trt_mod, input_shape=(batch_size, image_shape[2], image_shape[0], image_shape[1]), dtype=args.calibration_type, nruns=100)

    model_size, mse, inf_time = evaluate_model(args.model_name + '_trt_mod.jit.pt', trt_mod, dataset, testing_dataloader, dtype=args.calibration_type)


    print("Model size (MB):", model_size)
    print("MSE:", mse)
    print("Inference time (s):", inf_time)
