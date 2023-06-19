# Code adapted from: https://github.com/pytorch/TensorRT/blob/main/notebooks/vgg-qat.ipynb

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

########################################################
from pytorch_quantization import quant_modules
quant_modules.initialize()


def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    model.cuda()

def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistics"""
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def calibrate_model(model, model_name, data_loader, num_calib_batch, calibrator, hist_percentile, out_dir):
    """
        Feed data to the network and calibrate.
        Arguments:
            model: classification model
            model_name: name to use when creating state files
            data_loader: calibration data set
            num_calib_batch: amount of calibration passes to perform
            calibrator: type of calibration to use (max/histogram)
            hist_percentile: percentiles to be used for historgram calibration
            out_dir: dir to save state files in
    """

    if num_calib_batch > 0:
        print("Calibrating model")
        with torch.no_grad():
            collect_stats(model, data_loader, num_calib_batch)

        if not calibrator == "histogram":
            compute_amax(model, method="max")
            calib_output = os.path.join(
                out_dir,
                F"{model_name}-max-{num_calib_batch*data_loader.batch_size}.pth")
            torch.save(model.state_dict(), calib_output)
        else:
            for percentile in hist_percentile:
                print(F"{percentile} percentile calibration")
                compute_amax(model, method="percentile")
                calib_output = os.path.join(
                    out_dir,
                    F"{model_name}-percentile-{percentile}-{num_calib_batch*data_loader.batch_size}.pth")
                torch.save(model.state_dict(), calib_output)

            for method in ["mse", "entropy"]:
                print(F"{method} calibration")
                compute_amax(model, method=method)
                calib_output = os.path.join(
                    out_dir,
                    F"{model_name}-{method}-{num_calib_batch*data_loader.batch_size}.pth")
                torch.save(model.state_dict(), calib_output)



#####################3



pilotModel = PilotNet(image_shape, 3).eval().to(device)
pilotModel.load_state_dict(torch.load(model_dir))


inputs = [torch.rand(1, 3, 200, 66)] # Input should be a tensor

print(inputs[0].shape)

augmentations = 'all'
#augmentations = ''

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
#testing_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
testing_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

###################################################
'''
calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
    testing_dataloader,
    cache_file="./calibration.cache",
    use_cache=False,
    algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
    device=torch.device(device),
)
'''

import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import calib

#Calibrate the model using max calibration technique.
with torch.no_grad():
    calibrate_model(
        model=pilotModel,
        model_name="pilotNet",
        data_loader=testing_dataloader,
        #num_calib_batch=32,
        num_calib_batch=1,
        calibrator="max",
        hist_percentile=[99.9, 99.99, 99.999, 99.9999],
        out_dir="./")

import torch.optim as optim
import torch.nn.functional as F

# Declare Learning rate
#lr = 0.1
lr = 0.0001
state = {}
state["lr"] = lr

# Use cross entropy loss for classification and SGD optimizer
#crit = nn.CrossEntropyLoss()
#opt = optim.SGD(pilotModel.parameters(), lr=state["lr"], momentum=0.9, weight_decay=1e-4)

crit = nn.MSELoss()
opt = torch.optim.Adam(pilotModel.parameters(), lr=state["lr"])

# Adjust learning rate based on epoch number
def adjust_lr(optimizer, epoch):
    global state
    new_lr = lr * (0.5**(epoch // 12)) if state["lr"] > 1e-7 else state["lr"]
    if new_lr != state["lr"]:
        state["lr"] = new_lr
        print("Updating learning rate: {}".format(state["lr"]))
        for param_group in optimizer.param_groups:
            param_group["lr"] = state["lr"]


def train(model, dataloader, crit, opt, epoch):
#     global writer
    model.train()
    running_loss = 0.0
    for batch, (data, labels) in enumerate(dataloader):
        '''
        data, labels = data.cuda(), labels.cuda(non_blocking=True)
        opt.zero_grad()
        out = model(data)
        loss = crit(out, labels)
        loss.backward()
        opt.step()
        '''
        images = FLOAT(data).to(device)
        labels = FLOAT(labels.float()).to(device)
        # Run the forward pass
        outputs = model(images)
        loss = crit(outputs, labels)
        current_loss = loss.item()
        #train_loss += current_loss
        # Backprop and perform Adam optimisation
        opt.zero_grad()
        loss.backward()
        opt.step()


        running_loss += loss.item()
        if batch % 500 == 499:
            #print("Batch: [%5d | %5d] loss: %.3f" % (batch + 1, len(dataloader), running_loss / 100))
            print("Batch: [%5d | %5d] loss: %.3f" % (batch + 1, len(dataloader), running_loss / 500))
            running_loss = 0.0
        
def test(model, dataloader, crit, epoch):
    global writer
    global classes
    total = 0
    #correct = 0
    loss = 0.0
    #class_probs = []
    #class_preds = []
    model.eval()
    with torch.no_grad():
        for data, labels in dataloader:
            data = FLOAT(data).to(device)
            labels = FLOAT(labels.float()).to(device)
            #data, labels = data.cuda(), labels.cuda(non_blocking=True)
            out = model(data)
            loss += crit(out, labels)
            #preds = torch.max(out, 1)[1]
            #class_probs.append([F.softmax(i, dim=0) for i in out])
            #class_preds.append(preds)
            total += labels.size(0)
            #correct += (preds == labels).sum().item()

    #test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    #test_preds = torch.cat(class_preds)

    return loss / total #, correct / total

def save_checkpoint(state, ckpt_path="checkpoint.pth"):
    torch.save(state, ckpt_path)
    print("Checkpoint saved")


# Finetune the QAT model for 1 epoch
num_epochs=10
for epoch in range(num_epochs):
    adjust_lr(opt, epoch)
    print('Epoch: [%5d / %5d] LR: %f' % (epoch + 1, num_epochs, state["lr"]))

    train(pilotModel, testing_dataloader, crit, opt, epoch)
    #test_loss, test_acc = test(pilotModel, testing_dataloader, crit, epoch)
    test_loss = test(pilotModel, testing_dataloader, crit, epoch)

    print("Test Loss: {:.5f}".format(test_loss))
    
save_checkpoint({'epoch': epoch + 1,
                 'model_state_dict': pilotModel.state_dict(),
                 #'acc': test_acc,
                 'opt_state_dict': opt.state_dict(),
                 'state': state},
                ckpt_path="pilotNet_qat_ckpt")


quant_nn.TensorQuantizer.use_fb_fake_quant = True
with torch.no_grad():
    data = iter(testing_dataloader)
    images, _ = next(data)
    jit_model = torch.jit.trace(pilotModel, images.to("cuda"))
    torch.jit.save(jit_model, "trained_pilotNet_qat.jit.pt")


qat_model = torch.jit.load("trained_pilotNet_qat.jit.pt").eval()

compile_spec = {"inputs": [torch_tensorrt.Input([1, 3, 200, 66])],
                "enabled_precisions": torch.int8,
                }
trt_mod = torch_tensorrt.compile(qat_model, **compile_spec)

test_loss = test(trt_mod, testing_dataloader, crit, 0)
print("PilotNet QAT Loss using TensorRT: {:.5f}%".format(test_loss))


import time
import numpy as np

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# Helper function to benchmark the model
def benchmark(model, input_shape=(1024, 1, 32, 32), dtype='fp32', nwarmup=50, nruns=1000):
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
            output = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%100==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.size())
    print("Output shape:", output.shape)
    print('Average batch time: %.2f ms'%(np.mean(timings)*1000))


benchmark(jit_model, input_shape=(16, 3, 200, 66))



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

model_size, mse, inf_time = evaluate_model('trained_pilotNet_qat.jit.pt', trt_mod, dataset, testing_dataloader)

print("Model size (MB):", model_size)
print("MSE:", mse)
print("Inference time (s):", inf_time)
