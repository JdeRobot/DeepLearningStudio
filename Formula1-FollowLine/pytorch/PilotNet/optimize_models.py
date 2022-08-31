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
import time
import torch.nn.utils.prune as prune


# Device Selection (CPU/GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") # support available only for cpu
FLOAT = torch.FloatTensor

def measure_inference_time(model, val_set):
    # measure average inference time
    
    # GPU warm-up
    r_idx = np.random.randint(0, len(val_set), 50)
    for i in r_idx:
        image, _ = val_set[i]
        image = torch.unsqueeze(image, 0).to(device)
        _ = model(image) 
    
    # actual inference cal
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
        # print('Average MSE of the model on the test images: {} %'.format( MSE))

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


def evaluate_baseline(base_model, model_save_dir, val_set, val_loader):
    print()
    print('*'*8, 'Baseline evaluation', '*'*8)

    model_path = model_save_dir + '/baseline.pth'# providing dummy input for tracing; change according to image resolution
    traced_model = torch.jit.trace(base_model, torch.rand(1, 3, 200, 66)) 
    torch.jit.save(traced_model, model_path)

    # base_model = torch.jit.load(model_path)

    print("********** Baseline stats **********")
    model_size, mse, inf_time = evaluate_model(model_path, base_model,  val_set, val_loader)
    print("Model size (MB):", model_size)
    print("MSE:", mse)
    print("Inference time (s):", inf_time)
    return model_size, mse, inf_time


def dynamic_quantization(model, model_save_dir, val_set, val_loader):
    print()
    print("********* Start Dynamic range Quantization ***********")
    # Post-training dynamic range quantization

    quant_model = quantize_dynamic(
                    model=model, qconfig_spec={nn.Linear}, dtype=torch.qint8, 
                    inplace=False #can add nn.LSTM in qconfig_spec for LSTM layers
                    )
    
    qmodel_path = model_save_dir + '/dynamic_quan.pth'
    # providing dummy input for tracing; change according to image resolution
    traced_model = torch.jit.trace(quant_model, torch.rand(1, 3, 200, 66)) 
    torch.jit.save(traced_model, qmodel_path)
    # quant_model = torch.jit.load(qmodel_path)

    print("********** Dynamic range Q stats **********")
    model_size, mse, inf_time = evaluate_model(qmodel_path, quant_model,  val_set, val_loader)
    print("Model size (MB):", model_size)
    print("MSE:", mse)
    print("Inference time (s):", inf_time)
    return model_size, mse, inf_time


def static_quantization(model, model_save_dir, val_set, val_loader, train_loader):
    print()
    print("********* Start Static Quantization ***********")

    m = deepcopy(model)
    m.eval()
    
    # fuse modules/layers for reducing memory access, better accuracy and inference
    module_list = [[f'cn_{i}', f'relu{i}'] for i in range(1,6)]
    module_list +=  [[f'fc_{i}', f'relu_fc{i}'] for i in range(1,5)]
    torch.quantization.fuse_modules(m, module_list, inplace=True)
    
    backend = "fbgemm"
    
    """Insert stubs"""
    m = nn.Sequential(torch.quantization.QuantStub(), 
                    *nn.Sequential(*(m.children())), # bring every layer on same level of abstraction
                    torch.quantization.DeQuantStub())

    """Prepare"""
    m.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(m, inplace=True)

    """Calibrate
    - This example uses random data for convenience. Use representative (validation) data instead.
    """
    dataiter = iter(train_loader)
    with torch.inference_mode():
        for _ in range(10):
            imgs, _ = next(dataiter) 
            m(imgs)
        
    """Convert"""
    quant_model = torch.quantization.convert(m)
    
    """Save model"""
    qmodel_path = model_save_dir + '/static_quan.pth'
    # providing dummy input for tracing; change according to image resolution
    traced_model = torch.jit.trace(quant_model, torch.rand(1, 3, 200, 66)) 
    torch.jit.save(traced_model, qmodel_path)

    print("********** Static Q stats **********")
    model_size, mse, inf_time = evaluate_model(qmodel_path, quant_model,  val_set, val_loader)
    print("Model size (MB):", model_size)
    print("MSE:", mse)
    print("Inference time (s):", inf_time)

    return model_size, mse, inf_time


def quantization_aware_train(model, model_save_dir, val_set, val_loader, train_loader, args):
    print()
    print("********* Start Quantization Aware Training ***********")

    m = deepcopy(model)
    m.eval()
    
    backend = "fbgemm"  # running on a x86 CPU. Use "qnnpack" if running on ARM.

    # fuse modules/layers for reducing memory access, better accuracy and inference
    module_list = [[f'cn_{i}', f'relu{i}'] for i in range(1,6)]
    module_list +=  [[f'fc_{i}', f'relu_fc{i}'] for i in range(1,5)]
    torch.quantization.fuse_modules(m, module_list, inplace=True)
        
    """Insert stubs"""
    m = nn.Sequential(torch.quantization.QuantStub(), 
                    *nn.Sequential(*(m.children())), # bring every layer on same level of abstraction
                    torch.quantization.DeQuantStub())

    """Prepare"""
    m.train()
    m.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare_qat(m, inplace=True)

    """Training Loop"""
    n_epochs = args.num_epochs
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=args.lr)

    for epoch in range(n_epochs):
        for images, labels in tqdm(train_loader):
            images = torch.FloatTensor(images)#.to(device)
            labels = torch.FloatTensor(labels.float())#.to(device)
            out = m(images)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    """Convert"""
    quant_model = torch.quantization.convert(m)
    
    """Save model"""
    qmodel_path = model_save_dir + '/quan_aware.pth'
    # providing dummy input for tracing; change according to image resolution
    traced_model = torch.jit.trace(quant_model, torch.rand(1, 3, 200, 66)) 
    torch.jit.save(traced_model, qmodel_path)

    print("********** QAT stats **********")
    model_size, mse, inf_time = evaluate_model(qmodel_path, quant_model,  val_set, val_loader)
    print("Model size (MB):", model_size)
    print("MSE:", mse)
    print("Inference time (s):", inf_time)

    return model_size, mse, inf_time


def local_prune(model, model_save_dir, val_set, val_loader, train_loader, args):
    print()
    print("********* Start Local Pruning (Unstructured) ***********")

    ## Prune multiple parameters of the model
    for name, module in model.named_modules():
        # prune 20% of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=0.3)
        # prune 40% of connections in all linear layers
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.5)
    
    """Fine-tune Loop"""
    print("Fine-tuning pruned model .....")
    n_epochs = args.num_epochs
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(n_epochs):
        for images, labels in tqdm(train_loader):
            images = torch.FloatTensor(images).to(device)
            labels = torch.FloatTensor(labels.float()).to(device)
            out = model(images)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    ## remove re-parameterization to make pruning permanent
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, name='weight')
        elif isinstance(module, torch.nn.Linear):
            prune.remove(module, name='weight')
    
    """Save model"""
    model_path = model_save_dir + '/local_prune.pth'
    # providing dummy input for tracing; change according to image resolution
    traced_model = torch.jit.trace(model, torch.rand(1, 3, 200, 66)) 
    torch.jit.save(traced_model, model_path)

    print("********** Local Prune stats **********")
    model_size, mse, inf_time = evaluate_model(model_path, model, val_set, val_loader)
    print("Model size (MB):", model_size)
    print("MSE:", mse)
    print("Inference time (s):", inf_time)

    return model_size, mse, inf_time


def global_prune(model, model_save_dir, val_set, val_loader, train_loader, args):
    print()
    print("********* Start Global Pruning (Unstructured) ***********")

    # collect modules to prune (here conv2d and linear layers)
    parameters_to_prune = []
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, "weight"))

    ## Global prune
    # zero-out low tensors with lowest value according to pruning_method
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.3,
    )
        
    """Fine-tune Loop"""
    print("Fine-tuning pruned model .....")
    n_epochs = args.num_epochs
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(n_epochs):
        for images, labels in tqdm(train_loader):
            images = torch.FloatTensor(images).to(device)
            labels = torch.FloatTensor(labels.float()).to(device)
            out = model(images)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    

    ## remove re-parameterization to make pruning permanent
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, name='weight')
        elif isinstance(module, torch.nn.Linear):
            prune.remove(module, name='weight')
    
    """Save model"""
    model_path = model_save_dir + '/global_prune.pth'
    # providing dummy input for tracing; change according to image resolution
    traced_model = torch.jit.trace(model, torch.rand(1, 3, 200, 66)) 
    torch.jit.save(traced_model, model_path)

    print("********** global Prune stats **********")
    model_size, mse, inf_time = evaluate_model(model_path, model, val_set, val_loader)
    print("Model size (MB):", model_size)
    print("MSE:", mse)
    print("Inference time (s):", inf_time)

    return model_size, mse, inf_time



def prune_quan(model, model_save_dir, val_set, val_loader, train_loader, args):
    print()
    print("********* Start Pruning + Quantization ***********")

    """ Pruning steps """
    ## Prune multiple parameters of the model
    for name, module in model.named_modules():
        # prune 20% of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=0.3)
        # prune 40% of connections in all linear layers
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.5)
    
    """Fine-tune Loop"""
    print("Fine-tuning pruned model .....")
    n_epochs = args.num_epochs
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(n_epochs):
        for images, labels in tqdm(train_loader):
            images = torch.FloatTensor(images).to(device)
            labels = torch.FloatTensor(labels.float()).to(device)
            out = model(images)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    ## remove re-parameterization to make pruning permanent
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, name='weight')
        elif isinstance(module, torch.nn.Linear):
            prune.remove(module, name='weight')
    
    """ Quantization steps """
    m = deepcopy(model)
    m.eval()
    
    backend = "fbgemm"  # running on a x86 CPU. Use "qnnpack" if running on ARM.

    # fuse modules/layers for reducing memory access, better accuracy and inference
    module_list = [[f'cn_{i}', f'relu{i}'] for i in range(1,6)]
    module_list +=  [[f'fc_{i}', f'relu_fc{i}'] for i in range(1,5)]
    torch.quantization.fuse_modules(m, module_list, inplace=True)
        
    """Insert stubs"""
    m = nn.Sequential(torch.quantization.QuantStub(), 
                    *nn.Sequential(*(m.children())), # bring every layer on same level of abstraction
                    torch.quantization.DeQuantStub())

    """Prepare"""
    m.train()
    m.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare_qat(m, inplace=True)

    """Training Loop"""
    n_epochs = args.num_epochs
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=args.lr)

    for epoch in range(n_epochs):
        for images, labels in tqdm(train_loader):
            images = torch.FloatTensor(images)#.to(device)
            labels = torch.FloatTensor(labels.float())#.to(device)
            out = m(images)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    """Convert"""
    quant_model = torch.quantization.convert(m)
    
    """Save model"""
    qmodel_path = model_save_dir + '/prune_quan.pth'
    # providing dummy input for tracing; change according to image resolution
    traced_model = torch.jit.trace(quant_model, torch.rand(1, 3, 200, 66)) 
    torch.jit.save(traced_model, qmodel_path)

    print("********** Prune + Quanzation stats **********")
    model_size, mse, inf_time = evaluate_model(qmodel_path, quant_model,  val_set, val_loader)
    print("Model size (MB):", model_size)
    print("MSE:", mse)
    print("Inference time (s):", inf_time)

    return model_size, mse, inf_time



def set_device(args):
    if args.device_type == 'cuda':
        global device 
        device = torch.device('cuda')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", action='append', help="Directory to find Data")
    parser.add_argument("--val_dir", action='append', help="Directory to find Data")
    parser.add_argument("--preprocess", action='append', default=None, help="preprocessing information: choose from crop/nocrop and normal/extreme")
    parser.add_argument("--data_augs", action='append', type=str, default=None, help="Data Augmentations")
    parser.add_argument("--base_dir", type=str, default='optimized_models', help="Directory to save everything")
    parser.add_argument("--model_dir", type=str, default='trained_models/', help="Directory to trained models")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of Epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Policy Net")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--device_type", type=str, default='cpu', help="Decide which hardware to use for computation - 'cuda' or 'cpu'.")
    parser.add_argument('--eval_base', type=bool, default=False, help="If set to True, it will calculate accuracy, size and inference time for original model.")
    parser.add_argument("--tech", action='append', default=[], help="Techniques to apply for model compression. Options are: \n"+
                               "'dynamic_quan', 'static_quan', 'quan_aware', 'local_prune', 'global_prune', 'prune_quan' and 'all' .") 
    

    args = parser.parse_args()
    return args


if __name__=="__main__":

    args = parse_args()

    exp_setup = vars(args)

    set_device(args)    

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

    # Tensorboard Initialization
    writer = SummaryWriter(log_dir)

    # Define data transformations, Load data, DataLoader for batching
    transformations_train = createTransform(augmentations)
    train_set = PilotNetDataset(args.data_dir, transformations_train, preprocessing=args.preprocess)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    transformations_val = createTransform([]) # only need Normalize()
    val_set = PilotNetDataset(args.val_dir, transformations_val, preprocessing=args.preprocess)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Load Model
    pilotModel = PilotNet(val_set.image_shape, val_set.num_labels).to(device)
    pilotModel.load_state_dict(torch.load(args.model_dir))
    
    results = []

    if args.eval_base:
        res = evaluate_baseline(pilotModel, model_save_dir, val_set, val_loader)
        results.append(("Baseline",) + res)
    if "dynamic_quan" in args.tech or 'all' in args.tech : 
        res = dynamic_quantization(pilotModel, model_save_dir, val_set, val_loader)
        results.append(("Dynamic Range Q",) + res)
    if "static_quan" in args.tech or 'all' in args.tech : 
        res = static_quantization(pilotModel, model_save_dir, val_set, val_loader, train_loader)
        results.append(("Static Q",) + res)
    if "quan_aware" in args.tech or 'all' in args.tech : 
        res = quantization_aware_train(pilotModel, model_save_dir, val_set, val_loader, train_loader, args)
        results.append(("QAT",) + res)
    if "local_prune" in args.tech or 'all' in args.tech : 
        res = local_prune(pilotModel, model_save_dir, val_set, val_loader, train_loader, args)
        results.append(("Local Prune",) + res)
    if "global_prune" in args.tech or 'all' in args.tech : 
        res = global_prune(pilotModel, model_save_dir, val_set, val_loader, train_loader, args)
        results.append(("Global Prune",) + res)
    if "prune_quan" in args.tech or 'all' in args.tech : 
        res = prune_quan(pilotModel, model_save_dir, val_set, val_loader, train_loader, args)
        results.append(("Prune + Quantization",) + res)
    
    df = pd.DataFrame(results)
    df.columns = ["Method", "Model size (MB)", "MSE", "Inference time (s)"]
    df.to_csv("model_evaluation.csv", index=False)