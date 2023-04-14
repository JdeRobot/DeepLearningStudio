import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import os
from utils.processing_carla import *
from utils.pilot_net_dataset import PilotNetDataset, PilotNetDatasetTest
from utils.pilotnet import PilotNet
from utils.transform_helper import createTransform

import argparse
from PIL import Image

import json
import numpy as np
from copy import deepcopy
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", action='append', help="Directory to find Train Data")
    parser.add_argument("--test_dir", action='append', help="Directory to find Test Data")
    parser.add_argument("--preprocess", action='append', default=None, help="preprocessing information: choose from crop/nocrop and normal/extreme")
    parser.add_argument("--base_dir", type=str, default='exp_random', help="Directory to save everything")
    parser.add_argument("--comment", type=str, default='Random Experiment', help="Comment to know the experiment")
    parser.add_argument("--data_augs", action='append', type=str, default=None, help="Data Augmentations")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of Epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Policy Net")
    parser.add_argument("--val_split", type=float, default=0.2, help="Train test Split")
    parser.add_argument("--shuffle", type=bool, default=False, help="Shuffle dataset")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--save_iter", type=int, default=50, help="Iterations to save the model")
    parser.add_argument("--print_terminal", type=bool, default=False, help="Print progress in terminal")
    parser.add_argument("--seed", type=int, default=123, help="Seed for reproducing")

    args = parser.parse_args()
    return args

if __name__=="__main__":

    args = parse_args()

    exp_setup = vars(args)

    # Base Directory
    path_to_data = args.data_dir
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
    learning_rate = args.lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle
    save_iter = args.save_iter
    random_seed = args.seed
    print_terminal = args.print_terminal

    # Device Selection (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FLOAT = torch.FloatTensor

    # Tensorboard Initialization
    writer = SummaryWriter(log_dir)

    # Define data transformations
    transformations = createTransform(augmentations)
    # Load data
    dataset = PilotNetDataset(path_to_data, transformations, preprocessing=args.preprocess)

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_split = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(val_split)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    # Load Model
    pilotModel = PilotNet(dataset.image_shape, dataset.num_labels).to(device)
    if os.path.isfile( model_save_dir + '/pilot_net_model_{}.ckpt'.format(random_seed)):
        pilotModel.load_state_dict(torch.load(model_save_dir + '/pilot_net_model_{}.ckpt'.format(random_seed),map_location=device))
        last_epoch = json.load(open(model_save_dir+'/args.json',))['last_epoch']+1
    else:
        last_epoch = 0

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(pilotModel.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    global_iter = 0
    global_val_mse = 1e+5


    print("*********** Training Started ************")
    for epoch in range(last_epoch, num_epochs):
        pilotModel.train()
        train_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            
            images = FLOAT(images).to(device)
            labels = FLOAT(labels.float()).to(device)
            # Run the forward pass
            outputs = pilotModel(images)
            loss = criterion(outputs, labels)
            current_loss = loss.item()
            train_loss += current_loss
            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_iter % save_iter == 0:
                torch.save(pilotModel.state_dict(), model_save_dir + '/pilot_net_model_{}.ckpt'.format(random_seed))
            global_iter += 1

            if print_terminal and (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        
        # add entry of last epoch                            
        with open(model_save_dir+'/args.json', 'w') as fp:
            json.dump({'last_epoch': epoch}, fp)

        writer.add_scalar("performance/train_loss", train_loss/len(train_loader), epoch+1)

        # Validation 
        pilotModel.eval()
        with torch.no_grad():
            val_loss = 0 
            for images, labels in val_loader:
                images = FLOAT(images).to(device)
                labels = FLOAT(labels.float()).to(device)
                outputs = pilotModel(images)
                val_loss += criterion(outputs, labels).item()
                
            val_loss /= len(val_loader) # take average
            writer.add_scalar("performance/valid_loss", val_loss, epoch+1)

        # compare
        if val_loss < global_val_mse:
            global_val_mse = val_loss
            best_model = deepcopy(pilotModel)
            torch.save(best_model.state_dict(), model_save_dir + '/pilot_net_model_best_{}.pth'.format(random_seed))
            mssg = "Model Improved!!"
        else:
            mssg = "Not Improved!!"

        print('Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch + 1, num_epochs, val_loss), mssg)
                

    pilotModel = best_model # allot the best model on validation 
    # Test the model
    transformations_val = createTransform([]) # only need Normalize()
    if args.test_dir != None:
        test_set = PilotNetDatasetTest(args.test_dir, transformations_val, preprocessing=args.preprocess)
    else:
        test_set = PilotNetDatasetTest(path_to_data, transformations_val, preprocessing=args.preprocess)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    print("Check performance on testset")
    pilotModel.eval()
    with torch.no_grad():
        test_loss = 0
        for images, labels in tqdm(test_loader):
            images = FLOAT(images).to(device)
            labels = FLOAT(labels.float()).to(device)
            outputs = pilotModel(images)
            test_loss += criterion(outputs, labels).item()
    
    writer.add_scalar('performance/Test_MSE', test_loss/len(test_loader))
    print(f'Test loss: {test_loss/len(test_loader)}')
        
    # Save the model and plot
    torch.save(pilotModel.state_dict(), model_save_dir + '/pilot_net_model_{}.ckpt'.format(random_seed))