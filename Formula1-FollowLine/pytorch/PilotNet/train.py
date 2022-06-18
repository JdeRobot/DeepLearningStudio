import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import os
from utils.processing import *
from utils.pilot_net_dataset import PilotNetDataset
from utils.pilotnet import PilotNet
from utils.transform_helper import createTransform

import argparse
from PIL import Image

import json
import numpy as np
from copy import deepcopy

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", action='append', help="Directory to find Data")
    parser.add_argument("--preprocess", action='append', default=None, help="preprocessing information: choose from crop/nocrop and normal/extreme")
    parser.add_argument("--base_dir", type=str, default='exp_random', help="Directory to save everything")
    parser.add_argument("--comment", type=str, default='Random Experiment', help="Comment to know the experiment")
    parser.add_argument("--data_augs", action='append', type=str, default=None, help="Data Augmentations")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of Epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Policy Net")
    parser.add_argument("--test_split", type=float, default=0.2, help="Train test Split")
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
    test_split = args.test_split
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

    # Creating data indices for training and test splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_split = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_split)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

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
    loss_list = []
    acc_list = []
    val_acc = -1
    global_iter = 0

    for epoch in range(last_epoch, num_epochs):
        pilotModel.train()

        for i, (images, labels) in enumerate(train_loader):
            
            images = FLOAT(images).to(device)
            labels = FLOAT(labels.float()).to(device)
            
            # Run the forward pass
            outputs = pilotModel(images)
            loss = criterion(outputs, labels)
            current_loss = loss.item()

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            correct = (torch.linalg.norm(outputs - labels, axis=0) < 0.1).sum().item()
            current_acc = (correct / total)

            if global_iter % save_iter == 0:
                torch.save(pilotModel.state_dict(), model_save_dir + '/pilot_net_model_{}.ckpt'.format(random_seed))

            global_iter += 1

            writer.add_scalar("performance/loss", current_loss, global_iter)
            writer.add_scalar("performance/accuracy", current_acc, global_iter)
            writer.add_scalar("training/epochs", epoch+1, global_iter)

            if print_terminal and (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                            (correct / total) * 100))
        with open(model_save_dir+'/args.json', 'w') as fp:
            json.dump({'last_epoch': epoch}, fp)

        # Validation 
        pilotModel.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = FLOAT(images).to(device)
                labels = FLOAT(labels.float()).to(device)
                outputs = pilotModel(images)
                total += labels.size(0)
                correct += (torch.linalg.norm(outputs - labels) < 0.1).sum().item()

            curr_acc = (correct / total) * 100
            if curr_acc >= val_acc:
                val_acc = curr_acc
                best_model = deepcopy(pilotModel)
                mssg = "Accuracy Improved!!"
            else:
                mssg = "No Improved!!"

            print(f'Validation Accuracy of the model on the images: {curr_acc}\t' + mssg)

    # Test the model
    pilotModel = best_model # allot the best model on validation 
    pilotModel.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = FLOAT(images).to(device)
            labels = FLOAT(labels.float()).to(device)
            outputs = pilotModel(images)
            total += labels.size(0)
            correct += (torch.linalg.norm(outputs - labels) < 0.1).sum().item()

        print('Test Accuracy of the model on the test images: {} %'.format((correct / total) * 100))

    # Save the model and plot
    torch.save(pilotModel.state_dict(), model_save_dir + '/pilot_net_model_{}.ckpt'.format(random_seed))