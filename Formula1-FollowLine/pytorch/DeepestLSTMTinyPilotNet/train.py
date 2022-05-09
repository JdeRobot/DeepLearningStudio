import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import os
from utils.processing import *
from utils.dataset import DLTNetDataset
from utils.deepest_lstm_tinypilotnet import DeepestLSTMTinyPilotNet
from utils.transform_helper import createTransform

import argparse
from PIL import Image

import json
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", action='append', help="Directory to find Data")
    parser.add_argument("--preprocess", action='append', default=None, help="preprocessing information: choose from crop/nocrop and normal/extreme")
    parser.add_argument("--base_dir", type=str, default='exp_random', help="Directory to save everything")
    parser.add_argument("--comment", type=str, default='Random Experiment', help="Comment to know the experiment")
    parser.add_argument("--data_augs", action='append', type=str, default=None, help="Data Augmentations")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of Epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for Policy Net")
    parser.add_argument("--test_split", type=float, default=0.2, help="Train test Split")
    parser.add_argument("--shuffle", type=bool, default=False, help="Shuffle dataset")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--save_iter", type=int, default=50, help="Iterations to save the model")
    parser.add_argument("--print_terminal", type=bool, default=False, help="Print progress in terminal")
    parser.add_argument("--seed", type=int, default=123, help="Seed for reproducing")
    parser.add_argument("--img_shape", type=str, default=(200, 66, 3), help="Image shape") # Additional

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
    img_shape = tuple(map(int, args.img_shape.split(',')))

    # Device Selection (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FLOAT = torch.FloatTensor

    # Tensorboard Initialization
    writer = SummaryWriter(log_dir)

    # Define data transformations
    transformations = createTransform(augmentations)
    # Load data
    train_set = DLTNetDataset(path_to_data, img_shape, transformations, preprocessing=args.preprocess, trainset=True)
    valid_set = DLTNetDataset(path_to_data, img_shape, transformations, preprocessing=args.preprocess, trainset=False)

    # Dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)

    # Load Model
    DLTNetModel = DeepestLSTMTinyPilotNet(train_set.image_shape, train_set.num_labels).to(device)
    if os.path.isfile( model_save_dir + '/DLT_net_model_{}.ckpt'.format(random_seed)):
        DLTNetModel.load_state_dict(torch.load(model_save_dir + '/DLT_net_model_{}.ckpt'.format(random_seed),map_location=device))
        last_epoch = json.load(open(model_save_dir+'/args.json',))['last_epoch']+1
    else:
        last_epoch = 0

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(DLTNetModel.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    global_iter = 0
    for epoch in range(last_epoch, num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            
            images = FLOAT(images).to(device)
            labels = FLOAT(labels.float()).to(device)
            
            # Run the forward pass
            outputs = DLTNetModel(images)
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
                torch.save(DLTNetModel.state_dict(), model_save_dir + '/DLT_net_model_{}.ckpt'.format(random_seed))

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

    # Test the model
    DLTNetModel.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = FLOAT(images).to(device)
            labels = FLOAT(labels.float()).to(device)
            outputs = DLTNetModel(images)
            total += labels.size(0)
            correct += (torch.linalg.norm(outputs - labels) < 0.1).sum().item()

        print('Validation Accuracy of the model on the validation images: {} %'.format((correct / total) * 100))

    # Save the model and plot
    torch.save(DLTNetModel.state_dict(), model_save_dir + '/DLT_net_model_{}.ckpt'.format(random_seed))