import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

from utils.pilotnet import PilotNet

import argparse
from PIL import Image
import cv2

import json
import numpy as np

if __name__=="__main__":

    # Device Selection (CPU/GPU)
    device = torch.device("cpu")
    FLOAT = torch.FloatTensor

    transformations = transforms.Compose([
                                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
                                    transforms.ToTensor()
                                ])

    img = cv2.imread('./test_data/1.png')
    img = img[240:480, 0:640]
    img = cv2.resize(img, (int(200), int(66)))
    
    pilotModel = PilotNet(img.shape, 2).to(device)
    pilotModel.load_state_dict(torch.load('./experiments/27May1/trained_models/pilot_net_model_42.ckpt',map_location=device))

    # Test the model
    pilotModel.eval()
    with torch.no_grad():
        data = Image.fromarray(img)
        image = transformations(data).unsqueeze(0)
        image = FLOAT(image).to(device)
        outputs = pilotModel(image).numpy()

        print(outputs[0][1])


        