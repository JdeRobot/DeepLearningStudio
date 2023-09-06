# Carla Follow-lane: implementation, and optimization scripts

We provide here resources for implementing a follow-lane model for CARLA simulator based on bird-eye-view input view.
We also provide scripts for optimization of the baseline model for both PyTorch and Tensorflow, with support for TensorRT.

# Dataset

Dataset can be downloaded from: https://huggingface.co/datasets/YujiroS/traffic-6

# Models

Download the models from: https://huggingface.co/YujiroS/subjective_vision_pilotnet

# Running the code

## Tensorflow <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1200px-Tensorflow_logo.svg.png" alt="TF logo" width="50"/> 
```
## Training
cd PilotNet
python3 train.py --data_dir <DATA_DIRECTORY> \
    --preprocess crop \
    --data_augs 2 \
    --num_epochs 91 \
    --learning_rate 0.0001 \
    --batch_size 80 \
    --img_shape "66,200,4" \
