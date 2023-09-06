# Carla Follow-lane: implementation, and optimization scripts

We provide here resources for implementing a follow-lane model for CARLA simulator based on bird-eye-view input view.
We also provide scripts for optimization of the baseline model for both PyTorch and Tensorflow, with support for TensorRT.

# Dataset

Dataset can be downloaded from: https://huggingface.co/datasets/sergiopaniego/CarlaFollowLanePreviousV

# Models

Download the models from: https://huggingface.co/sergiopaniego/OptimizedPilotNet

# Running the code

## PyTorch <img src="https://pytorch.org/assets/images/pytorch-logo.png" alt="Pytorch logo" width="50"/> 

```
## Training
cd PilotNet
python train.py --data_dir  <DATA_DIRECTORY> \
		--preprocess 'crop' \
		--preprocess 'extreme' \
    --base_dir testcase \
    --comment 'Selected Augmentations: gaussian, affine' \
    --data_augs 'gaussian' \
    --data_augs 'affine' \
    --num_epochs 150 \
    --lr 1e-3 \
    --test_split 0.2 \
    --shuffle True \
    --batch_size 128 \
    --save_iter 50 \
    --print_terminal True \
    --seed 123  

## PyTorch optimizations
python3 optimize_models.py --data_dir <DATA_DIRECTORY> \
	--preprocess 'crop' \
	--preprocess 'extreme' \
	--data_augs 'all' \
	--model_dir experiments/retrain_best/trained_models/pilotnet_model_best_108.pth \
	--num_epochs 2 \
	--batch_size 1024 \
	--lr 1e-4 \
	--eval_base True \
	--tech all

## TensorRT (TF-TRT) optimization
python3 tensorrt_optimize.py --data_dir <DATA_DIRECTORY> \
	--preprocess 'crop' \
	--preprocess 'extreme' \
	--data_augs 'all' \
	--model_dir <BASE_MODEL> \
	--num_epochs 2 \
	--batch_size 1024 \
	--lr 1e-4 \
	--eval_base True \
	--tech all

## TensorRT (TF-TRT) optimization INT8
python3 tensorrt_optimize_int8.py --data_dir <DATA_DIRECTORY> \
	--preprocess 'crop' \
	--preprocess 'extreme' \
	--data_augs 'all' \
	--model_dir <BASE_MODEL> \
	--num_epochs 2 \
	--batch_size 1024 \
	--lr 1e-4 \
	--eval_base True \
	--tech all
```

## Tensorflow <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1200px-Tensorflow_logo.svg.png" alt="TF logo" width="50"/> 
```
## Training
cd PilotNet
python3 train.py --data_dir <DATA_DIRECTORY> \
    --preprocess crop \
    --preprocess extreme \
    --data_augs 2 \
    --num_epochs 1 \
    --learning_rate 0.0001 \
    --batch_size 50 \
    --img_shape "3,100,50,3"

## PyTorch optimizations
python3 optimize_models.py --data_dir <DATA_DIRECTORY> \
	--preprocess crop \
	--preprocess extreme \
	--data_augs 2 \
	--img_shape "200,66,3" \
	--batch_size 64 \
	--model_path <BASE_MODEL> \
	--model_name pilotnet \
	--eval_base True \
 	--tech all 

## TensorRT (TF-TRT) optimization
python3 tensorrt_optimize.py --data_dir <DATA_DIRECTORY> \
	--preprocess crop \
	--preprocess extreme \
	--data_augs 2 \
	--img_shape "200,66,3" \
	--batch_size 64 \
	--model_path <BASE_MODEL> \
	--model_name pilotnet \
	--eval_base True \
 	--precision all

```
