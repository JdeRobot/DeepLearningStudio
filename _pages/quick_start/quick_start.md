---
title: Quick Start
layout: posts
permalink: /quick_start/

collection: posts

classes: wide

sidebar:
  nav: "docs"

gallery1:
  - url: /assets/images/behavior_suite_diagram.png
    image_path: /assets/images/behavior_suite_diagram.png
    alt: ""
gallery2:
  - url: /assets/images/matrix_schema.png
    image_path: /assets/images/matrix_schema.png
    alt: ""
gallery3:
  - url: /assets/images/config1.png
    image_path: /assets/images/config1.png
    alt: ""
gallery4:
  - url: /assets/images/default_config.png
    image_path: /assets/images/default_config.png
    alt: ""
gallery5:
  - url: /assets/images/main_window.png
    image_path: /assets/images/main_window.png
    alt: ""
gallery6:
  - url: /assets/images/toolbar.png
    image_path: /assets/images/toolbar.png
    alt: ""
gallery6.1:
  - url: /assets/images/stats.png
    image_path: /assets/images/stats.png
    alt: ""
gallery7:
  - url: /assets/images/dataset.png
    image_path: /assets/images/dataset.png
    alt: ""
gallery8:
  - url: /assets/images/brain.png
    image_path: /assets/images/brain.png
    alt: ""
gallery9:
  - url: /assets/images/change_brain.gif
    image_path: /assets/images/change_brain.gif
    alt: ""
gallery10:
  - url: /assets/images/simulation.png
    image_path: /assets/images/simulation.png
    alt: ""
gallery11:
  - url: /assets/images/gzclient.gif
    image_path: /assets/images/gzclient.gif
    alt: ""
gallery12:
  - url: /assets/images/brain_sim.gif
    image_path: /assets/images/brain_sim.gif
    alt: ""
gallery13:
  - url: /assets/images/reload_sim.gif
    image_path: /assets/images/reload_sim.gif
    alt: ""
gallery14:
  - url: /assets/images/layout.png
    image_path: /assets/images/layout.png
    alt: ""
gallery15:
  - url: /assets/images/frame.png
    image_path: /assets/images/frame.png
    alt: ""
gallery16:
  - url: /assets/images/rename.gif
    image_path: /assets/images/rename.gif
    alt: ""
gallery17:
  - url: /assets/images/frame_config.gif
    image_path: /assets/images/frame_config.gif
    alt: ""
---

First of all you need to install Deep Learning Studio. If you haven't completed that step, please go to the [installation section](/install/).

We additionally have some implemented algorithms that you can use in Deep Learning Studio. Find them in the [algorithms zoo](/quick_start/algorithms_zoo/).

If you'd like to train your own brain, we provide you with the [datasets](/quick_start/datasets).

This repository contains the deep learning regression and classification models for all robots used in the JdeRobot community.


## Structure of the branch

    ├── Formula1-FollowLine
    |   |
    |   |── pytorch
    |   |   |── PilotNet                                # Pilot Net pytorch implementation
    |   |   |   ├── scripts                             # scripts for running experiments 
    |   |   |   ├── utils                               
    |   |   |   |   ├── pilot_net_dataset.py            # Torchvision custom dataset
    |   |   |   |   ├── pilotnet.py                     # CNN for PilotNet
    |   |   |   |   ├── transform_helpers.py            # Data Augmentation
    |   |   |   |   └── processing.py                   # Data collecting, processing and utilities
    |   |   |   └── train.py                            # training code
    |   |   |
    |   |   └── PilotNetStacked                         # Pilot Net Stacked Image implementation
    |   |       ├── scripts                             # scripts for running experiments 
    |   |       ├── utils                               
    |   |       |   ├── pilot_net_dataset.py            # Sequentially stacked image dataset
    |   |       |   ├── pilotnet.py                     # Modified Hyperparams 
    |   |       |   ├── transform_helpers.py            # Data Augmentation
    |   |       |   └── processing.py                   # Data collecting, processing and utilities
    |   |       └── train.py                            # training code
    |   |
    |   +── tensoflow
    |       +── PilotNet                                # Pilot Net tensorflow implementation
    |           ├── utils                               
    |           |   ├── dataset.py                      # Custom dataset
    |           |   ├── pilotnet.py                     # CNN for PilotNet
    |           |   └── processing.py                   # Data collecting, processing and utilities
    |           └── train.py                            # training code
    +── Drone-FollowLine
        |
        +── DeepPilot                               # DeepPilot CNN pytorch implementation
            ├── scripts                             # scripts for running experiments 
            ├── utils                               
            |   ├── pilot_net_dataset.py            # Torchvision custom dataset
            |   ├── pilotnet.py                     # CNN for DeepPilot
            |   ├── transform_helpers.py            # Data Augmentation
            |   └── processing.py                   # Data collecting, processing and utilities
            └── train.py                            # training code


# Formula1 Followline Algorithms: Implementation and Baseline

It contains some deep learning regression models for Formula1 Line Following task.

The models implemented are derived from:
1. PilotNet for Autonomous Driving with Behaviour Metrics dataset
2. PilotNetStacked as an extension of PilotNet with stacked images

The algorithms are modular and can adapt to various other datasets. They are both implemented in pytorch and tensorflow.

# Pytorch <img src="https://pytorch.org/assets/images/pytorch-logo.png" alt="Pytorch logo" width="50"/> 

## Preparing Dataset 

For PilotNet, we use our custom datasets:
- Complete dataset: contains images with annotations from different circuits [https://drive.google.com/file/d/1Xdiu69DLj7lKK37F94qrUWsXkVg4ymGv/view?usp=sharing](https://drive.google.com/file/d/1Xdiu69DLj7lKK37F94qrUWsXkVg4ymGv/view?usp=sharing)
- Curves dataset: contains images with annotations from many_curves circuit: [https://drive.google.com/file/d/1zCJPFJRqCa34Q6jvktjDBY8Z49bIbvLJ/view?usp=sharing](https://drive.google.com/file/d/1zCJPFJRqCa34Q6jvktjDBY8Z49bIbvLJ/view?usp=sharing)

```
    PilotNet                                # Extract PilotNet dataset here
    ├── complete_dataset                    # Extract PilotNet complete_dataset here           
    |   ├── Images/                         # Train and Test Images
    |   └── data.json                       # Annotations
    └── curves_dataset                      # Extract PilotNet curves_dataset here  
        ├── Images/                         # Train and Test Images
        └── data.json                       # Annotations
```

## Hyperparameters for the code

```
# For PilotNet

-h, --help                            show this help message and exit
--data_dir            DATA_DIR        Directory to find Data
--preprocess	      PREPROCESSING   Preprocessing information about cropping and extreme cases 
--base_dir            BASE_DIR        Directory to save everything
--comment             COMMENT         Comment to know the experiment
--data_augs           AUGMENTATIONS   Data augmentations
--num_epochs          NUM_EPOCHS      Number of Epochs
--lr                  LR              Learning rate for Policy Net
--test_split          TEST_SPLIT      Train test Split
--shuffle             SHUFFLE         Shuffle dataset
--batch_size          BATCH_SIZE      Batch size
--save_iter           SAVE_ITER       Iterations to save the model
--print_terminal      PRINT_TERMINAL  Print progress in terminal
--seed                SEED            Seed for reproducing

# For PilotNetStacked, add

--horizon             HORIZON         Stacking horizon to use

```

## Running the Code

```bash
source ~/pyenvs/dlstudio/bin/activate
cd DeepLearningStudio/Formula1-FollowLine/pytorch

# For PilotNet

cd PilotNet
python train.py --data_dir '../datasets/complete_dataset' \
	    --data_dir '../datasets/curves_only' \
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

# For PilotNetStacked

cd PilotNetStacked
python train.py --data_dir '../datasets/complete_dataset' \
		--data_dir '../datasets/curves_only' \
		--preprocess 'crop' \
		--preprocess 'extreme' \
		--base_dir testcase \
		--comment 'Selected Augmentations: gaussian' \
		--data_augs 'gaussian' \
		--num_epochs 150 \
		--horizon 3 \
		--lr 1e-3 \
		--test_split 0.2 \
		--shuffle True \
		--batch_size 256 \
		--save_iter 50 \
		--print_terminal True \
		--seed 123
```

The results are saved in the `./experiments/` directory and the structure is given below. 
Tensorboard can be launched with `./experiments/base_dir/log` directory.


# Tensorflow <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1200px-Tensorflow_logo.svg.png" alt="TF logo" width="50"/> 

The models implemented are derived from:
1. PilotNet for Autonomous Driving with Behaviour Metrics dataset
2. DeepestLSTMTinyPilotNet as an extension of PilotNet with ConvLSTM layers.

## Preparing Dataset

The same workflow as for PyTorch is followed, refer to the previous section

## Hyperparameters for the code

```
# For PilotNet or DeepestLSTMTinyPilotNet

-h, --help                            show this help message and exit
--data_dir            DATA_DIR        Directory to find Data
--preprocess          PREPROCESSING   Preprocessing information about cropping and extreme cases 
--data_augs           AUGMENTATIONS   Data augmentations
--num_epochs          NUM_EPOCHS      Number of Epochs
--learning_rate       LR              Learning rate for Policy Net
--batch_size          BATCH_SIZE      Batch size
--img_shape	      IMG_SHAPE	      Image shape


```

## Running the Code

```bash
source ~/pyenvs/dlstudio/bin/activate
cd DeepLearningStudio/Formula1-FollowLine/tensorflow

# For PilotNet

cd PilotNet
python train.py --data_dir ../complete_dataset/ \
	--preprocess crop \
	--preprocess extreme \
	--data_augs True \
	--num_epochs 1 \
	--batch_size 50 \
	--learning_rate 0.0001 \
	--img_shape "200,66,3"
	
	
# For DeepestLSTMTinyPilotNet
python train.py --data_dir ../complete_dataset/ \
	--preprocess crop \
	--preprocess extreme \
	--data_augs True \
	--num_epochs 1 \
	--batch_size 50 \
	--learning_rate 0.0001 \
	--img_shape "100,50,3"

```

The results are saved in  `./` directory and the structure is given below. 
Tensorboard can be launched with `logs/fit` directory.


# Drone Followline Algorithms: Implementation and Baseline

It contains some deep learning regression models for Iris drone Line Following task.

The models implemented are derived from:
1. DeepPilot for Autonomous Drone Racing with Behaviour Metrics dataset

The algorithms are modular and can adapt to various other datasets. 

## Preparing Dataset

For DeepPilot, we use our custom datasets:
- Complete dataset: contains images with annotations from different circuits [https://drive.google.com/file/d/1Xdiu69DLj7lKK37F94qrUWsXkVg4ymGv/view?usp=sharing](https://drive.google.com/file/d/1Xdiu69DLj7lKK37F94qrUWsXkVg4ymGv/view?usp=sharing)
- Curves dataset: contains images with annotations from many_curves circuit: [https://drive.google.com/file/d/1zCJPFJRqCa34Q6jvktjDBY8Z49bIbvLJ/view?usp=sharing](https://drive.google.com/file/d/1zCJPFJRqCa34Q6jvktjDBY8Z49bIbvLJ/view?usp=sharing)
(To Be updated)

```
    DeepPilot                               # Extract PilotNet dataset here
    ├── complete_dataset                    # Extract PilotNet complete_dataset here           
    |   ├── Images/                         # Train and Test Images
    |   └── data.json                       # Annotations
    └── curves_dataset                      # Extract PilotNet curves_dataset here  
        ├── Images/                         # Train and Test Images
        └── data.json                       # Annotations
```

## Hyperparameters for the code

```
# For DeepPilot

-h, --help                            show this help message and exit
--data_dir            DATA_DIR        Directory to find Data
--preprocess		  PREPROCESSING	  Preprocessing information about cropping and extreme cases 
--base_dir            BASE_DIR        Directory to save everything
--comment             COMMENT         Comment to know the experiment
--data_augs           AUGMENTATIONS   Data augmentations
--num_epochs          NUM_EPOCHS      Number of Epochs
--lr                  LR              Learning rate for Policy Net
--test_split          TEST_SPLIT      Train test Split
--shuffle             SHUFFLE         Shuffle dataset
--batch_size          BATCH_SIZE      Batch size
--save_iter           SAVE_ITER       Iterations to save the model
--print_terminal      PRINT_TERMINAL  Print progress in terminal
--seed                SEED            Seed for reproducing

```

## Running the Code

```bash
source ~/pyenvs/dlstudio/bin/activate
cd DL_studio/Drone-FollowLine/

# For PilotNet

cd DeepPilot
python train.py --data_dir '../datasets/complete_dataset' \
	    --data_dir '../datasets/curves_only' \
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
```

The results are saved in the `./experiments/` directory and the structure is given below. 
Tensorboard can be launched with `./experiments/base_dir/log` directory.
