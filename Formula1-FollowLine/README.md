# Formula1 Followline Algorithms: Implementation and Baseline

Here you will find some deep learning regression models for following a line using a Formula 1 car.

The algorithms are modular and can adapt to other datasets. They are both implemented in pytorch and tensorflow.

## Preparing Dataset 

Two possible datasets:
* The dataset for training is available in the following [link](https://drive.google.com/file/d/1EL2Pzzdoj7jLRRi9DXNDVCwZZ0zGGu7H/view?usp=sharing).
* The **10fps** dataset for training is available in the following [link](https://drive.google.com/file/d/1NxSsbpUqlRisMFSVtIiVZV-QSck3ohKU/view?usp=sharing). It includes timestamp for each sample.

It's generated from running an explicitly programmed brain over different circuits. It's divided as follows:


| Circuit      | Direction | Number of images-annotations |
| ----------- | ----------- | ----------- |
| Simple circuit      | Clockwise       | 2190       |
| Simple circuit   | Anticlockwise        | 2432      |
| Many curves      | Clockwise       | 4653       |
| Many curves  | Anticlockwise        | 5165       |
| Extended simple circuit      | Clockwise       | 3590       |
| Extended simple circuit  | Anticlockwise        | 3509       |
| Monaco      | Clockwise       | 5603	|
| Monaco  | Anticlockwise        | 5206       |
| Nurburgring      | Clockwise       | 3808	|
| Nurburgring  | Anticlockwise        | 4045      |
| Only curves      | Recorded curves from different circuits       | 3008	|
| Difficult situations 1-2     | Recorded difficult situations from different circuits       | 4292	|
| Montmeló      | Clockwise       | 10507	|
| **TOTAL**      | -       | **92280** |


The model weights avaiable here only used the following part of the whole dataset for training/validation:


| Circuit      | Direction |
| ----------- | ----------- |
| Extended simple circuit      | Clockwise       |
| Many curves      | Clockwise       |
| Nurburgring      | Clockwise       |
| Monaco      | Clockwise       |
| Only curves      | Recorded curves from different circuits       |
| Difficult situations 1-2     | Recorded difficult situations from different circuits       |


The dataset v and W values range is:

| V min | V max | W min | W min |
| ----------- | ----------- |----------- |----------- |
| 6.5 |  24 | -7.1 | 7.1 |


# Pytorch <img src="https://pytorch.org/assets/images/pytorch-logo.png" alt="Pytorch logo" width="50"/> 

The models implemented are derived from:
1. PilotNet for Autonomous Driving with Behaviour Metrics dataset
2. PilotNetStacked as an extension of PilotNet with stacked images

## Preparing Dataset

Extract the dataset and place it on the following fashion:

```
    PilotNet                                # Extract PilotNet dataset here
    ├── complete_dataset                    # Extract PilotNet complete_dataset here           
    |   ├── Images/                         # Train and Test Images
    |   └── data.json                       # Annotations
    └── curves_dataset                      # Extract PilotNet curves_dataset here  
        ├── Images/                         # Train and Test Images
        └── data.json                       # Annotations
```

For new dataset, place it inside `/pytorch` directory.

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
python train.py --data_dir '../dataset_opencv/extended_simple_circuit_01_04_2022_anticlockwise_1' \
		--data_dir '../dataset_opencv/extended_simple_circuit_01_04_2022_clockwise_1' \
		--data_dir '../dataset_opencv/many_curves_01_04_2022_anticlockwise_1' \
		--data_dir '../dataset_opencv/many_curves_01_04_2022_clockwise_1' \
		--data_dir '../dataset_opencv/monaco_01_04_2022_anticlockwise_1' \
		--data_dir '../dataset_opencv/monaco_01_04_2022_clockwise_1' \
		--data_dir '../dataset_opencv/nurburgring_01_04_2022_anticlockwise_1' \
		--data_dir '../dataset_opencv/nurburgring_01_04_2022_clockwise_1' \
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
--data_augs           AUGMENTATIONS   Data augmentations (0=No / 1=Normal / 2=Normal+Weather changes)
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
python3 train.py --data_dir ../../../../datasets_opencv/ \
	--preprocess crop \
	--preprocess extreme \
	--data_augs 2 \
	--num_epochs 1 \
	--learning_rate 0.0001 \
	--batch_size 50 \
	--img_shape "200,66,3"
	
	
# For DeepestLSTMTinyPilotNet
cd DeepestLSTMTinyPilotNet
python3 train.py --data_dir ../../../../datasets_opencv/ \
    --preprocess crop \
    --preprocess extreme \
    --data_augs 2 \
    --num_epochs 1 \
    --learning_rate 0.0001 \
    --batch_size 50 \
    --img_shape "100,50,3"

# For memDCCP
cd memDCCP
python3 train.py --data_dir ../../../../datasets_opencv/ \
    --preprocess crop \
    --preprocess extreme \
    --data_augs 2 \
    --num_epochs 1 \
    --learning_rate 0.0001 \
    --batch_size 50 \
    --img_shape "3,100,50,3"
    
# For PilotNetx3
cd PilotNetx3
python3 train.py --data_dir ../../../../datasets_opencv/ \
    --preprocess crop \
    --preprocess extreme \
    --data_augs 2 \
    --num_epochs 1 \
    --learning_rate 0.0001 \
    --batch_size 50 \
    --img_shape "3,100,50,3"

```

The results are saved in  `./` directory and the structure is given below. 
Tensorboard can be launched with `logs/fit` directory.
