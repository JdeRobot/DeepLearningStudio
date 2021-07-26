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