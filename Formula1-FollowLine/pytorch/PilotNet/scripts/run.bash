#!/bin/bash

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
	--comment 'All Augmentations trial v1' \
	--data_augs 'all' \
	--num_epochs 150 \
	--lr 3e-3 \
	--test_split 0.2 \
	--shuffle True \
	--batch_size 256 \
	--save_iter 50 \
	--print_terminal True \
	--seed 123