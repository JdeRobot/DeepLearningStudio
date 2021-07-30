#!/bin/bash

python train.py \
	--data_dir '/home/utkubuntu/GSoC2021/datasets/drone_dataset/drone_models/complete_dataset' \
	--data_dir '/home/utkubuntu/GSoC2021/datasets/drone_dataset/drone_models/curves_only' \
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
	--batch_size 16 \
	--save_iter 50 \
	--print_terminal True \
	--seed 123