#!/bin/bash

python train.py --data_dir '../datasets/complete_dataset' \
	--data_dir '../datasets/curves_only' \
	--preprocess 'crop' \
	--preprocess 'extreme' \
	--base_dir testcase \
	--comment 'Random Augmentations trial' \
	--data_augs 'auto' \
	--num_epochs 150 \
	--lr 3e-3 \
	--test_split 0.2 \
	--shuffle True \
	--batch_size 256 \
	--save_iter 50 \
	--print_terminal True \
	--seed 123