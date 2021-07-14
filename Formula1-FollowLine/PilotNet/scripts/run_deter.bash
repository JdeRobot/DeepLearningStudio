#!/bin/bash

python train.py --data_dir '../datasets/complete_dataset' \
	--data_dir '../datasets/curves_only' \
	--base_dir 13Jul1 \
	--comment 'Labels not normalized; Selected Augmentations: gaussian, jitter' \
	--data_augs 'gaussian' \
	--data_augs 'jitter' \
	--num_epochs 300 \
	--lr 1e-4 \
	--test_split 0.2 \
	--shuffle True \
	--batch_size 256 \
	--save_iter 50 \
	--print_terminal True \
	--seed 123
