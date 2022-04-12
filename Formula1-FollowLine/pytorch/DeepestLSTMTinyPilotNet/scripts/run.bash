#!/bin/bash

python train.py --data_dir '../datasets' \
		--preprocess 'crop' \
		--preprocess 'extreme' \
	    --base_dir testcase \
	    --comment 'Selected Augmentations: gaussian, affine' \
	    --data_augs 'gaussian' \
	    --data_augs 'affine' \
	    --num_epochs 150 \
	    --lr 1e-4 \
	    --batch_size 64 \
	    --save_iter 10 \
	    --print_terminal True \
        --img_shape "100,50,3" \
	    --seed 123  
	    # --test_split 0.2 \
	    # --shuffle True \