#!/bin/bash

python3 train.py --data_dir ../complete_dataset/ \
	--preprocess crop \
	--preprocess extreme \
	--data_augs True \
	--num_epochs 1 \
	--batch_size 50 \
	--learning_rate 0.0001 \
	--img_shape "100,50,3"