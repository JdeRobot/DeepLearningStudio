#!/bin/bash

python3 train.py --data_dir ./../ \
	--preprocess crop \
	--preprocess extreme \
	--data_augs True \
	--num_epochs 150 \
	--batch_size 50 \
	--learning_rate 0.0001 \
	--img_shape "100,50,3"