#!/bin/bash

python3 train.py --data_dir ../../../../datasets_opencv/ \
	--preprocess crop \
	--preprocess extreme \
	--data_augs 2 \
	--num_epochs 1 \
	--learning_rate 0.0001 \
	--batch_size 50 \
	--img_shape "200,66,3"

# python3 optimize_models.py --data_dir ../datasets_opencv/ \
# 	--preprocess crop \
# 	--preprocess extreme \
# 	--data_augs 2 \
# 	--img_shape "200,66,3" \
# 	--batch_size 64 \
# 	--model_path ../trained_models/pilotnet.h5 \
# 	--model_name pilotnet \
# 	--eval_base True \
# 	--tech dynamic_quan
