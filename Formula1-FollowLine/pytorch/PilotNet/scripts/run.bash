#!/bin/bash

python train.py \
	--data_dir '../datasets_opencv/many_curves_01_04_2022_clockwise_1/' \
	--data_dir '../datasets_opencv/nurburgring_01_04_2022_clockwise_1/' \
	--data_dir '../datasets_opencv/monaco_01_04_2022_clockwise_1/' \
	--data_dir '../datasets_opencv/extended_simple_circuit_01_04_2022_clockwise_1/' \
	--data_dir '../datasets_opencv/only_curves_01_04_2022/nurburgring_1/' \
	--data_dir '../datasets_opencv/only_curves_01_04_2022/nurburgring_2/' \
	--data_dir '../datasets_opencv/only_curves_01_04_2022/nurburgring_3/' \
	--data_dir '../datasets_opencv/only_curves_01_04_2022/nurburgring_4/' \
	--data_dir '../datasets_opencv/only_curves_01_04_2022/nurburgring_5/' \
	--data_dir '../datasets_opencv/only_curves_01_04_2022/nurburgring_6/' \
	--data_dir '../datasets_opencv/only_curves_01_04_2022/monaco_1/' \
	--data_dir '../datasets_opencv/only_curves_01_04_2022/monaco_2/' \
	--data_dir '../datasets_opencv/only_curves_01_04_2022/monaco_3/' \
	--data_dir '../datasets_opencv/only_curves_01_04_2022/monaco_4/' \
	--data_dir '../datasets_opencv/only_curves_01_04_2022/many_curves_1/' \
	--data_dir '../datasets_opencv/only_curves_01_04_2022/many_curves_2/' \
	--data_dir '../datasets_opencv/difficult_situations_01_04_2022/many_curves_1/' \
	--data_dir '../datasets_opencv/difficult_situations_01_04_2022/many_curves_2/' \
	--data_dir '../datasets_opencv/difficult_situations_01_04_2022/many_curves_3/' \
	--data_dir '../datasets_opencv/difficult_situations_01_04_2022/many_curves_4/' \
	--data_dir '../datasets_opencv/difficult_situations_01_04_2022/monaco_1/' \
	--data_dir '../datasets_opencv/difficult_situations_01_04_2022/monaco_2/' \
	--data_dir '../datasets_opencv/difficult_situations_01_04_2022/monaco_3/' \
	--data_dir '../datasets_opencv/difficult_situations_01_04_2022/monaco_4/' \
	--data_dir '../datasets_opencv/difficult_situations_01_04_2022/monaco_5/' \
	--data_dir '../datasets_opencv/difficult_situations_01_04_2022/monaco_6/' \
	--data_dir '../datasets_opencv/difficult_situations_01_04_2022/nurburgring_1/' \
	--data_dir '../datasets_opencv/difficult_situations_01_04_2022/nurburgring_2/' \
	--data_dir '../datasets_opencv/difficult_situations_01_04_2022/nurburgring_3/' \
	--data_dir '../datasets_opencv/difficult_situations_01_04_2022/nurburgring_4/' \
	--test_dir '../datasets_opencv/montmelo_12_05_2022_opencv_clockwise_1/' \
	--test_dir '../datasets_opencv/montreal_12_05_2022_opencv_clockwise_1/' \
	--test_dir '../datasets_opencv/simple_circuit_01_04_2022_clockwise_1/' \
	--preprocess 'crop' \
	--preprocess 'extreme' \
	--base_dir retrain_best \
	--comment 'Retrain to best performance' \
	--data_augs 'all' \
	--num_epochs 50 \
	--lr 1e-4 \
	--val_split 0.1 \
	--shuffle True \
	--batch_size 1024 \
	--save_iter 100 \
	--print_terminal True \
	--seed 121


	# --data_augs ["gaussian", "jitter"] \