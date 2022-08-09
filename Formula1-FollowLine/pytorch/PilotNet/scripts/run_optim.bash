#!/bin/bash

	# --data_dir '../datasets_opencv/many_curves_01_04_2022_clockwise_1' \
	# --data_dir '../datasets_opencv/nurburgring_01_04_2022_clockwise_1' \
	# --data_dir '../datasets_opencv/monaco_01_04_2022_clockwise_1' \
	# --data_dir '../datasets_opencv/extended_simple_circuit_01_04_2022_clockwise_1' \
	# --data_dir '../datasets_opencv/only_curves_01_04_2022/nurburgring_1/' \
	# --data_dir '../datasets_opencv/only_curves_01_04_2022/nurburgring_2/' \
	# --data_dir '../datasets_opencv/only_curves_01_04_2022/nurburgring_3/' \
	# --data_dir '../datasets_opencv/only_curves_01_04_2022/nurburgring_4/' \
	# --data_dir '../datasets_opencv/only_curves_01_04_2022/nurburgring_5/' \
	# --data_dir '../datasets_opencv/only_curves_01_04_2022/nurburgring_6/' \
	# --data_dir '../datasets_opencv/only_curves_01_04_2022/monaco_1/' \
	# --data_dir '../datasets_opencv/only_curves_01_04_2022/monaco_2/' \
	# --data_dir '../datasets_opencv/only_curves_01_04_2022/monaco_3/' \
	# --data_dir '../datasets_opencv/only_curves_01_04_2022/monaco_4/' \
	# --data_dir '../datasets_opencv/only_curves_01_04_2022/many_curves_1/' \
	# --data_dir '../datasets_opencv/only_curves_01_04_2022/many_curves_2/' \
	# --data_dir '../datasets_opencv/difficult_situations_01_04_2022/many_curves_1/' \
	# --data_dir '../datasets_opencv/difficult_situations_01_04_2022/many_curves_2/' \
	# --data_dir '../datasets_opencv/difficult_situations_01_04_2022/many_curves_3/' \
	# --data_dir '../datasets_opencv/difficult_situations_01_04_2022/many_curves_4/' \
	# --data_dir '../datasets_opencv/difficult_situations_01_04_2022/monaco_1/' \
	# --data_dir '../datasets_opencv/difficult_situations_01_04_2022/monaco_2/' \
	# --data_dir '../datasets_opencv/difficult_situations_01_04_2022/monaco_3/' \
	# --data_dir '../datasets_opencv/difficult_situations_01_04_2022/monaco_4/' \
	# --data_dir '../datasets_opencv/difficult_situations_01_04_2022/monaco_5/' \
	# --data_dir '../datasets_opencv/difficult_situations_01_04_2022/monaco_6/' \
	# --data_dir '../datasets_opencv/difficult_situations_01_04_2022/nurburgring_1/' \
	# --data_dir '../datasets_opencv/difficult_situations_01_04_2022/nurburgring_2/' \
	# --data_dir '../datasets_opencv/difficult_situations_01_04_2022/nurburgring_3/' \
	# --data_dir '../datasets_opencv/difficult_situations_01_04_2022/nurburgring_4/' \
	# --val_dir '../datasets_opencv/montmelo_12_05_2022_opencv_clockwise_1/' \
	# --val_dir '../datasets_opencv/montreal_12_05_2022_opencv_clockwise_1/' \
python optimize_models.py \
	--val_dir '../datasets_opencv/simple_circuit_01_04_2022_clockwise_1/' \
	--preprocess 'crop' \
	--preprocess 'extreme' \
	--data_augs 'all' \
	--model_dir experiments/best_case/trained_models/pilot_net_model_123.ckpt \
	--lr 3e-3 \
	--batch_size 256 \
	--eval_base True
	# --tech dynamic_quan
	# --num_epochs 150 \
	# --base_dir testcase \