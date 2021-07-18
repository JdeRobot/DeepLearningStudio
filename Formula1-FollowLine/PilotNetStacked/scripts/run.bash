
for horizon in 2 3 4; do
	python train.py \
		--data_dir '../datasets/curves_only' \
		--base_dir testcase \
		--comment 'Selected Augmentations: gaussian' \
		--data_augs 'gaussian' \
		--num_epochs 150 \
		--horizon ${horizon} \
		--lr 1e-3 \
		--test_split 0.2 \
		--shuffle True \
		--batch_size 16 \
		--save_iter 50 \
		--print_terminal True \
		--seed 123
done
