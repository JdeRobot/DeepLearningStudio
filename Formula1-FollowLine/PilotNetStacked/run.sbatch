#!/bin/bash
#SBATCH --account=def-mtaylor3
#SBATCH -J 3_stacked
#SBATCH -o ./slurm/%j.out
#SBATCH -e ./slurm/%j.er
#SBATCH --gres=gpu:p100:2
#SBATCH --cpus-per-task=6  # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G
#SBATCH --time=0-24:00      # time (DD-HH:MM)
#SBATCH --mail-user=utkarsh75477@gmail.com
#SBATCH --mail-type=FAIL

module load StdEnv/2020
source /scratch/umishra/simrl/bin/activate

python train.py --data_dir '/scratch/umishra/complete_dataset' \
        --data_dir '/scratch/umishra/curves_only' \
        --base_dir 21Jul_extreme_graham \
        --comment 'Labels not normalized; Selected Augmentations: gaussian, jitter' \
        --data_augs 'gaussian' \
        --data_augs 'jitter' \
        --num_epochs 300 \
        --horizon 3 \
        --lr 3e-4 \
        --test_split 0.2 \
        --shuffle True \
        --batch_size 256 \
        --save_iter 50 \
        --print_terminal True \
        --seed 123

echo "Program finished"
