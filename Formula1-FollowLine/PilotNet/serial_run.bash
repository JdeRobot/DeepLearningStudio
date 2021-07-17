#!/bin/bash

sbatch scripts/run2.sbatch &&
sleep 11500 &&
sbatch scripts/run2.sbatch &&
sleep 11500 &&
sbatch scripts/run2.sbatch &&
sleep 11500 &&
sbatch scripts/run2.sbatch &
