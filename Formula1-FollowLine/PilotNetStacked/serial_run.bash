#!/bin/bash

sbatch scripts/run.sbatch &&
sleep 11500 &&
sbatch scripts/run.sbatch &&
sleep 11500 &&
sbatch scripts/run.sbatch &&
sleep 11500 &&
sbatch scripts/run.sbatch &