#!/bin/bash

injection_attempt="$1"

project=phenom-distribution-fit
pathtoscratch="$HOME/scratch/$project/data/raw"
current_time=$(date +"%Y%m%d%H%M%S")
output_directory="$pathtoscratch/flowmc-$current_time"
mkdir -p "$output_directory"
echo "Run label: flowmc-$current_time"

source activate phenom-distribution-fit
export LD_LIBRARY_PATH=/opt/conda/envs/phenom-distribution-fit/lib/:$LD_LIBRARY_PATH

nvidia-smi
nvcc --version

echo "About to start Python script. Inspect the .err and .out Slurm files for details."
python3.10 inference.py "$injection_attempt" "$output_directory"
echo "Python process ended."