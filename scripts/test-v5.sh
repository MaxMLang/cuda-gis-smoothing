#!/bin/bash
# test-v5.sh - runs all v5 smoothing configs
# set the number of nodes and processes per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short

# set max wallclock time
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

# set custom output and error file names
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# set name of job
#SBATCH --job-name=extended_neighborhoods_study

# use our reservation
#SBATCH --reservation=cuda2025

echo "=== Testing CUDA v5: Extended Neighborhoods ==="
echo

# Compile
echo "Compiling CUDA v5..."
make bin/hex_smooth_cuda_v5

# Run all configurations
echo -e "\n--- Configuration 1: First-order only, single variable ---"
./bin/hex_smooth_cuda_v5 data/hex_data.bin --single

echo -e "\n--- Configuration 2: First-order only, multi-variable fusion ---"
./bin/hex_smooth_cuda_v5 data/hex_data.bin

echo -e "\n--- Configuration 3: Both orders, single variable ---"
./bin/hex_smooth_cuda_v5 data/hex_data.bin --second-order --single

echo -e "\n--- Configuration 4: Both orders, multi-variable fusion ---"
./bin/hex_smooth_cuda_v5 data/hex_data.bin --second-order

echo -e "\n--- Configuration 5: Both orders, uniform weights, fusion ---"
./bin/hex_smooth_cuda_v5 data/hex_data.bin --second-order --uniform

echo -e "\n--- Running comprehensive benchmark ---"
./bin/hex_smooth_cuda_v5 data/hex_data.bin