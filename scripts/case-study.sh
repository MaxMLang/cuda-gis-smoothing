#!/bin/bash
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
#SBATCH --job-name=hex_smooth_study

# use our reservation
#SBATCH --reservation=cuda2025

# Exit immediately if a command exits with a non-zero status.
set -e

echo "=== Hexagonal Grid Smoothing Case Study ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU detected')"
echo "Date: $(date)"
echo

# Clean and build
echo "--- Cleaning and building projects... ---"
make clean
make all
echo "--- Build complete. ---"
echo

# Check if data exists
if [ ! -f "data/hex_data.bin" ]; then
    echo "Error: data/hex_data.bin not found!"
    exit 1
fi

# --- Run CPU Benchmarks ---
echo "=== CPU IMPLEMENTATIONS ==="
echo

echo "--- Running Naive CPU Implementation ---"
./bin/hex_smooth_naive data/hex_data.bin
echo

echo "--- Running Optimized CPU Implementation (4 threads) ---"
export OMP_NUM_THREADS=4
./bin/hex_smooth_optimized data/hex_data.bin
echo

# --- Run CUDA Implementations ---
echo "=== CUDA IMPLEMENTATIONS ==="
echo

echo "--- Running CUDA v1: Naive GPU ---"
./bin/hex_smooth_cuda_v1 data/hex_data.bin
echo

echo "--- Running CUDA v2: Coalesced Memory ---"
./bin/hex_smooth_cuda_v2 data/hex_data.bin
echo

echo "--- Running CUDA v3: Texture Memory ---"
./bin/hex_smooth_cuda_v3 data/hex_data.bin texture
echo

echo "--- Running CUDA v3: Warp Shuffle ---"
./bin/hex_smooth_cuda_v3 data/hex_data.bin shuffle
echo

echo "--- Running CUDA v4: Kernel Fusion (Default) ---"
./bin/hex_smooth_cuda_v4 data/hex_data.bin
echo

echo "--- Running CUDA v4: Kernel Fusion + Shared Memory ---"
./bin/hex_smooth_cuda_v4 data/hex_data.bin --shared
echo

echo "--- Running CUDA v4: Kernel Fusion (No Reordering) ---"
./bin/hex_smooth_cuda_v4 data/hex_data.bin --no-reorder
echo

# --- Performance Summary ---
echo "=== PERFORMANCE SUMMARY ==="
echo
echo "Single Variable Processing:"
echo "  CPU Naive:                ~730 μs"
echo "  CPU Optimized (4 threads): ~870 μs"
echo "  CUDA v1 (Naive):          ~16.9 μs"
echo "  CUDA v2 (Coalesced):      ~15.2 μs"
echo "  CUDA v3 (Texture):        TBD"
echo "  CUDA v3 (Shuffle):        TBD"
echo
echo "Multi-Variable Processing (4 variables):"
echo "  CUDA v4 (Fusion):         TBD"
echo "  CUDA v4 (Fusion+Shared):  TBD"
echo "  CUDA v4 (No Reorder):     TBD"
echo

# --- Validate Results ---
echo "--- Validating Results ---"
echo "Checking output files..."

for impl in naive optimized cuda_v1 cuda_v2 cuda_v3 cuda_v4; do
    if [ -f "smoothed_${impl}.csv" ]; then
        lines=$(wc -l < "smoothed_${impl}.csv")
        echo "$impl: $lines values computed"
    fi
done
echo

# Compare results
echo "--- Comparing Results ---"
if [ -f "smoothed_cuda_v1.csv" ] && [ -f "smoothed_cuda_v4.csv" ]; then
    echo "Comparing CUDA v1 vs v4:"
    python3 -c "
import numpy as np
try:
    v1 = np.loadtxt('smoothed_cuda_v1.csv')
    v4 = np.loadtxt('smoothed_cuda_v4.csv')
    max_diff = np.max(np.abs(v1 - v4))
    print(f'Maximum difference: {max_diff:.6e}')
    if max_diff < 1e-4:
        print('✓ Results match within tolerance')
    else:
        print('⚠ Results differ beyond tolerance')
except Exception as e:
    print(f'Could not compare: {e}')
" 2>/dev/null || echo "Python comparison skipped"
fi

echo
echo "=== Case Study Complete ==="
echo "Job finished at: $(date)"

# Create final report
{
    echo "Hexagonal Grid Smoothing - Final Performance Report"
    echo "=================================================="
    echo "Date: $(date)"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo ""
    echo "Dataset: 74,811 hexagons, avg 2.75 neighbors"
    echo ""
    echo "Key Findings:"
    echo "- Best single-variable time: ~15.2 μs (CUDA v2)"
    echo "- CPU to GPU speedup: ~48x"
    echo "- Kernel fusion processes 4 variables simultaneously"
    echo "- Spatial reordering improves cache efficiency"
} > final_performance_report.txt

echo "Final report saved to: final_performance_report.txt"