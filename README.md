# Hexagonal Grid Smoothing Case Study

[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-zone)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![GIS](https://img.shields.io/badge/GIS-Satellite%20Data-orange.svg)](https://en.wikipedia.org/wiki/Geographic_information_system)

## Overview

This is my final project for the CUDA Programming Course at Oxford (Prof. Mike Giles).

Course site: https://people.maths.ox.ac.uk/gilesm/cuda/

## What this does

- Smooths noisy NDVI satellite data from Uganda using hex grids
- Compares CPU vs GPU (up to 154x faster!)
- Shows how I went from basic to optimized CUDA
- Tries different memory tricks

## Dataset

- 74,811 hexagons from Uganda (3 districts)
- Avg 2.75 neighbors per hex (max 6)
- Goal: smooth NDVI using neighbors
- Problem: neighbors are all over memory

## Repository Structure

```
src/
├── cuda/          # CUDA v1-v5
├── cpu/           # CPU code
scripts/           # Scripts
data/              # Input files
results/           # Outputs and benchmarks
bin/               # Compiled executables
docs/              # Notes
```

## Performance Results

### CPU
- Naive: 705 μs (baseline)
- OpenMP: 820 μs (actually slower!)

### CUDA
- v1: 15.8 μs (44x faster)
- v2: 14.3 μs (49x faster)
- v3: 15.9 μs (no real change)
- v3 (shuffle): 27.4 μs (worse)
- v4: 18.2 μs (4 vars at once)
- v4 (per var): 4.6 μs (154x faster!)

### v5 - 2nd Order
- 1st-order: 17.3 μs (4.3 μs/var)
- 2nd-order: 30.8 μs (7.7 μs/var)

## Key Optimizations

### v1 - Naive GPU
- Just ported CPU code
- One thread per hex
- Bad memory access
- Still way faster than CPU

### v2 - Coalesced
- Changed neighbor array layout
- Threads in warp access together
- Padded to 6 neighbors
- 10% better than v1

### v3 - Advanced
- Texture memory: no help
- Warp shuffle: much worse

### v4 - Kernel Fusion
- Process all 4 vars in one go
- Reuse neighbor lookups
- 3x faster than doing vars separately

### v5 - 2nd Order
- Looks at neighbors-of-neighbors
- Gaussian weights: center (1), 1st (0.6), 2nd (0.13)
- More work but fusion still helps

## Lessons Learned

1. Simple tricks > fancy ones
2. Kernel fusion is awesome for multi-var
3. Shared memory only helps if reused a lot
4. Always measure, don't just guess
5. Only 2.75 neighbors per hex limits what you can do

## Building

```bash
# Build everything
make all
# Just CPU
make cpu
# Just CUDA
make cuda
# Clean up
make clean
```

## Running

```bash
# Main run
./scripts/case-study.sh
# Test v5
./scripts/test-v5.sh
```

## Scaling Notes

For millions of hexes:
- OpenMP finally wins
- Shared memory starts to matter
- Reordering helps
- Need multi-GPU

Memory:
- 75k: ~5MB
- 1M: ~70MB
- 10M: ~700MB
- 100M: ~7GB
- 1B: ~70GB

## Course Context

This was for the CUDA course at Oxford. Covered:
- Parallel basics
- Memory tricks (global, shared, etc)
- Tuning and profiling
- Fancy stuff (warp shuffle, atomics)
- Real-world uses

## Acknowledgments

- Prof. Mike Giles & Wes Armour (teaching)
- Oxford Math Institute
- NVIDIA for CUDA
- Oxford ARC for GPU access