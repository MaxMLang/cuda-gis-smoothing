# Hexagonal Grid Smoothing Case Study

[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-zone)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![GIS](https://img.shields.io/badge/GIS-Satellite%20Data-orange.svg)](https://en.wikipedia.org/wiki/Geographic_information_system)

## Overview

My capstone/case study final project for the CUDA Programming Course taught by Prof. Mike Giles at Oxford.

Course website: https://people.maths.ox.ac.uk/gilesm/cuda/

## What this does

- Smooths noisy NDVI satellite data from Uganda using hexagonal grids
- Compares CPU vs GPU performance (got up to 154.6x speedup!)
- Shows evolution from naive code to optimized CUDA kernels
- Tests different memory access patterns and optimizations

## Dataset

- 74,811 hexagons from Uganda (3 districts)
- Average 2.75 neighbors per hexagon (max 6)
- Goal: smooth NDVI measurements using neighbor relationships
- Problem: neighbors scattered in memory = bad access patterns

## Repository Structure

```
src/
├── cuda/          # CUDA versions v1-v5
├── cpu/           # CPU implementations
scripts/           # Build scripts
data/              # Input files
results/           # Outputs and benchmarks
bin/               # Compiled executables
docs/              # Notes
```

## Performance Results

### CPU
- Naive: 705.30 μs (baseline)
- OpenMP "optimized": 820.60 μs (actually slower!)

### CUDA Versions
- v1 (naive): 15.77 μs (44.7x speedup)
- v2 (coalesced): 14.26 μs (49.5x speedup)
- v3 (texture): 15.87 μs (no improvement)
- v3 (shuffle): 27.42 μs (much worse!)
- v4 (fusion): 18.23 μs (38.7x for 4 variables)
- v4 (per variable): 4.56 μs (154.6x speedup!)

### v5 - 2nd Order Neighborhood
- 1st-order fusion: 17.27 μs (4.32 μs per variable)
- 2nd-order fusion: 30.82 μs (7.71 μs per variable)

## Key Optimizations

### v1 - Naive GPU Port
- Direct translation from CPU
- One thread per hexagon
- Uncoalesced memory access
- Still 44.7x faster than CPU

### v2 - Coalesced Memory Access
- Changed neighbor array layout to column-major
- All threads in warp access neighbors together
- Padded to MAX_NEIGHBORS=6
- 10% improvement over v1

### v3 - Advanced Memory
- Texture memory: no improvement (wrong access pattern)
- Warp shuffle: much worse (divergence issues)

### v4 - Kernel Fusion
- Process NDVI, MNDWI, EVI, NDWI in single kernel
- Reuse neighbor lookups across all variables
- 3.1x improvement over separate processing

### v5 - 2nd Order Neighborhood
- Extended to neighbors-of-neighbors
- Gaussian weights: center (1.0), 1st-order (0.607), 2nd-order (0.135)
- More work but fusion still helps

## Lessons Learned

1. Simple optimizations beat complex ones (coalescing > shuffle)
2. Kernel fusion is huge win for multi-variable processing
3. Shared memory needs sufficient reuse to be worth it
4. Always measure - theoretical optimizations may backfire
5. Problem characteristics matter - only 2.75 avg neighbors limits optimization

## Building

```bash
# Build everything
make all

# Build only CPU
make cpu

# Build only CUDA
make cuda

# Clean
make clean
```

## Running

```bash
# Main case study
./scripts/case-study.sh

# Test v5
./scripts/test-v5.sh
```

## Scaling Notes

For millions of hexagons:
- OpenMP would finally beat naive CPU
- Shared memory becomes worthwhile
- Spatial reordering would show benefits
- Multi-GPU becomes essential

Memory requirements:
- 75k hexagons: ~5MB
- 1M hexagons: ~70MB
- 10M hexagons: ~700MB
- 100M hexagons: ~7GB
- 1B hexagons: ~70GB

## Course Context

This was for the CUDA Programming Course at Oxford covering:
- Parallel computing fundamentals
- Memory optimization (global, shared, constant, texture)
- Performance tuning and profiling
- Advanced features (warp shuffles, atomics, libraries)
- Real-world applications

## Acknowledgments

- Prof. Mike Giles and Prof. Wes Armour for teaching the course
- Oxford Mathematical Institute for hosting
- NVIDIA for CUDA tools
- Oxford ARC for GPU server access