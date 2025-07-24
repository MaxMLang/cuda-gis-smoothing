# Implementation Notes

Quick notes on each version and what I tried.

## CPU Versions

### Naive CPU (hex_smooth_naive.cpp)
- Just loops, no tricks
- 705 μs baseline
- Row-major
- Single-threaded

### Optimized CPU (hex_smooth_optimized.cpp)
- OpenMP, 4 threads
- 820 μs (slower than naive!)
- Why? Overhead too high, chunking hurts cache

## CUDA Versions

### v1 - Naive GPU (hex_smooth_cuda_v1.cu)
- Just ported CPU code
- One thread per hex
- Bad memory access
- 15.8 μs (44x faster)
- GPU is fast even with bad memory

### v2 - Coalesced (hex_smooth_cuda_v2.cu)
- Changed neighbor array to column-major
- Threads in warp access together
- Padded to 6 neighbors
- 14.3 μs (49x, 10% better)
- Shared memory: 16.1 μs (not worth it)

### v3 - Advanced (hex_smooth_cuda_v3.cu)
- Texture: 15.9 μs (no help)
  - Texture cache didn't help
- Warp shuffle: 27.4 μs (worse)
  - Neighbors not in same warp
  - Falls back to global most of the time
- Lesson: fancy stuff isn't always better

### v4 - Kernel Fusion (hex_smooth_cuda_v4.cu)
- Process all 4 vars in one kernel
- Reuse neighbor lookups
- Results:
  - Morton codes: 22.5 μs
  - Shared memory: 24.1 μs
  - No tricks: 18.2 μs (best)
  - Per-var: 4.6 μs
- 3x better than separate

### v5 - 2nd Order (hex_smooth_cuda_v5.cu)
- Looks at neighbors-of-neighbors
- Gaussian weights: center (1), 1st (0.6), 2nd (0.13)
- 2nd-order neighbors on CPU first (avg 3.2 per hex)
- Results:
  - 1st-order fusion: 17.3 μs (4.3 μs/var)
  - 2nd-order fusion: 30.8 μs (7.7 μs/var)
  - Uniform: 30.9 μs (no diff)

## Memory Layout Changes

### Coalesced Access
```
Original:
neighbors[hex_id][neighbor_idx] = neighbor_hex_id

Now:
neighbors[neighbor_idx * n_hexagons + hex_id] = neighbor_hex_id
```

### Kernel Fusion Code
```cuda
__global__ void smooth_all_variables(
    float* ndvi, float* mndwi, float* evi, float* ndwi,
    const int* neighbors, const int* n_neighbors,
    int n_hexagons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_hexagons) return;
    // Load neighbors once
    int my_neighbors[MAX_NEIGHBORS];
    int my_n_neighbors = n_neighbors[idx];
    for (int i = 0; i < my_n_neighbors; i++) {
        my_neighbors[i] = neighbors[i * n_hexagons + idx];
    }
    // Process all vars
    float ndvi_sum = ndvi[idx];
    float mndwi_sum = mndwi[idx];
    float evi_sum = evi[idx];
    float ndwi_sum = ndwi[idx];
    for (int i = 0; i < my_n_neighbors; i++) {
        int neighbor = my_neighbors[i];
        ndvi_sum += ndvi[neighbor];
        mndwi_sum += mndwi[neighbor];
        evi_sum += evi[neighbor];
        ndwi_sum += ndwi[neighbor];
    }
    // Store results
    ndvi[idx] = ndvi_sum / (my_n_neighbors + 1);
    mndwi[idx] = mndwi_sum / (my_n_neighbors + 1);
    evi[idx] = evi_sum / (my_n_neighbors + 1);
    ndwi[idx] = ndwi_sum / (my_n_neighbors + 1);
}
```

## Performance Issues Found

### Bottlenecks
1. Memory bandwidth - weird access
2. Warp divergence - neighbors all over
3. Cache misses - bad locality
4. Some threads have fewer neighbors

### What Worked vs What Didn't
- Coalescing: good (10% better)
- Kernel fusion: really good (3x)
- Texture: no help
- Warp shuffle: bad (overhead)
- Shared memory: not enough reuse

## Scaling Thoughts

### For Big Datasets
- OpenMP finally wins
- Shared memory helps
- Reordering helps
- Need multi-GPU
- Out-of-core for billions

### Memory Scaling
- 75k: ~5MB
- 1M: ~70MB (fits in L2)
- 10M: ~700MB (fits in GPU)
- 100M: ~7GB (need management)
- 1B: ~70GB (need multi-GPU) 