# Implementation Notes

Quick notes on each version and what I tried.

## CPU Versions

### Naive CPU (hex_smooth_naive.cpp)
- Simple nested loops, no optimization
- 705.30 μs baseline
- Row-major traversal
- Single-threaded

### Optimized CPU (hex_smooth_optimized.cpp)
- OpenMP with 4 threads
- 820.60 μs (slower than naive!)
- Problems:
  - OpenMP overhead too high for this size
  - Thread creation/sync costs
  - Chunk processing hurt cache locality

## CUDA Versions

### v1 - Naive GPU Port (hex_smooth_cuda_v1.cu)
- Direct translation from CPU
- One thread per hexagon
- Uncoalesced memory access
- 15.77 μs (44.7x speedup)
- Shows GPU power even with bad memory patterns

### v2 - Coalesced Memory (hex_smooth_cuda_v2.cu)
- Changed neighbor array layout to column-major
- All threads in warp access neighbors[idx + i * n_hexagons] together
- Padded to MAX_NEIGHBORS=6
- 14.26 μs (49.5x speedup, 10% improvement)
- Shared memory version: 16.06 μs (slower, limited reuse)

### v3 - Advanced Memory (hex_smooth_cuda_v3.cu)
- Texture memory: 15.87 μs (no improvement)
  - Texture cache didn't help scattered access
- Warp shuffle: 27.42 μs (much worse!)
  - Most neighbors not in same warp
  - Divergence from checking warp boundaries
  - Falls back to global memory most of time
- Lesson: fancy features aren't always better

### v4 - Kernel Fusion (hex_smooth_cuda_v4.cu)
- Process NDVI, MNDWI, EVI, NDWI in single kernel
- Reuse neighbor lookups across all variables
- Results:
  - With Morton codes: 22.45 μs
  - With shared memory: 24.05 μs
  - Without fancy stuff: 18.23 μs (best!)
  - Per-variable time: 4.56 μs
- 3.1x improvement over separate processing

### v5 - 2nd Order Neighborhood (hex_smooth_cuda_v5.cu)
- Extended to neighbors-of-neighbors for wider smoothing
- Gaussian weights: center (1.0), 1st-order (0.607), 2nd-order (0.135)
- Calculate 2nd-order neighbors on CPU first (avg 3.2 new neighbors per hex)
- Results:
  - 1st-order fusion: 17.27 μs (4.32 μs per variable)
  - 2nd-order fusion: 30.82 μs (7.71 μs per variable)
  - Uniform weights: 30.88 μs (no difference)

## Memory Layout Changes

### Coalesced Access
```
Original (row-major):
neighbors[hex_id][neighbor_idx] = neighbor_hex_id

Changed to (column-major):
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
    
    // Process all variables
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
1. Memory bandwidth - irregular access patterns
2. Warp divergence - neighbors scattered across warps
3. Cache misses - poor spatial locality
4. Thread utilization - some threads have fewer neighbors

### What Worked vs What Didn't
- Coalescing: high impact (10% improvement)
- Kernel fusion: very high impact (3.1x improvement)
- Texture memory: no impact (wrong pattern)
- Warp shuffle: negative impact (divergence overhead)
- Shared memory: limited impact (not enough reuse)

## Scaling Thoughts

### For Bigger Datasets (millions of hexagons)
- OpenMP would finally beat naive CPU
- Shared memory becomes worthwhile
- Spatial reordering would show benefits
- Multi-GPU becomes essential
- Out-of-core processing for billion+ hexagons

### Memory Scaling
- 75k hexagons: ~5MB
- 1M hexagons: ~70MB (fits in L2)
- 10M hexagons: ~700MB (fits in GPU)
- 100M hexagons: ~7GB (needs management)
- 1B hexagons: ~70GB (needs multi-GPU) 