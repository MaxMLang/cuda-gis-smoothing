// hex_smooth_cuda_v4.cu - Kernel fusion and spatial reordering
#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << error << " \"" << cudaGetErrorString(error) << "\"" << std::endl; \
            exit(1); \
        } \
    } while(0)

// Constants
constexpr int MAX_NEIGHBORS = 6;
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_VARIABLES = 4;  // NDVI, MNDWI, EVI, NDWI

// Morton code (Z-order) for spatial reordering
__host__ __device__ inline uint32_t morton2D(uint16_t x, uint16_t y) {
    uint32_t xx = x;
    uint32_t yy = y;
    
    xx = (xx | (xx << 8)) & 0x00FF00FF;
    xx = (xx | (xx << 4)) & 0x0F0F0F0F;
    xx = (xx | (xx << 2)) & 0x33333333;
    xx = (xx | (xx << 1)) & 0x55555555;
    
    yy = (yy | (yy << 8)) & 0x00FF00FF;
    yy = (yy | (yy << 4)) & 0x0F0F0F0F;
    yy = (yy | (yy << 2)) & 0x33333333;
    yy = (yy | (yy << 1)) & 0x55555555;
    
    return xx | (yy << 1);
}

// Data structure for multiple variables
struct HexagonGPUv4 {
    float* values[NUM_VARIABLES];      // NDVI, MNDWI, EVI, NDWI
    int* neighbor_indices;             // Neighbors, reordered
    float* smoothed_values[NUM_VARIABLES];
    int* neighbor_counts;
    int* reorder_map;                  // Maps old to new
    int* inverse_reorder_map;          // Maps new to old
    int n_hexagons;
    int padded_neighbors_size;
};

// Fused kernel - process all variables in one pass
__global__ void smoothMultiVariableFused(
    float* __restrict__ values_ndvi,
    float* __restrict__ values_mndwi,
    float* __restrict__ values_evi,
    float* __restrict__ values_ndwi,
    const int* __restrict__ neighbor_indices,
    const int* __restrict__ neighbor_counts,
    float* __restrict__ smoothed_ndvi,
    float* __restrict__ smoothed_mndwi,
    float* __restrict__ smoothed_evi,
    float* __restrict__ smoothed_ndwi,
    int n_hexagons)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_hexagons) return;
    
    // Load all center values
    float4 center_vals = make_float4(
        values_ndvi[idx],
        values_mndwi[idx],
        values_evi[idx],
        values_ndwi[idx]
    );
    
    float4 sums = center_vals;
    int count = 1;
    
    int n_neighbors = neighbor_counts[idx];
    
    // Process all neighbors for all variables simultaneously
    #pragma unroll
    for (int i = 0; i < MAX_NEIGHBORS; i++) {
        if (i < n_neighbors) {
            int neighbor_idx = neighbor_indices[idx + i * n_hexagons];
            
            if (neighbor_idx >= 0 && neighbor_idx < n_hexagons) {
                // Vectorized load for all 4 variables
                sums.x += values_ndvi[neighbor_idx];
                sums.y += values_mndwi[neighbor_idx];
                sums.z += values_evi[neighbor_idx];
                sums.w += values_ndwi[neighbor_idx];
                count++;
            }
        }
    }
    
    // Vectorized division and store
    float inv_count = 1.0f / count;
    smoothed_ndvi[idx] = sums.x * inv_count;
    smoothed_mndwi[idx] = sums.y * inv_count;
    smoothed_evi[idx] = sums.z * inv_count;
    smoothed_ndwi[idx] = sums.w * inv_count;
}

// Shared memory version with fusion
__global__ void smoothMultiVariableFusedShared(
    float* __restrict__ values_ndvi,
    float* __restrict__ values_mndwi,
    float* __restrict__ values_evi,
    float* __restrict__ values_ndwi,
    const int* __restrict__ neighbor_indices,
    const int* __restrict__ neighbor_counts,
    float* __restrict__ smoothed_ndvi,
    float* __restrict__ smoothed_mndwi,
    float* __restrict__ smoothed_evi,
    float* __restrict__ smoothed_ndwi,
    int n_hexagons)
{
    extern __shared__ float shared_data[];
    
    // Divide shared memory among variables
    float* shared_ndvi = shared_data;
    float* shared_mndwi = shared_ndvi + BLOCK_SIZE;
    float* shared_evi = shared_mndwi + BLOCK_SIZE;
    float* shared_ndwi = shared_evi + BLOCK_SIZE;
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Load all variables into shared memory
    if (idx < n_hexagons) {
        shared_ndvi[tid] = values_ndvi[idx];
        shared_mndwi[tid] = values_mndwi[idx];
        shared_evi[tid] = values_evi[idx];
        shared_ndwi[tid] = values_ndwi[idx];
    }
    __syncthreads();
    
    if (idx >= n_hexagons) return;
    
    float4 sums = make_float4(
        shared_ndvi[tid],
        shared_mndwi[tid],
        shared_evi[tid],
        shared_ndwi[tid]
    );
    int count = 1;
    
    int n_neighbors = neighbor_counts[idx];
    
    #pragma unroll
    for (int i = 0; i < MAX_NEIGHBORS; i++) {
        if (i < n_neighbors) {
            int neighbor_idx = neighbor_indices[idx + i * n_hexagons];
            
            if (neighbor_idx >= 0 && neighbor_idx < n_hexagons) {
                int neighbor_block_idx = neighbor_idx - blockIdx.x * blockDim.x;
                
                if (neighbor_block_idx >= 0 && neighbor_block_idx < blockDim.x) {
                    // Use shared memory
                    sums.x += shared_ndvi[neighbor_block_idx];
                    sums.y += shared_mndwi[neighbor_block_idx];
                    sums.z += shared_evi[neighbor_block_idx];
                    sums.w += shared_ndwi[neighbor_block_idx];
                } else {
                    // Fall back to global memory
                    sums.x += values_ndvi[neighbor_idx];
                    sums.y += values_mndwi[neighbor_idx];
                    sums.z += values_evi[neighbor_idx];
                    sums.w += values_ndwi[neighbor_idx];
                }
                count++;
            }
        }
    }
    
    float inv_count = 1.0f / count;
    smoothed_ndvi[idx] = sums.x * inv_count;
    smoothed_mndwi[idx] = sums.y * inv_count;
    smoothed_evi[idx] = sums.z * inv_count;
    smoothed_ndwi[idx] = sums.w * inv_count;
}

// Gaussian smoothing with fusion
__global__ void smoothGaussianFused(
    float* __restrict__ values_ndvi,
    float* __restrict__ values_mndwi,
    float* __restrict__ values_evi,
    float* __restrict__ values_ndwi,
    const int* __restrict__ neighbor_indices,
    const int* __restrict__ neighbor_counts,
    float* __restrict__ smoothed_ndvi,
    float* __restrict__ smoothed_mndwi,
    float* __restrict__ smoothed_evi,
    float* __restrict__ smoothed_ndwi,
    int n_hexagons,
    float weight_center,
    float weight_neighbor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_hexagons) return;
    
    float4 center_vals = make_float4(
        values_ndvi[idx],
        values_mndwi[idx],
        values_evi[idx],
        values_ndwi[idx]
    );
    
    float4 weighted_sums = make_float4(
        center_vals.x * weight_center,
        center_vals.y * weight_center,
        center_vals.z * weight_center,
        center_vals.w * weight_center
    );
    float weight_sum = weight_center;
    
    int n_neighbors = neighbor_counts[idx];
    
    #pragma unroll
    for (int i = 0; i < MAX_NEIGHBORS; i++) {
        if (i < n_neighbors) {
            int neighbor_idx = neighbor_indices[idx + i * n_hexagons];
            
            if (neighbor_idx >= 0 && neighbor_idx < n_hexagons) {
                weighted_sums.x += values_ndvi[neighbor_idx] * weight_neighbor;
                weighted_sums.y += values_mndwi[neighbor_idx] * weight_neighbor;
                weighted_sums.z += values_evi[neighbor_idx] * weight_neighbor;
                weighted_sums.w += values_ndwi[neighbor_idx] * weight_neighbor;
                weight_sum += weight_neighbor;
            }
        }
    }
    
    float inv_weight = 1.0f / weight_sum;
    smoothed_ndvi[idx] = weighted_sums.x * inv_weight;
    smoothed_mndwi[idx] = weighted_sums.y * inv_weight;
    smoothed_evi[idx] = weighted_sums.z * inv_weight;
    smoothed_ndwi[idx] = weighted_sums.w * inv_weight;
}

class HexGridCUDAv4 {
private:
    HexagonGPUv4 d_data;
    std::vector<std::vector<float>> h_values;  // [variable][hexagon]
    std::vector<int> h_neighbor_indices_padded;
    std::vector<int> h_neighbor_counts;
    std::vector<std::vector<float>> h_smoothed_values;
    std::vector<int> h_reorder_map;
    std::vector<int> h_inverse_reorder_map;
    
    bool use_shared_memory = false;
    bool use_spatial_reordering = true;
    
    // Create spatial reordering based on hex coordinates
    void createSpatialReordering(const std::vector<int>& hex_ids) {
        int n = hex_ids.size();
        h_reorder_map.resize(n);
        h_inverse_reorder_map.resize(n);
        
        // Extract coordinates and create Morton codes
        std::vector<std::pair<uint32_t, int>> morton_pairs;
        morton_pairs.reserve(n);
        
        for (int i = 0; i < n; i++) {
            // Extract x,y from hex_id (based on your encoding)
            int hex_x = hex_ids[i] / 10000;
            int hex_y = hex_ids[i] % 10000;
            
            // Normalize to 16-bit range
            uint16_t x = (uint16_t)(hex_x & 0xFFFF);
            uint16_t y = (uint16_t)(hex_y & 0xFFFF);
            
            uint32_t morton = morton2D(x, y);
            morton_pairs.push_back({morton, i});
        }
        
        // Sort by Morton code
        std::sort(morton_pairs.begin(), morton_pairs.end());
        
        // Create mapping
        for (int i = 0; i < n; i++) {
            int original_idx = morton_pairs[i].second;
            h_reorder_map[original_idx] = i;      // old -> new
            h_inverse_reorder_map[i] = original_idx; // new -> old
        }
        
        std::cout << "Created spatial reordering using Morton codes\n";
    }

public:
    HexGridCUDAv4() {
        d_data = {
            {nullptr, nullptr, nullptr, nullptr},
            nullptr,
            {nullptr, nullptr, nullptr, nullptr},
            nullptr, nullptr, nullptr, 0, 0
        };
        h_values.resize(NUM_VARIABLES);
        h_smoothed_values.resize(NUM_VARIABLES);
    }
    
    ~HexGridCUDAv4() {
        for (int i = 0; i < NUM_VARIABLES; i++) {
            if (d_data.values[i]) cudaFree(d_data.values[i]);
            if (d_data.smoothed_values[i]) cudaFree(d_data.smoothed_values[i]);
        }
        if (d_data.neighbor_indices) cudaFree(d_data.neighbor_indices);
        if (d_data.neighbor_counts) cudaFree(d_data.neighbor_counts);
        if (d_data.reorder_map) cudaFree(d_data.reorder_map);
        if (d_data.inverse_reorder_map) cudaFree(d_data.inverse_reorder_map);
    }

    void setUseSharedMemory(bool use_shared) {
        use_shared_memory = use_shared;
    }

    void setUseSpatialReordering(bool use_reorder) {
        use_spatial_reordering = use_reorder;
    }

    void loadFromBinary(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        int n_hexagons;
        file.read(reinterpret_cast<char*>(&n_hexagons), sizeof(int));
        d_data.n_hexagons = n_hexagons;
        d_data.padded_neighbors_size = n_hexagons * MAX_NEIGHBORS;
        
        // Initialize host arrays
        for (int i = 0; i < NUM_VARIABLES; i++) {
            h_values[i].resize(n_hexagons);
            h_smoothed_values[i].resize(n_hexagons);
        }
        h_neighbor_counts.resize(n_hexagons);
        h_neighbor_indices_padded.resize(d_data.padded_neighbors_size, -1);
        
        std::vector<int> hex_ids;
        hex_ids.reserve(n_hexagons);
        
        // Read data - for now just replicate NDVI for other variables
        std::vector<std::vector<int>> original_neighbors(n_hexagons);
        
        for (int i = 0; i < n_hexagons; i++) {
            float ndvi;
            file.read(reinterpret_cast<char*>(&ndvi), sizeof(float));
            
            // Simulate multiple variables (in real use, load actual data)
            h_values[0][i] = ndvi;                          // NDVI
            h_values[1][i] = ndvi * 0.9f + 0.05f;          // MNDWI (simulated)
            h_values[2][i] = ndvi * 1.1f - 0.05f;          // EVI (simulated)
            h_values[3][i] = ndvi * 0.95f;                 // NDWI (simulated)
            
            int n_neighbors;
            file.read(reinterpret_cast<char*>(&n_neighbors), sizeof(int));
            h_neighbor_counts[i] = n_neighbors;
            
            original_neighbors[i].resize(n_neighbors);
            for (int j = 0; j < n_neighbors; j++) {
                file.read(reinterpret_cast<char*>(&original_neighbors[i][j]), sizeof(int));
            }
            
            // Assume hex_id can be derived from index for Morton coding
            hex_ids.push_back(325000000 + i);  // Placeholder
        }
        
        file.close();
        
        // Create spatial reordering if enabled
        if (use_spatial_reordering) {
            createSpatialReordering(hex_ids);
            
            // Reorder data
            std::vector<std::vector<float>> reordered_values(NUM_VARIABLES);
            std::vector<int> reordered_counts(n_hexagons);
            
            for (int v = 0; v < NUM_VARIABLES; v++) {
                reordered_values[v].resize(n_hexagons);
                for (int i = 0; i < n_hexagons; i++) {
                    int new_idx = h_reorder_map[i];
                    reordered_values[v][new_idx] = h_values[v][i];
                }
                h_values[v] = std::move(reordered_values[v]);
            }
            
            // Reorder neighbor information
            for (int i = 0; i < n_hexagons; i++) {
                int new_idx = h_reorder_map[i];
                reordered_counts[new_idx] = h_neighbor_counts[i];
                
                // Remap neighbor indices
                for (int j = 0; j < original_neighbors[i].size(); j++) {
                    int neighbor_old = original_neighbors[i][j];
                    int neighbor_new = (neighbor_old >= 0) ? h_reorder_map[neighbor_old] : -1;
                    h_neighbor_indices_padded[new_idx + j * n_hexagons] = neighbor_new;
                }
            }
            h_neighbor_counts = std::move(reordered_counts);
        } else {
            // No reordering - use original layout
            for (int i = 0; i < n_hexagons; i++) {
                for (int j = 0; j < original_neighbors[i].size(); j++) {
                    h_neighbor_indices_padded[i + j * n_hexagons] = original_neighbors[i][j];
                }
            }
        }
        
        // Allocate device memory
        for (int i = 0; i < NUM_VARIABLES; i++) {
            CUDA_CHECK(cudaMalloc(&d_data.values[i], n_hexagons * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_data.smoothed_values[i], n_hexagons * sizeof(float)));
        }
        CUDA_CHECK(cudaMalloc(&d_data.neighbor_indices, d_data.padded_neighbors_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_data.neighbor_counts, n_hexagons * sizeof(int)));
        
        if (use_spatial_reordering) {
            CUDA_CHECK(cudaMalloc(&d_data.reorder_map, n_hexagons * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_data.inverse_reorder_map, n_hexagons * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_data.reorder_map, h_reorder_map.data(),
                                  n_hexagons * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_data.inverse_reorder_map, h_inverse_reorder_map.data(),
                                  n_hexagons * sizeof(int), cudaMemcpyHostToDevice));
        }
        
        // Copy to device
        for (int i = 0; i < NUM_VARIABLES; i++) {
            CUDA_CHECK(cudaMemcpy(d_data.values[i], h_values[i].data(),
                                  n_hexagons * sizeof(float), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(d_data.neighbor_indices, h_neighbor_indices_padded.data(),
                              d_data.padded_neighbors_size * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_data.neighbor_counts, h_neighbor_counts.data(),
                              n_hexagons * sizeof(int), cudaMemcpyHostToDevice));
        
        std::cout << "Loaded " << n_hexagons << " hexagons with " << NUM_VARIABLES << " variables\n";
    }

    void smoothSimpleAverage() {
        const int gridSize = (d_data.n_hexagons + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        if (use_shared_memory) {
            size_t shared_mem_size = BLOCK_SIZE * NUM_VARIABLES * sizeof(float);
            smoothMultiVariableFusedShared<<<gridSize, BLOCK_SIZE, shared_mem_size>>>(
                d_data.values[0], d_data.values[1], d_data.values[2], d_data.values[3],
                d_data.neighbor_indices,
                d_data.neighbor_counts,
                d_data.smoothed_values[0], d_data.smoothed_values[1],
                d_data.smoothed_values[2], d_data.smoothed_values[3],
                d_data.n_hexagons
            );
        } else {
            smoothMultiVariableFused<<<gridSize, BLOCK_SIZE>>>(
                d_data.values[0], d_data.values[1], d_data.values[2], d_data.values[3],
                d_data.neighbor_indices,
                d_data.neighbor_counts,
                d_data.smoothed_values[0], d_data.smoothed_values[1],
                d_data.smoothed_values[2], d_data.smoothed_values[3],
                d_data.n_hexagons
            );
        }
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void smoothGaussian(float sigma) {
        const int gridSize = (d_data.n_hexagons + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        float weight_center = 1.0f;
        float weight_neighbor = expf(-1.0f / (2.0f * sigma * sigma));
        
        smoothGaussianFused<<<gridSize, BLOCK_SIZE>>>(
            d_data.values[0], d_data.values[1], d_data.values[2], d_data.values[3],
            d_data.neighbor_indices,
            d_data.neighbor_counts,
            d_data.smoothed_values[0], d_data.smoothed_values[1],
            d_data.smoothed_values[2], d_data.smoothed_values[3],
            d_data.n_hexagons,
            weight_center,
            weight_neighbor
        );
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    double benchmarkSimple(int iterations) {
        smoothSimpleAverage();
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        
        for (int i = 0; i < iterations; i++) {
            smoothSimpleAverage();
        }
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        
        return (milliseconds * 1000.0) / iterations;
    }

    void saveResults(const std::string& filename) {
        // Copy back and reorder if necessary
        for (int v = 0; v < NUM_VARIABLES; v++) {
            CUDA_CHECK(cudaMemcpy(h_smoothed_values[v].data(), d_data.smoothed_values[v],
                                  d_data.n_hexagons * sizeof(float), cudaMemcpyDeviceToHost));
        }
        
        // Save first variable (NDVI) for comparison
        std::ofstream file(filename);
        for (int i = 0; i < d_data.n_hexagons; i++) {
            int output_idx = use_spatial_reordering ? h_inverse_reorder_map[i] : i;
            file << h_smoothed_values[0][output_idx] << "\n";
        }
        file.close();
    }

    void printStats() {
        std::cout << "Processing " << NUM_VARIABLES << " variables simultaneously\n";
        std::cout << "Spatial reordering: " << (use_spatial_reordering ? "enabled" : "disabled") << "\n";
        std::cout << "Shared memory: " << (use_shared_memory ? "enabled" : "disabled") << "\n";
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data.bin> [--shared] [--no-reorder]\n";
        return 1;
    }

    bool use_shared = false;
    bool use_reorder = true;
    
    for (int i = 2; i < argc; i++) {
        if (std::string(argv[i]) == "--shared") use_shared = true;
        if (std::string(argv[i]) == "--no-reorder") use_reorder = false;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << "\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n\n";

    HexGridCUDAv4 grid;
    grid.setUseSharedMemory(use_shared);
    grid.setUseSpatialReordering(use_reorder);
    grid.loadFromBinary(argv[1]);
    grid.printStats();

    const int iterations = 100;
    
    std::cout << "\n=== CUDA v4: Kernel Fusion + Spatial Reordering ===\n";
    
    double time_simple = grid.benchmarkSimple(iterations);
    std::cout << "Simple Average (4 variables): " << time_simple << " μs\n";
    std::cout << "Per-variable time: " << time_simple / NUM_VARIABLES << " μs\n";
    
    grid.smoothSimpleAverage();
    grid.saveResults("smoothed_cuda_v4.csv");
    
    return 0;
}