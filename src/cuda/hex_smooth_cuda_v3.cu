// hex_smooth_cuda_v3.cu - Texture memory and warp-level optimizations
#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

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
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;

// Data structure
struct HexagonGPUv3 {
    float* ndvi_values;
    int* neighbor_indices;
    float* smoothed_values;
    int* neighbor_counts;
    int n_hexagons;
    int padded_neighbors_size;
    cudaTextureObject_t tex_obj;
};

// Kernel using texture memory
__global__ void smoothSimpleAverageTexture(
    const int* __restrict__ neighbor_indices,
    const int* __restrict__ neighbor_counts,
    float* __restrict__ smoothed_values,
    int n_hexagons,
    cudaTextureObject_t tex_ndvi)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_hexagons) return;
    
    // Read center value from texture
    float sum = tex1Dfetch<float>(tex_ndvi, idx);
    int count = 1;
    
    int n_neighbors = neighbor_counts[idx];
    
    #pragma unroll
    for (int i = 0; i < MAX_NEIGHBORS; i++) {
        if (i < n_neighbors) {
            int neighbor_idx = neighbor_indices[idx + i * n_hexagons];
            if (neighbor_idx >= 0 && neighbor_idx < n_hexagons) {
                float neighbor_val = tex1Dfetch<float>(tex_ndvi, neighbor_idx);
                sum += neighbor_val;
                count++;
            }
        }
    }
    
    smoothed_values[idx] = sum / count;
}

// Warp shuffle kernel
__global__ void smoothSimpleAverageWarpShuffle(
    const float* __restrict__ ndvi_values,
    const int* __restrict__ neighbor_indices,
    const int* __restrict__ neighbor_counts,
    float* __restrict__ smoothed_values,
    int n_hexagons)
{
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_hexagons) return;
    
    float my_ndvi = ndvi_values[idx];
    float sum = my_ndvi;
    int count = 1;
    
    int n_neighbors = neighbor_counts[idx];
    
    #pragma unroll
    for (int i = 0; i < MAX_NEIGHBORS; i++) {
        if (i < n_neighbors) {
            int neighbor_idx = neighbor_indices[idx + i * n_hexagons];
            
            if (neighbor_idx >= 0 && neighbor_idx < n_hexagons) {
                int neighbor_warp_base = neighbor_idx & ~(WARP_SIZE - 1);
                int my_warp_base = idx & ~(WARP_SIZE - 1);
                
                if (neighbor_warp_base == my_warp_base) {
                    int source_lane = neighbor_idx % WARP_SIZE;
                    float neighbor_val = warp.shfl(my_ndvi, source_lane);
                    sum += neighbor_val;
                } else {
                    sum += ndvi_values[neighbor_idx];
                }
                count++;
            }
        }
    }
    
    smoothed_values[idx] = sum / count;
}

// Gaussian kernel with texture
__global__ void smoothGaussianTexture(
    const int* __restrict__ neighbor_indices,
    const int* __restrict__ neighbor_counts,
    float* __restrict__ smoothed_values,
    int n_hexagons,
    cudaTextureObject_t tex_ndvi,
    float weight_center,
    float weight_neighbor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_hexagons) return;
    
    float weighted_sum = tex1Dfetch<float>(tex_ndvi, idx) * weight_center;
    float weight_sum = weight_center;
    int n_neighbors = neighbor_counts[idx];
    
    #pragma unroll
    for (int i = 0; i < MAX_NEIGHBORS; i++) {
        if (i < n_neighbors) {
            int neighbor_idx = neighbor_indices[idx + i * n_hexagons];
            if (neighbor_idx >= 0 && neighbor_idx < n_hexagons) {
                float neighbor_val = tex1Dfetch<float>(tex_ndvi, neighbor_idx);
                weighted_sum += neighbor_val * weight_neighbor;
                weight_sum += weight_neighbor;
            }
        }
    }
    
    smoothed_values[idx] = weighted_sum / weight_sum;
}

class HexGridCUDAv3 {
private:
    HexagonGPUv3 d_data;
    std::vector<float> h_ndvi_values;
    std::vector<int> h_neighbor_indices_padded;
    std::vector<int> h_neighbor_counts;
    std::vector<float> h_smoothed_values;
    
    bool use_texture = true;
    
    void createTextureObject() {
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = d_data.ndvi_values;
        resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
        resDesc.res.linear.desc.x = 32;
        resDesc.res.linear.sizeInBytes = d_data.n_hexagons * sizeof(float);
        
        cudaTextureDesc texDesc = {};
        texDesc.readMode = cudaReadModeElementType;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.normalizedCoords = false;
        
        CUDA_CHECK(cudaCreateTextureObject(&d_data.tex_obj, &resDesc, &texDesc, nullptr));
    }

public:
    HexGridCUDAv3() : d_data{nullptr, nullptr, nullptr, nullptr, 0, 0, 0} {}
    
    ~HexGridCUDAv3() {
        if (d_data.tex_obj) cudaDestroyTextureObject(d_data.tex_obj);
        if (d_data.ndvi_values) cudaFree(d_data.ndvi_values);
        if (d_data.neighbor_indices) cudaFree(d_data.neighbor_indices);
        if (d_data.neighbor_counts) cudaFree(d_data.neighbor_counts);
        if (d_data.smoothed_values) cudaFree(d_data.smoothed_values);
    }

    void setOptimizationType(const std::string& type) {
        use_texture = (type != "shuffle");
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
        
        h_ndvi_values.resize(n_hexagons);
        h_neighbor_counts.resize(n_hexagons);
        h_smoothed_values.resize(n_hexagons);
        // Padded neighbor array for coalesced access
        h_neighbor_indices_padded.resize(d_data.padded_neighbors_size, -1);
        
        for (int i = 0; i < n_hexagons; i++) {
            file.read(reinterpret_cast<char*>(&h_ndvi_values[i]), sizeof(float));
            
            int n_neighbors;
            file.read(reinterpret_cast<char*>(&n_neighbors), sizeof(int));
            h_neighbor_counts[i] = n_neighbors;
            
            for (int j = 0; j < n_neighbors; j++) {
                int neighbor_idx;
                file.read(reinterpret_cast<char*>(&neighbor_idx), sizeof(int));
                h_neighbor_indices_padded[i + j * n_hexagons] = neighbor_idx;
            }
        }
        
        file.close();
        
        int total_neighbors = 0;
        for (int i = 0; i < n_hexagons; i++) {
            total_neighbors += h_neighbor_counts[i];
        }
        
        std::cout << "Loaded " << n_hexagons << " hexagons with " 
                  << total_neighbors << " total neighbor connections\n";
        
        CUDA_CHECK(cudaMalloc(&d_data.ndvi_values, n_hexagons * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_data.neighbor_indices, d_data.padded_neighbors_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_data.neighbor_counts, n_hexagons * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_data.smoothed_values, n_hexagons * sizeof(float)));
        
        CUDA_CHECK(cudaMemcpy(d_data.ndvi_values, h_ndvi_values.data(), 
                              n_hexagons * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_data.neighbor_indices, h_neighbor_indices_padded.data(), 
                              d_data.padded_neighbors_size * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_data.neighbor_counts, h_neighbor_counts.data(), 
                              n_hexagons * sizeof(int), cudaMemcpyHostToDevice));
        
        createTextureObject();
    }

    void smoothSimpleAverage() {
        const int gridSize = (d_data.n_hexagons + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        if (use_texture) {
            smoothSimpleAverageTexture<<<gridSize, BLOCK_SIZE>>>(
                d_data.neighbor_indices,
                d_data.neighbor_counts,
                d_data.smoothed_values,
                d_data.n_hexagons,
                d_data.tex_obj
            );
        } else {
            smoothSimpleAverageWarpShuffle<<<gridSize, BLOCK_SIZE>>>(
                d_data.ndvi_values,
                d_data.neighbor_indices,
                d_data.neighbor_counts,
                d_data.smoothed_values,
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
        
        smoothGaussianTexture<<<gridSize, BLOCK_SIZE>>>(
            d_data.neighbor_indices,
            d_data.neighbor_counts,
            d_data.smoothed_values,
            d_data.n_hexagons,
            d_data.tex_obj,
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

    double benchmarkGaussian(float sigma, int iterations) {
        smoothGaussian(sigma);
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        
        for (int i = 0; i < iterations; i++) {
            smoothGaussian(sigma);
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
        CUDA_CHECK(cudaMemcpy(h_smoothed_values.data(), d_data.smoothed_values,
                              d_data.n_hexagons * sizeof(float), cudaMemcpyDeviceToHost));
        
        std::ofstream file(filename);
        for (int i = 0; i < d_data.n_hexagons; i++) {
            file << h_smoothed_values[i] << "\n";
        }
        file.close();
    }

    void printStats() {
        int min_neighbors = 999, max_neighbors = 0;
        float avg_neighbors = 0;
        
        for (int i = 0; i < d_data.n_hexagons; i++) {
            int n = h_neighbor_counts[i];
            min_neighbors = std::min(min_neighbors, n);
            max_neighbors = std::max(max_neighbors, n);
            avg_neighbors += n;
        }
        avg_neighbors /= d_data.n_hexagons;
        
        std::cout << "Neighbor stats: min=" << min_neighbors 
                  << ", max=" << max_neighbors 
                  << ", avg=" << avg_neighbors << "\n";
    }
};

// Main function
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data.bin> [texture|shuffle]\n";
        return 1;
    }

    std::string opt_type = "texture";
    if (argc > 2) {
        opt_type = argv[2];
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << "\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Warp size: " << prop.warpSize << "\n\n";

    HexGridCUDAv3 grid;
    grid.setOptimizationType(opt_type);
    grid.loadFromBinary(argv[1]);
    grid.printStats();

    const int iterations = 100;
    
    std::cout << "\n=== CUDA v3: " << opt_type << " optimization ===\n";
    
    double time_simple = grid.benchmarkSimple(iterations);
    std::cout << "Simple Average: " << time_simple << " μs\n";
    
    double time_gaussian = grid.benchmarkGaussian(1.0f, iterations);
    std::cout << "Gaussian: " << time_gaussian << " μs\n";
    
    grid.smoothSimpleAverage();
    grid.saveResults("smoothed_cuda_v3.csv");
    
    return 0;
}