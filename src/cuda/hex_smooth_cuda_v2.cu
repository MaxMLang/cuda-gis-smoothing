// hex_smooth_cuda_v2.cu - Improved memory access patterns
#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>

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

// Constants for optimization
constexpr int WARP_SIZE = 32;
constexpr int MAX_NEIGHBORS = 6;  // Hexagons have at most 6 neighbors
constexpr int BLOCK_SIZE = 256;   // Multiple of warp size

// Improved data structure with padding for aligned access
struct HexagonGPUv2 {
    float* ndvi_values;
    int* neighbor_indices;      // Padded to MAX_NEIGHBORS per hexagon
    float* smoothed_values;
    int* neighbor_counts;       // Actual neighbor count per hexagon
    int n_hexagons;
    int padded_neighbors_size;  // n_hexagons * MAX_NEIGHBORS
};

// Coalesced memory access kernel - threads in warp access contiguous memory
__global__ void smoothSimpleAverageCoalesced(
    const float* __restrict__ ndvi_values,
    const int* __restrict__ neighbor_indices,
    const int* __restrict__ neighbor_counts,
    float* __restrict__ smoothed_values,
    int n_hexagons)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_hexagons) return;
    
    // Coalesced read of center value
    float sum = ndvi_values[idx];
    int count = 1;
    
    // Get actual neighbor count
    int n_neighbors = neighbor_counts[idx];
    
    // Process neighbors with better memory pattern
    // All threads in warp process their first neighbor together, then second, etc.
    #pragma unroll
    for (int i = 0; i < MAX_NEIGHBORS; i++) {
        if (i < n_neighbors) {
            // Coalesced access: thread idx accesses neighbor_indices[idx + i * n_hexagons]
            int neighbor_idx = neighbor_indices[idx + i * n_hexagons];
            
            if (neighbor_idx >= 0 && neighbor_idx < n_hexagons) {
                // This access is still somewhat scattered but better than v1
                sum += ndvi_values[neighbor_idx];
                count++;
            }
        }
    }
    
    // Coalesced write
    smoothed_values[idx] = sum / count;
}

// Shared memory version for better neighbor value reuse
__global__ void smoothSimpleAverageShared(
    const float* __restrict__ ndvi_values,
    const int* __restrict__ neighbor_indices,
    const int* __restrict__ neighbor_counts,
    float* __restrict__ smoothed_values,
    int n_hexagons)
{
    extern __shared__ float shared_ndvi[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Load a chunk of NDVI values into shared memory
    // Each thread loads one value
    if (idx < n_hexagons) {
        shared_ndvi[tid] = ndvi_values[idx];
    }
    __syncthreads();
    
    if (idx >= n_hexagons) return;
    
    float sum = shared_ndvi[tid];
    int count = 1;
    int n_neighbors = neighbor_counts[idx];
    
    #pragma unroll
    for (int i = 0; i < MAX_NEIGHBORS; i++) {
        if (i < n_neighbors) {
            int neighbor_idx = neighbor_indices[idx + i * n_hexagons];
            
            if (neighbor_idx >= 0 && neighbor_idx < n_hexagons) {
                // Check if neighbor is in shared memory
                int neighbor_block_idx = neighbor_idx - blockIdx.x * blockDim.x;
                if (neighbor_block_idx >= 0 && neighbor_block_idx < blockDim.x) {
                    // Use shared memory
                    sum += shared_ndvi[neighbor_block_idx];
                } else {
                    // Fall back to global memory
                    sum += ndvi_values[neighbor_idx];
                }
                count++;
            }
        }
    }
    
    smoothed_values[idx] = sum / count;
}

// Gaussian kernel with coalesced access
__global__ void smoothGaussianCoalesced(
    const float* __restrict__ ndvi_values,
    const int* __restrict__ neighbor_indices,
    const int* __restrict__ neighbor_counts,
    float* __restrict__ smoothed_values,
    int n_hexagons,
    float weight_center,
    float weight_neighbor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_hexagons) return;
    
    float weighted_sum = ndvi_values[idx] * weight_center;
    float weight_sum = weight_center;
    int n_neighbors = neighbor_counts[idx];
    
    #pragma unroll
    for (int i = 0; i < MAX_NEIGHBORS; i++) {
        if (i < n_neighbors) {
            int neighbor_idx = neighbor_indices[idx + i * n_hexagons];
            
            if (neighbor_idx >= 0 && neighbor_idx < n_hexagons) {
                weighted_sum += ndvi_values[neighbor_idx] * weight_neighbor;
                weight_sum += weight_neighbor;
            }
        }
    }
    
    smoothed_values[idx] = weighted_sum / weight_sum;
}

class HexGridCUDAv2 {
private:
    HexagonGPUv2 d_data;
    std::vector<float> h_ndvi_values;
    std::vector<int> h_neighbor_indices_padded;
    std::vector<int> h_neighbor_counts;
    std::vector<float> h_smoothed_values;
    
    bool use_shared_memory = false;

public:
    HexGridCUDAv2() : d_data{nullptr, nullptr, nullptr, nullptr, 0, 0} {}
    
    ~HexGridCUDAv2() {
        if (d_data.ndvi_values) cudaFree(d_data.ndvi_values);
        if (d_data.neighbor_indices) cudaFree(d_data.neighbor_indices);
        if (d_data.neighbor_counts) cudaFree(d_data.neighbor_counts);
        if (d_data.smoothed_values) cudaFree(d_data.smoothed_values);
    }

    void setUseSharedMemory(bool use_shared) {
        use_shared_memory = use_shared;
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
        
        // Prepare host data
        h_ndvi_values.resize(n_hexagons);
        h_neighbor_counts.resize(n_hexagons);
        h_smoothed_values.resize(n_hexagons);
        
        // Padded neighbor array for coalesced access
        // Layout: neighbor i of all hexagons are contiguous
        h_neighbor_indices_padded.resize(d_data.padded_neighbors_size, -1);
        
        // Read data and reorganize for better GPU access
        for (int i = 0; i < n_hexagons; i++) {
            file.read(reinterpret_cast<char*>(&h_ndvi_values[i]), sizeof(float));
            
            int n_neighbors;
            file.read(reinterpret_cast<char*>(&n_neighbors), sizeof(int));
            h_neighbor_counts[i] = n_neighbors;
            
            // Read neighbors and place in padded structure
            for (int j = 0; j < n_neighbors; j++) {
                int neighbor_idx;
                file.read(reinterpret_cast<char*>(&neighbor_idx), sizeof(int));
                // Store in column-major order for coalesced access
                h_neighbor_indices_padded[i + j * n_hexagons] = neighbor_idx;
            }
            
            // Skip remaining neighbor slots in file if any
            for (int j = n_neighbors; j < MAX_NEIGHBORS && j < n_neighbors; j++) {
                int dummy;
                file.read(reinterpret_cast<char*>(&dummy), sizeof(int));
            }
        }
        
        file.close();
        
        // Count statistics
        int total_neighbors = 0;
        for (int i = 0; i < n_hexagons; i++) {
            total_neighbors += h_neighbor_counts[i];
        }
        
        std::cout << "Loaded " << n_hexagons << " hexagons with " 
                  << total_neighbors << " total neighbor connections\n";
        std::cout << "Using padded layout with " << MAX_NEIGHBORS << " slots per hexagon\n";
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_data.ndvi_values, n_hexagons * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_data.neighbor_indices, d_data.padded_neighbors_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_data.neighbor_counts, n_hexagons * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_data.smoothed_values, n_hexagons * sizeof(float)));
        
        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_data.ndvi_values, h_ndvi_values.data(), 
                              n_hexagons * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_data.neighbor_indices, h_neighbor_indices_padded.data(), 
                              d_data.padded_neighbors_size * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_data.neighbor_counts, h_neighbor_counts.data(), 
                              n_hexagons * sizeof(int), cudaMemcpyHostToDevice));
    }

    void smoothSimpleAverage() {
        const int gridSize = (d_data.n_hexagons + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        if (use_shared_memory) {
            size_t shared_mem_size = BLOCK_SIZE * sizeof(float);
            smoothSimpleAverageShared<<<gridSize, BLOCK_SIZE, shared_mem_size>>>(
                d_data.ndvi_values,
                d_data.neighbor_indices,
                d_data.neighbor_counts,
                d_data.smoothed_values,
                d_data.n_hexagons
            );
        } else {
            smoothSimpleAverageCoalesced<<<gridSize, BLOCK_SIZE>>>(
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
        
        smoothGaussianCoalesced<<<gridSize, BLOCK_SIZE>>>(
            d_data.ndvi_values,
            d_data.neighbor_indices,
            d_data.neighbor_counts,
            d_data.smoothed_values,
            d_data.n_hexagons,
            weight_center,
            weight_neighbor
        );
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    double benchmarkSimple(int iterations) {
        // Warm-up
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
        
        return (milliseconds * 1000.0) / iterations; // Return microseconds
    }

    double benchmarkGaussian(float sigma, int iterations) {
        // Warm-up
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
        
        return (milliseconds * 1000.0) / iterations; // Return microseconds
    }

    void saveResults(const std::string& filename) {
        // Copy results back
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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data.bin> [--shared]\n";
        return 1;
    }

    bool use_shared = false;
    if (argc > 2 && std::string(argv[2]) == "--shared") {
        use_shared = true;
    }

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << "\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock << " bytes\n\n";

    HexGridCUDAv2 grid;
    grid.setUseSharedMemory(use_shared);
    grid.loadFromBinary(argv[1]);
    grid.printStats();

    const int iterations = 100;  // More iterations for better timing
    
    std::cout << "\n=== CUDA v2: Coalesced Memory Access ";
    if (use_shared) std::cout << "(with Shared Memory) ";
    std::cout << "===\n";
    
    // Benchmark simple average
    double time_simple = grid.benchmarkSimple(iterations);
    std::cout << "Simple Average: " << time_simple << " μs\n";
    
    // Benchmark Gaussian
    double time_gaussian = grid.benchmarkGaussian(1.0f, iterations);
    std::cout << "Gaussian: " << time_gaussian << " μs\n";
    
    // Save results
    grid.smoothSimpleAverage();
    grid.saveResults("smoothed_cuda_v2.csv");
    
    return 0;
}