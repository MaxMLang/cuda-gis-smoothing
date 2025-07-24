// hex_smooth_cuda_v1.cu - Fixed version
#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <chrono>

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

// Device structure for hexagon data
struct HexagonGPU {
    float* ndvi_values;
    int* neighbor_indices;  // All neighbors in one big list
    int* neighbor_offsets;  // Where each hex's neighbors start
    int* neighbor_counts;   // How many neighbors each hex has
    float* smoothed_values;
    int n_hexagons;
};

// Naive kernel - one thread per hexagon
__global__ void smoothSimpleAverageNaive(
    const float* ndvi_values,
    const int* neighbor_indices,
    const int* neighbor_offsets,
    const int* neighbor_counts,
    float* smoothed_values,
    int n_hexagons)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_hexagons) return;
    
    // Get this hexagon's neighbors
    int start = neighbor_offsets[idx];
    int count = neighbor_counts[idx];
    
    float sum = ndvi_values[idx];
    int total_count = 1;
    
    // Loop through neighbors (slow, not coalesced)
    for (int i = 0; i < count; i++) {
        int neighbor_idx = neighbor_indices[start + i];
        if (neighbor_idx >= 0 && neighbor_idx < n_hexagons) {
            sum += ndvi_values[neighbor_idx];
            total_count++;
        }
    }
    
    smoothed_values[idx] = sum / total_count;
}

// Naive Gaussian kernel
__global__ void smoothGaussianNaive(
    const float* ndvi_values,
    const int* neighbor_indices,
    const int* neighbor_offsets,
    const int* neighbor_counts,
    float* smoothed_values,
    int n_hexagons,
    float weight_center,
    float weight_neighbor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_hexagons) return;
    
    int start = neighbor_offsets[idx];
    int count = neighbor_counts[idx];
    
    float weighted_sum = ndvi_values[idx] * weight_center;
    float weight_sum = weight_center;
    
    for (int i = 0; i < count; i++) {
        int neighbor_idx = neighbor_indices[start + i];
        if (neighbor_idx >= 0 && neighbor_idx < n_hexagons) {
            weighted_sum += ndvi_values[neighbor_idx] * weight_neighbor;
            weight_sum += weight_neighbor;
        }
    }
    
    smoothed_values[idx] = weighted_sum / weight_sum;
}

class HexGridCUDAv1 {
private:
    HexagonGPU d_data;
    std::vector<float> h_ndvi_values;
    std::vector<int> h_neighbor_indices;
    std::vector<int> h_neighbor_offsets;
    std::vector<int> h_neighbor_counts;
    std::vector<float> h_smoothed_values;

public:
    HexGridCUDAv1() : d_data{nullptr, nullptr, nullptr, nullptr, nullptr, 0} {}
    
    ~HexGridCUDAv1() {
        if (d_data.ndvi_values) cudaFree(d_data.ndvi_values);
        if (d_data.neighbor_indices) cudaFree(d_data.neighbor_indices);
        if (d_data.neighbor_offsets) cudaFree(d_data.neighbor_offsets);
        if (d_data.neighbor_counts) cudaFree(d_data.neighbor_counts);
        if (d_data.smoothed_values) cudaFree(d_data.smoothed_values);
    }

    void loadFromBinary(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        int n_hexagons;
        file.read(reinterpret_cast<char*>(&n_hexagons), sizeof(int));
        d_data.n_hexagons = n_hexagons;
        
        // Prepare host data
        h_ndvi_values.resize(n_hexagons);
        h_neighbor_offsets.resize(n_hexagons + 1);
        h_neighbor_counts.resize(n_hexagons);
        h_smoothed_values.resize(n_hexagons);
        
        int total_neighbors = 0;
        h_neighbor_offsets[0] = 0;
        
        // First pass: read data and count neighbors
        std::vector<std::vector<int>> temp_neighbors(n_hexagons);
        
        for (int i = 0; i < n_hexagons; i++) {
            file.read(reinterpret_cast<char*>(&h_ndvi_values[i]), sizeof(float));
            
            int n_neighbors;
            file.read(reinterpret_cast<char*>(&n_neighbors), sizeof(int));
            h_neighbor_counts[i] = n_neighbors;
            
            temp_neighbors[i].resize(n_neighbors);
            for (int j = 0; j < n_neighbors; j++) {
                file.read(reinterpret_cast<char*>(&temp_neighbors[i][j]), sizeof(int));
            }
            
            total_neighbors += n_neighbors;
            h_neighbor_offsets[i + 1] = total_neighbors;
        }
        
        // Flatten neighbor indices
        h_neighbor_indices.resize(total_neighbors);
        int idx = 0;
        for (int i = 0; i < n_hexagons; i++) {
            for (size_t j = 0; j < temp_neighbors[i].size(); j++) {
                h_neighbor_indices[idx++] = temp_neighbors[i][j];
            }
        }
        
        file.close();
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_data.ndvi_values, n_hexagons * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_data.neighbor_indices, total_neighbors * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_data.neighbor_offsets, (n_hexagons + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_data.neighbor_counts, n_hexagons * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_data.smoothed_values, n_hexagons * sizeof(float)));
        
        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_data.ndvi_values, h_ndvi_values.data(), 
                              n_hexagons * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_data.neighbor_indices, h_neighbor_indices.data(), 
                              total_neighbors * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_data.neighbor_offsets, h_neighbor_offsets.data(), 
                              (n_hexagons + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_data.neighbor_counts, h_neighbor_counts.data(), 
                              n_hexagons * sizeof(int), cudaMemcpyHostToDevice));
        
        std::cout << "Loaded " << n_hexagons << " hexagons with " 
                  << total_neighbors << " total neighbor connections\n";
    }

    void smoothSimpleAverage() {
        const int blockSize = 256;
        const int gridSize = (d_data.n_hexagons + blockSize - 1) / blockSize;
        
        smoothSimpleAverageNaive<<<gridSize, blockSize>>>(
            d_data.ndvi_values,
            d_data.neighbor_indices,
            d_data.neighbor_offsets,
            d_data.neighbor_counts,
            d_data.smoothed_values,
            d_data.n_hexagons
        );
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void smoothGaussian(float sigma) {
        const int blockSize = 256;
        const int gridSize = (d_data.n_hexagons + blockSize - 1) / blockSize;
        
        float weight_center = 1.0f;
        float weight_neighbor = expf(-1.0f / (2.0f * sigma * sigma));
        
        smoothGaussianNaive<<<gridSize, blockSize>>>(
            d_data.ndvi_values,
            d_data.neighbor_indices,
            d_data.neighbor_offsets,
            d_data.neighbor_counts,
            d_data.smoothed_values,
            d_data.n_hexagons,
            weight_center,
            weight_neighbor
        );
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Fixed: Separate benchmark functions for different signatures
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
        std::cerr << "Usage: " << argv[0] << " <data.bin>\n";
        return 1;
    }

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << "\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n\n";

    HexGridCUDAv1 grid;
    grid.loadFromBinary(argv[1]);
    grid.printStats();

    const int iterations = 10;
    
    std::cout << "\n=== CUDA v1: Naive Implementation ===\n";
    
    // Benchmark simple average
    double time_simple = grid.benchmarkSimple(iterations);
    std::cout << "Simple Average: " << time_simple << " μs\n";
    
    // Benchmark Gaussian
    double time_gaussian = grid.benchmarkGaussian(1.0f, iterations);
    std::cout << "Gaussian: " << time_gaussian << " μs\n";
    
    // Save results
    grid.smoothSimpleAverage();
    grid.saveResults("smoothed_cuda_v1.csv");
    
    return 0;
}