// hex_smooth_cuda_v5.cu - Extended neighborhood smoothing
#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <set> // <--- FIX: Added the required header for std::set

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
constexpr int MAX_NEIGHBORS_1ST = 6;   // First-order neighbors
constexpr int MAX_NEIGHBORS_2ND = 12;  // Second-order neighbors  
constexpr int MAX_NEIGHBORS_TOTAL = 18; // Total possible neighbors
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_VARIABLES = 4;

// Gaussian weights based on hex distance
constexpr float WEIGHT_CENTER = 1.0f;
constexpr float WEIGHT_FIRST = 0.6065f;   // exp(-1.0/(2*1*1)) for distance 1
constexpr float WEIGHT_SECOND = 0.1353f;  // exp(-3.0/(2*1*1)) for distance sqrt(3)

// Data structure
struct HexagonGPUv5 {
    float* values[NUM_VARIABLES];
    int* neighbor_indices_1st;     // First-order neighbors (coalesced layout)
    int* neighbor_indices_2nd;     // Second-order neighbors (coalesced layout)
    int* neighbor_counts_1st;      // Number of 1st order neighbors per hex
    int* neighbor_counts_2nd;      // Number of 2nd order neighbors per hex
    float* smoothed_values[NUM_VARIABLES];
    int n_hexagons;
};

// Single variable kernel - first order only
__global__ void smoothFirstOrderOnly(
    const float* __restrict__ values,
    const int* __restrict__ neighbor_indices_1st,
    const int* __restrict__ neighbor_counts_1st,
    float* __restrict__ smoothed_values,
    int n_hexagons,
    bool use_gaussian_weights)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_hexagons) return;
    
    float sum = values[idx];
    float weight_sum = 1.0f;
    int n_neighbors = neighbor_counts_1st[idx];
    
    #pragma unroll
    for (int i = 0; i < MAX_NEIGHBORS_1ST; i++) {
        if (i < n_neighbors) {
            int neighbor_idx = neighbor_indices_1st[idx + i * n_hexagons];
            if (neighbor_idx >= 0 && neighbor_idx < n_hexagons) {
                if (use_gaussian_weights) {
                    sum += values[neighbor_idx] * WEIGHT_FIRST;
                    weight_sum += WEIGHT_FIRST;
                } else {
                    sum += values[neighbor_idx];
                    weight_sum += 1.0f;
                }
            }
        }
    }
    
    smoothed_values[idx] = sum / weight_sum;
}

// Single variable kernel - first and second order
__global__ void smoothBothOrders(
    const float* __restrict__ values,
    const int* __restrict__ neighbor_indices_1st,
    const int* __restrict__ neighbor_indices_2nd,
    const int* __restrict__ neighbor_counts_1st,
    const int* __restrict__ neighbor_counts_2nd,
    float* __restrict__ smoothed_values,
    int n_hexagons,
    bool use_gaussian_weights)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_hexagons) return;
    
    float sum = values[idx];
    float weight_sum = 1.0f;
    
    // First-order neighbors
    int n_neighbors_1st = neighbor_counts_1st[idx];
    #pragma unroll
    for (int i = 0; i < MAX_NEIGHBORS_1ST; i++) {
        if (i < n_neighbors_1st) {
            int neighbor_idx = neighbor_indices_1st[idx + i * n_hexagons];
            if (neighbor_idx >= 0 && neighbor_idx < n_hexagons) {
                if (use_gaussian_weights) {
                    sum += values[neighbor_idx] * WEIGHT_FIRST;
                    weight_sum += WEIGHT_FIRST;
                } else {
                    sum += values[neighbor_idx];
                    weight_sum += 1.0f;
                }
            }
        }
    }
    
    // Second-order neighbors
    int n_neighbors_2nd = neighbor_counts_2nd[idx];
    #pragma unroll
    for (int i = 0; i < MAX_NEIGHBORS_2ND; i++) {
        if (i < n_neighbors_2nd) {
            int neighbor_idx = neighbor_indices_2nd[idx + i * n_hexagons];
            if (neighbor_idx >= 0 && neighbor_idx < n_hexagons) {
                if (use_gaussian_weights) {
                    sum += values[neighbor_idx] * WEIGHT_SECOND;
                    weight_sum += WEIGHT_SECOND;
                } else {
                    sum += values[neighbor_idx];
                    weight_sum += 1.0f;
                }
            }
        }
    }
    
    smoothed_values[idx] = sum / weight_sum;
}

// Multi-variable kernel - first order only with fusion
__global__ void smoothFirstOrderFused(
    float* __restrict__ values_ndvi,
    float* __restrict__ values_mndwi,
    float* __restrict__ values_evi,
    float* __restrict__ values_ndwi,
    const int* __restrict__ neighbor_indices_1st,
    const int* __restrict__ neighbor_counts_1st,
    float* __restrict__ smoothed_ndvi,
    float* __restrict__ smoothed_mndwi,
    float* __restrict__ smoothed_evi,
    float* __restrict__ smoothed_ndwi,
    int n_hexagons,
    bool use_gaussian_weights)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_hexagons) return;
    
    float4 sums = make_float4(
        values_ndvi[idx],
        values_mndwi[idx],
        values_evi[idx],
        values_ndwi[idx]
    );
    float weight_sum = 1.0f;
    
    int n_neighbors = neighbor_counts_1st[idx];
    #pragma unroll
    for (int i = 0; i < MAX_NEIGHBORS_1ST; i++) {
        if (i < n_neighbors) {
            int neighbor_idx = neighbor_indices_1st[idx + i * n_hexagons];
            if (neighbor_idx >= 0 && neighbor_idx < n_hexagons) {
                float weight = use_gaussian_weights ? WEIGHT_FIRST : 1.0f;
                sums.x += values_ndvi[neighbor_idx] * weight;
                sums.y += values_mndwi[neighbor_idx] * weight;
                sums.z += values_evi[neighbor_idx] * weight;
                sums.w += values_ndwi[neighbor_idx] * weight;
                weight_sum += weight;
            }
        }
    }
    
    float inv_weight = 1.0f / weight_sum;
    smoothed_ndvi[idx] = sums.x * inv_weight;
    smoothed_mndwi[idx] = sums.y * inv_weight;
    smoothed_evi[idx] = sums.z * inv_weight;
    smoothed_ndwi[idx] = sums.w * inv_weight;
}

// Multi-variable kernel - both orders with fusion
__global__ void smoothBothOrdersFused(
    float* __restrict__ values_ndvi,
    float* __restrict__ values_mndwi,
    float* __restrict__ values_evi,
    float* __restrict__ values_ndwi,
    const int* __restrict__ neighbor_indices_1st,
    const int* __restrict__ neighbor_indices_2nd,
    const int* __restrict__ neighbor_counts_1st,
    const int* __restrict__ neighbor_counts_2nd,
    float* __restrict__ smoothed_ndvi,
    float* __restrict__ smoothed_mndwi,
    float* __restrict__ smoothed_evi,
    float* __restrict__ smoothed_ndwi,
    int n_hexagons,
    bool use_gaussian_weights)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_hexagons) return;
    
    float4 sums = make_float4(
        values_ndvi[idx],
        values_mndwi[idx],
        values_evi[idx],
        values_ndwi[idx]
    );
    float weight_sum = 1.0f;
    
    // First-order neighbors
    int n_neighbors_1st = neighbor_counts_1st[idx];
    #pragma unroll
    for (int i = 0; i < MAX_NEIGHBORS_1ST; i++) {
        if (i < n_neighbors_1st) {
            int neighbor_idx = neighbor_indices_1st[idx + i * n_hexagons];
            if (neighbor_idx >= 0 && neighbor_idx < n_hexagons) {
                float weight = use_gaussian_weights ? WEIGHT_FIRST : 1.0f;
                sums.x += values_ndvi[neighbor_idx] * weight;
                sums.y += values_mndwi[neighbor_idx] * weight;
                sums.z += values_evi[neighbor_idx] * weight;
                sums.w += values_ndwi[neighbor_idx] * weight;
                weight_sum += weight;
            }
        }
    }
    
    // Second-order neighbors
    int n_neighbors_2nd = neighbor_counts_2nd[idx];
    #pragma unroll
    for (int i = 0; i < MAX_NEIGHBORS_2ND; i++) {
        if (i < n_neighbors_2nd) {
            int neighbor_idx = neighbor_indices_2nd[idx + i * n_hexagons];
            if (neighbor_idx >= 0 && neighbor_idx < n_hexagons) {
                float weight = use_gaussian_weights ? WEIGHT_SECOND : 1.0f;
                sums.x += values_ndvi[neighbor_idx] * weight;
                sums.y += values_mndwi[neighbor_idx] * weight;
                sums.z += values_evi[neighbor_idx] * weight;
                sums.w += values_ndwi[neighbor_idx] * weight;
                weight_sum += weight;
            }
        }
    }
    
    float inv_weight = 1.0f / weight_sum;
    smoothed_ndvi[idx] = sums.x * inv_weight;
    smoothed_mndwi[idx] = sums.y * inv_weight;
    smoothed_evi[idx] = sums.z * inv_weight;
    smoothed_ndwi[idx] = sums.w * inv_weight;
}

class HexGridCUDAv5 {
private:
    HexagonGPUv5 d_data;
    std::vector<std::vector<float>> h_values;
    std::vector<int> h_neighbor_indices_1st_padded;
    std::vector<int> h_neighbor_indices_2nd_padded;
    std::vector<int> h_neighbor_counts_1st;
    std::vector<int> h_neighbor_counts_2nd;
    std::vector<std::vector<float>> h_smoothed_values;
    
    bool use_second_order = false;
    bool use_gaussian_weights = true;
    bool use_kernel_fusion = true;
    
    // Generate second-order neighbors from first-order
    void generateSecondOrderNeighbors(const std::vector<std::vector<int>>& first_order_neighbors) {
        int n = first_order_neighbors.size();
        h_neighbor_counts_2nd.resize(n);
        h_neighbor_indices_2nd_padded.resize(n * MAX_NEIGHBORS_2ND, -1);
        
        for (int i = 0; i < n; i++) {
            std::set<int> second_order;
            
            // For each first-order neighbor
            for (int n1 : first_order_neighbors[i]) {
                if (n1 >= 0 && n1 < n) {
                    // Add its neighbors (excluding self and first-order)
                    for (int n2 : first_order_neighbors[n1]) {
                        if (n2 >= 0 && n2 < n && n2 != i) {
                            // Check if n2 is not already a first-order neighbor
                            if (std::find(first_order_neighbors[i].begin(), 
                                        first_order_neighbors[i].end(), n2) == 
                                        first_order_neighbors[i].end()) {
                                second_order.insert(n2);
                            }
                        }
                    }
                }
            }
            
            h_neighbor_counts_2nd[i] = second_order.size();
            int j = 0;
            for (int neighbor : second_order) {
                if (j < MAX_NEIGHBORS_2ND) {
                    h_neighbor_indices_2nd_padded[i + j * n] = neighbor;
                    j++;
                }
            }
        }
        
        // Calculate statistics
        int total_2nd = 0;
        for (int c : h_neighbor_counts_2nd) total_2nd += c;
        std::cout << "Generated second-order neighbors: avg " 
                  << (float)total_2nd / n << " per hexagon\n";
    }

public:
    HexGridCUDAv5() {
        memset(&d_data, 0, sizeof(d_data));
        h_values.resize(NUM_VARIABLES);
        h_smoothed_values.resize(NUM_VARIABLES);
    }
    
    ~HexGridCUDAv5() {
        for (int i = 0; i < NUM_VARIABLES; i++) {
            if (d_data.values[i]) cudaFree(d_data.values[i]);
            if (d_data.smoothed_values[i]) cudaFree(d_data.smoothed_values[i]);
        }
        if (d_data.neighbor_indices_1st) cudaFree(d_data.neighbor_indices_1st);
        if (d_data.neighbor_indices_2nd) cudaFree(d_data.neighbor_indices_2nd);
        if (d_data.neighbor_counts_1st) cudaFree(d_data.neighbor_counts_1st);
        if (d_data.neighbor_counts_2nd) cudaFree(d_data.neighbor_counts_2nd);
    }

    void setOptions(bool second_order, bool gaussian, bool fusion) {
        use_second_order = second_order;
        use_gaussian_weights = gaussian;
        use_kernel_fusion = fusion;
    }

    void loadFromBinary(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        int n_hexagons;
        file.read(reinterpret_cast<char*>(&n_hexagons), sizeof(int));
        d_data.n_hexagons = n_hexagons;
        
        // Initialize arrays
        for (int i = 0; i < NUM_VARIABLES; i++) {
            h_values[i].resize(n_hexagons);
            h_smoothed_values[i].resize(n_hexagons);
        }
        h_neighbor_counts_1st.resize(n_hexagons);
        h_neighbor_indices_1st_padded.resize(n_hexagons * MAX_NEIGHBORS_1ST, -1);
        
        std::vector<std::vector<int>> first_order_neighbors(n_hexagons);
        
        // Read data
        for (int i = 0; i < n_hexagons; i++) {
            float ndvi;
            file.read(reinterpret_cast<char*>(&ndvi), sizeof(float));
            
            // Simulate multiple variables
            h_values[0][i] = ndvi;
            h_values[1][i] = ndvi * 0.9f + 0.05f;
            h_values[2][i] = ndvi * 1.1f - 0.05f;
            h_values[3][i] = ndvi * 0.95f;
            
            int n_neighbors;
            file.read(reinterpret_cast<char*>(&n_neighbors), sizeof(int));
            h_neighbor_counts_1st[i] = n_neighbors;
            
            first_order_neighbors[i].resize(n_neighbors);
            for (int j = 0; j < n_neighbors; j++) {
                file.read(reinterpret_cast<char*>(&first_order_neighbors[i][j]), sizeof(int));
                if (j < MAX_NEIGHBORS_1ST) {
                    h_neighbor_indices_1st_padded[i + j * n_hexagons] = first_order_neighbors[i][j];
                }
            }
        }
        file.close();
        
        // Generate second-order neighbors if needed
        if (use_second_order) {
            generateSecondOrderNeighbors(first_order_neighbors);
        }
        
        // Allocate device memory
        for (int i = 0; i < NUM_VARIABLES; i++) {
            CUDA_CHECK(cudaMalloc(&d_data.values[i], n_hexagons * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_data.smoothed_values[i], n_hexagons * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_data.values[i], h_values[i].data(),
                                  n_hexagons * sizeof(float), cudaMemcpyHostToDevice));
        }
        
        CUDA_CHECK(cudaMalloc(&d_data.neighbor_indices_1st, n_hexagons * MAX_NEIGHBORS_1ST * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_data.neighbor_counts_1st, n_hexagons * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_data.neighbor_indices_1st, h_neighbor_indices_1st_padded.data(),
                              n_hexagons * MAX_NEIGHBORS_1ST * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_data.neighbor_counts_1st, h_neighbor_counts_1st.data(),
                              n_hexagons * sizeof(int), cudaMemcpyHostToDevice));
        
        if (use_second_order) {
            CUDA_CHECK(cudaMalloc(&d_data.neighbor_indices_2nd, n_hexagons * MAX_NEIGHBORS_2ND * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_data.neighbor_counts_2nd, n_hexagons * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_data.neighbor_indices_2nd, h_neighbor_indices_2nd_padded.data(),
                                  n_hexagons * MAX_NEIGHBORS_2ND * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_data.neighbor_counts_2nd, h_neighbor_counts_2nd.data(),
                                  n_hexagons * sizeof(int), cudaMemcpyHostToDevice));
        }
        
        std::cout << "Loaded " << n_hexagons << " hexagons\n";
    }

    void smooth() {
        const int gridSize = (d_data.n_hexagons + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        if (use_kernel_fusion) {
            // Multi-variable processing
            if (use_second_order) {
                smoothBothOrdersFused<<<gridSize, BLOCK_SIZE>>>(
                    d_data.values[0], d_data.values[1], d_data.values[2], d_data.values[3],
                    d_data.neighbor_indices_1st, d_data.neighbor_indices_2nd,
                    d_data.neighbor_counts_1st, d_data.neighbor_counts_2nd,
                    d_data.smoothed_values[0], d_data.smoothed_values[1],
                    d_data.smoothed_values[2], d_data.smoothed_values[3],
                    d_data.n_hexagons, use_gaussian_weights
                );
            } else {
                smoothFirstOrderFused<<<gridSize, BLOCK_SIZE>>>(
                    d_data.values[0], d_data.values[1], d_data.values[2], d_data.values[3],
                    d_data.neighbor_indices_1st, d_data.neighbor_counts_1st,
                    d_data.smoothed_values[0], d_data.smoothed_values[1],
                    d_data.smoothed_values[2], d_data.smoothed_values[3],
                    d_data.n_hexagons, use_gaussian_weights
                );
            }
        } else {
            // Single variable processing (just NDVI)
            if (use_second_order) {
                smoothBothOrders<<<gridSize, BLOCK_SIZE>>>(
                    d_data.values[0], d_data.neighbor_indices_1st, d_data.neighbor_indices_2nd,
                    d_data.neighbor_counts_1st, d_data.neighbor_counts_2nd,
                    d_data.smoothed_values[0], d_data.n_hexagons, use_gaussian_weights
                );
            } else {
                smoothFirstOrderOnly<<<gridSize, BLOCK_SIZE>>>(
                    d_data.values[0], d_data.neighbor_indices_1st, d_data.neighbor_counts_1st,
                    d_data.smoothed_values[0], d_data.n_hexagons, use_gaussian_weights
                );
            }
        }
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    double benchmark(int iterations) {
        // Warm-up
        smooth();
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        
        for (int i = 0; i < iterations; i++) {
            smooth();
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
        CUDA_CHECK(cudaMemcpy(h_smoothed_values[0].data(), d_data.smoothed_values[0],
                              d_data.n_hexagons * sizeof(float), cudaMemcpyDeviceToHost));
        
        std::ofstream file(filename);
        for (int i = 0; i < d_data.n_hexagons; i++) {
            file << h_smoothed_values[0][i] << "\n";
        }
        file.close();
    }

    void printConfiguration() {
        std::cout << "Configuration:\n";
        std::cout << "  Neighbor orders: " << (use_second_order ? "1st + 2nd" : "1st only") << "\n";
        std::cout << "  Weights: " << (use_gaussian_weights ? "Gaussian" : "Uniform") << "\n";
        std::cout << "  Processing: " << (use_kernel_fusion ? "Multi-variable fusion" : "Single variable") << "\n";
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data.bin> [options]\n";
        std::cerr << "Options:\n";
        std::cerr << "  --second-order    Include second-order neighbors\n";
        std::cerr << "  --uniform         Use uniform weights (default: Gaussian)\n";
        std::cerr << "  --single          Process single variable (default: fusion)\n";
        return 1;
    }

    bool use_second_order = false;
    bool use_gaussian = true;
    bool use_fusion = true;
    
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--second-order") use_second_order = true;
        if (arg == "--uniform") use_gaussian = false;
        if (arg == "--single") use_fusion = false;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << "\n\n";

    HexGridCUDAv5 grid;
    grid.setOptions(use_second_order, use_gaussian, use_fusion);
    grid.loadFromBinary(argv[1]);
    grid.printConfiguration();

    const int iterations = 100;
    
    std::cout << "\n=== CUDA v5: Extended Neighborhood Smoothing ===\n";
    
    double time = grid.benchmark(iterations);
    if (use_fusion) {
        std::cout << "Total time (4 variables): " << time << " μs\n";
        std::cout << "Per-variable time: " << time / NUM_VARIABLES << " μs\n";
    } else {
        std::cout << "Single variable time: " << time << " μs\n";
    }
    
    grid.smooth();
    grid.saveResults("smoothed_cuda_v5.csv");
    
    // Benchmark all configurations
    if (argc == 2) {  // No specific options, run all benchmarks
        std::cout << "\n=== Comprehensive Benchmark ===\n";
        
        struct Config {
            bool second_order;
            bool gaussian;
            bool fusion;
            const char* name;
        };
        
        Config configs[] = {
            {false, true, false, "1st-order, Gaussian, Single"},
            {false, true, true,  "1st-order, Gaussian, Fusion"},
            {true,  true, false, "Both-orders, Gaussian, Single"},
            {true,  true, true,  "Both-orders, Gaussian, Fusion"},
            {true,  false, true, "Both-orders, Uniform, Fusion"},
        };
        
        for (const auto& config : configs) {
            HexGridCUDAv5 test_grid;
            test_grid.setOptions(config.second_order, config.gaussian, config.fusion);
            test_grid.loadFromBinary(argv[1]);
            
            double test_time = test_grid.benchmark(iterations);
            std::cout << config.name << ": " << test_time << " μs";
            if (config.fusion) {
                std::cout << " (" << test_time / NUM_VARIABLES << " μs/var)";
            }
            std::cout << "\n";
        }
    }
    
    return 0;
}
