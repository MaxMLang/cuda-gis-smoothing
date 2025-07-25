// hex_smooth_cuda_v5.cu - Extended neighborhood smoothing with recursive bisection reordering
#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <set>
#include <climits>
#include <cfloat>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

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
constexpr int MAX_NEIGHBORS_1ST = 6;   // 1st-order neighbors
constexpr int MAX_NEIGHBORS_2ND = 12;  // 2nd-order neighbors  
constexpr int MAX_NEIGHBORS_TOTAL = 18; // Max possible neighbors
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_VARIABLES = 4;
constexpr int MAX_BISECTION_LEVELS = 8; // Maximum recursion depth

// Gaussian weights based on hex distance
constexpr float WEIGHT_CENTER = 1.0f;
constexpr float WEIGHT_FIRST = 0.6065f;   // For distance 1
constexpr float WEIGHT_SECOND = 0.1353f;  // For distance sqrt(3)

// Coordinate structure for bisection
struct __align__(8) Coord2D {
    float x, y;
    
    __host__ __device__ Coord2D() : x(0), y(0) {}
    __host__ __device__ Coord2D(float x_, float y_) : x(x_), y(y_) {}
};

// Bisection key structure
struct __align__(8) BisectionKey {
    uint32_t key;
    int original_index;
    
    __host__ __device__ BisectionKey() : key(0), original_index(0) {}
    __host__ __device__ BisectionKey(uint32_t k, int idx) : key(k), original_index(idx) {}
};

// Data structure with reordering support
struct HexagonGPUv5 {
    float* values[NUM_VARIABLES];
    int* neighbor_indices_1st;     // 1st-order neighbors
    int* neighbor_indices_2nd;     // 2nd-order neighbors
    int* neighbor_counts_1st;      // How many 1st order neighbors
    int* neighbor_counts_2nd;      // How many 2nd order neighbors
    float* smoothed_values[NUM_VARIABLES];
    
    // Reordering data
    Coord2D* coordinates;          // Original coordinates
    int* reorder_map;             // Maps original -> reordered
    int* inverse_reorder_map;     // Maps reordered -> original
    BisectionKey* bisection_keys; // Keys for sorting
    
    int n_hexagons;
    bool use_reordering;
};

// Functor for comparing bisection keys
struct BisectionKeyCompare {
    __device__ __host__ bool operator()(const BisectionKey& a, const BisectionKey& b) const {
        return a.key < b.key;
    }
};

// Functor for extracting original index from bisection key
struct ExtractIndex {
    __device__ __host__ int operator()(const BisectionKey& key) const {
        return key.original_index;
    }
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

// Scan operation for counting elements in quadrants
__global__ void countQuadrantElements(
    const Coord2D* __restrict__ coords,
    int* __restrict__ quadrant_counts,
    int n_elements,
    float mid_x, float mid_y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    
    Coord2D coord = coords[idx];
    int quadrant = 0;
    
    if (coord.x >= mid_x) quadrant |= 1;
    if (coord.y >= mid_y) quadrant |= 2;
    
    // Use atomic to count elements in each quadrant
    atomicAdd(&quadrant_counts[quadrant], 1);
}

// Generate bisection keys recursively
__global__ void generateBisectionKeys(
    const Coord2D* __restrict__ coords,
    BisectionKey* __restrict__ keys,
    int* __restrict__ original_indices,
    int n_elements,
    float min_x, float max_x, float min_y, float max_y,
    int level, int max_levels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    
    Coord2D coord = coords[idx];
    uint32_t key = 0;
    
    // Generate hierarchical key based on recursive bisection
    float mid_x = (min_x + max_x) * 0.5f;
    float mid_y = (min_y + max_y) * 0.5f;
    
    for (int l = 0; l < level && l < max_levels; l++) {
        key <<= 2; // 2 bits per level for 4 quadrants
        
        int quadrant = 0;
        if (coord.x >= mid_x) quadrant |= 1;
        if (coord.y >= mid_y) quadrant |= 2;
        
        key |= quadrant;
        
        // Update bounds for next level
        if (quadrant & 1) min_x = mid_x;
        else max_x = mid_x;
        if (quadrant & 2) min_y = mid_y;
        else max_y = mid_y;
        
        mid_x = (min_x + max_x) * 0.5f;
        mid_y = (min_y + max_y) * 0.5f;
    }
    
    keys[idx] = BisectionKey(key, original_indices[idx]);
}

// Reorder data using the computed mapping
__global__ void reorderData(
    const float* __restrict__ input_values,
    float* __restrict__ output_values,
    const int* __restrict__ reorder_map,
    int n_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    
    int new_idx = reorder_map[idx];
    output_values[new_idx] = input_values[idx];
}

// Reorder neighbor indices
__global__ void reorderNeighbors(
    const int* __restrict__ input_neighbors,
    int* __restrict__ output_neighbors,
    const int* __restrict__ reorder_map,
    const int* __restrict__ inverse_reorder_map,
    int n_hexagons,
    int max_neighbors)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_hexagons) return;
    
    int new_idx = reorder_map[idx];
    
    for (int i = 0; i < max_neighbors; i++) {
        int neighbor = input_neighbors[idx + i * n_hexagons];
        if (neighbor >= 0 && neighbor < n_hexagons) {
            // Map neighbor to new ordering
            output_neighbors[new_idx + i * n_hexagons] = reorder_map[neighbor];
        } else {
            output_neighbors[new_idx + i * n_hexagons] = neighbor;
        }
    }
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
    
    // Reordering data
    std::vector<Coord2D> h_coordinates;
    std::vector<int> h_reorder_map;
    std::vector<int> h_inverse_reorder_map;
    std::vector<BisectionKey> h_bisection_keys;
    
    bool use_second_order = false;
    bool use_gaussian_weights = true;
    bool use_kernel_fusion = true;
    bool use_recursive_bisection = true;
    int bisection_levels = 6;
    
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

    // Create recursive bisection reordering using scan operations
    void createRecursiveBisectionReordering(const std::vector<int>& hex_ids) {
        int n = hex_ids.size();
        h_coordinates.resize(n);
        h_reorder_map.resize(n);
        h_inverse_reorder_map.resize(n);
        h_bisection_keys.resize(n);
        
        // Extract coordinates from hex_ids (assuming hex_id contains x,y info)
        float min_x = FLT_MAX, max_x = -FLT_MAX;
        float min_y = FLT_MAX, max_y = -FLT_MAX;
        
        for (int i = 0; i < n; i++) {
            // Extract x,y from hex_id (based on your encoding)
            int hex_x = hex_ids[i] / 10000;
            int hex_y = hex_ids[i] % 10000;
            
            // Convert to float coordinates
            float x = (float)hex_x;
            float y = (float)hex_y;
            
            h_coordinates[i] = Coord2D(x, y);
            
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);
        }
        
        // Allocate device memory for reordering
        CUDA_CHECK(cudaMalloc(&d_data.coordinates, n * sizeof(Coord2D)));
        CUDA_CHECK(cudaMalloc(&d_data.bisection_keys, n * sizeof(BisectionKey)));
        CUDA_CHECK(cudaMalloc(&d_data.reorder_map, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_data.inverse_reorder_map, n * sizeof(int)));
        
        // Copy coordinates to device
        CUDA_CHECK(cudaMemcpy(d_data.coordinates, h_coordinates.data(),
                              n * sizeof(Coord2D), cudaMemcpyHostToDevice));
        
        // Create original indices array
        thrust::device_vector<int> d_original_indices(n);
        thrust::sequence(d_original_indices.begin(), d_original_indices.end());
        
        // Generate bisection keys
        const int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        generateBisectionKeys<<<gridSize, BLOCK_SIZE>>>(
            d_data.coordinates,
            d_data.bisection_keys,
            thrust::raw_pointer_cast(d_original_indices.data()),
            n, min_x, max_x, min_y, max_y,
            bisection_levels, MAX_BISECTION_LEVELS
        );
        CUDA_CHECK(cudaGetLastError());
        
        // Sort by bisection keys using thrust
        thrust::device_vector<BisectionKey> d_keys(n);
        CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_keys.data()), d_data.bisection_keys,
                              n * sizeof(BisectionKey), cudaMemcpyDeviceToDevice));
        
        thrust::sort(d_keys.begin(), d_keys.end(), BisectionKeyCompare());
        
        // Create reorder maps
        thrust::device_vector<int> d_reorder_map(n);
        thrust::device_vector<int> d_inverse_reorder_map(n);
        
        // Extract original indices from sorted keys
        thrust::transform(d_keys.begin(), d_keys.end(), d_reorder_map.begin(), ExtractIndex());
        
        // Create inverse mapping using scan-like operation
        thrust::device_vector<int> d_temp(n);
        thrust::sequence(d_temp.begin(), d_temp.end());
        thrust::scatter(d_temp.begin(), d_temp.end(), d_reorder_map.begin(), d_inverse_reorder_map.begin());
        
        // Copy back to host
        CUDA_CHECK(cudaMemcpy(h_reorder_map.data(), thrust::raw_pointer_cast(d_reorder_map.data()),
                              n * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_inverse_reorder_map.data(), thrust::raw_pointer_cast(d_inverse_reorder_map.data()),
                              n * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(d_data.reorder_map, thrust::raw_pointer_cast(d_reorder_map.data()),
                              n * sizeof(int), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_data.inverse_reorder_map, thrust::raw_pointer_cast(d_inverse_reorder_map.data()),
                              n * sizeof(int), cudaMemcpyDeviceToDevice));
        
        std::cout << "Created recursive bisection reordering with " << bisection_levels << " levels\n";
        
        // Calculate spatial locality improvement
        int total_distance = 0;
        for (int i = 1; i < n; i++) {
            int prev_idx = h_inverse_reorder_map[i-1];
            int curr_idx = h_inverse_reorder_map[i];
            int distance = abs(prev_idx - curr_idx);
            total_distance += distance;
        }
        float avg_distance = (float)total_distance / (n - 1);
        std::cout << "Average distance between consecutive elements: " << avg_distance << "\n";
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
        
        // Free reordering memory
        if (d_data.coordinates) cudaFree(d_data.coordinates);
        if (d_data.bisection_keys) cudaFree(d_data.bisection_keys);
        if (d_data.reorder_map) cudaFree(d_data.reorder_map);
        if (d_data.inverse_reorder_map) cudaFree(d_data.inverse_reorder_map);
    }

    void setOptions(bool second_order, bool gaussian, bool fusion, bool bisection = true, int levels = 6) {
        use_second_order = second_order;
        use_gaussian_weights = gaussian;
        use_kernel_fusion = fusion;
        use_recursive_bisection = bisection;
        bisection_levels = levels;
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
        
        // Apply recursive bisection reordering if enabled
        if (use_recursive_bisection) {
            std::vector<int> hex_ids;
            hex_ids.reserve(n_hexagons);
            
            // Create hex_ids from indices (assuming they can be derived from position)
            for (int i = 0; i < n_hexagons; i++) {
                hex_ids.push_back(325000000 + i);  // Placeholder encoding
            }
            
            createRecursiveBisectionReordering(hex_ids);
            
            // Reorder the data using the computed mapping
            std::vector<std::vector<float>> reordered_values(NUM_VARIABLES);
            std::vector<int> reordered_counts_1st(n_hexagons);
            std::vector<int> reordered_indices_1st(n_hexagons * MAX_NEIGHBORS_1ST, -1);
            
            // Reorder values
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
                reordered_counts_1st[new_idx] = h_neighbor_counts_1st[i];
                
                for (int j = 0; j < MAX_NEIGHBORS_1ST; j++) {
                    int neighbor = h_neighbor_indices_1st_padded[i + j * n_hexagons];
                    if (neighbor >= 0 && neighbor < n_hexagons) {
                        reordered_indices_1st[new_idx + j * n_hexagons] = h_reorder_map[neighbor];
                    } else {
                        reordered_indices_1st[new_idx + j * n_hexagons] = neighbor;
                    }
                }
            }
            
            h_neighbor_counts_1st = std::move(reordered_counts_1st);
            h_neighbor_indices_1st_padded = std::move(reordered_indices_1st);
            
            // Reorder second-order neighbors if they exist
            if (use_second_order) {
                std::vector<int> reordered_counts_2nd(n_hexagons);
                std::vector<int> reordered_indices_2nd(n_hexagons * MAX_NEIGHBORS_2ND, -1);
                
                for (int i = 0; i < n_hexagons; i++) {
                    int new_idx = h_reorder_map[i];
                    reordered_counts_2nd[new_idx] = h_neighbor_counts_2nd[i];
                    
                    for (int j = 0; j < MAX_NEIGHBORS_2ND; j++) {
                        int neighbor = h_neighbor_indices_2nd_padded[i + j * n_hexagons];
                        if (neighbor >= 0 && neighbor < n_hexagons) {
                            reordered_indices_2nd[new_idx + j * n_hexagons] = h_reorder_map[neighbor];
                        } else {
                            reordered_indices_2nd[new_idx + j * n_hexagons] = neighbor;
                        }
                    }
                }
                
                h_neighbor_counts_2nd = std::move(reordered_counts_2nd);
                h_neighbor_indices_2nd_padded = std::move(reordered_indices_2nd);
            }
            
            // Update device memory with reordered data
            for (int i = 0; i < NUM_VARIABLES; i++) {
                CUDA_CHECK(cudaMemcpy(d_data.values[i], h_values[i].data(),
                                      n_hexagons * sizeof(float), cudaMemcpyHostToDevice));
            }
            
            CUDA_CHECK(cudaMemcpy(d_data.neighbor_indices_1st, h_neighbor_indices_1st_padded.data(),
                                  n_hexagons * MAX_NEIGHBORS_1ST * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_data.neighbor_counts_1st, h_neighbor_counts_1st.data(),
                                  n_hexagons * sizeof(int), cudaMemcpyHostToDevice));
            
            if (use_second_order) {
                CUDA_CHECK(cudaMemcpy(d_data.neighbor_indices_2nd, h_neighbor_indices_2nd_padded.data(),
                                      n_hexagons * MAX_NEIGHBORS_2ND * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_data.neighbor_counts_2nd, h_neighbor_counts_2nd.data(),
                                      n_hexagons * sizeof(int), cudaMemcpyHostToDevice));
            }
        }
        
        std::cout << "Loaded " << n_hexagons << " hexagons";
        if (use_recursive_bisection) {
            std::cout << " with recursive bisection reordering";
        }
        std::cout << "\n";
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
            int output_idx = use_recursive_bisection ? h_inverse_reorder_map[i] : i;
            file << h_smoothed_values[0][output_idx] << "\n";
        }
        file.close();
    }

    void printConfiguration() {
        std::cout << "Configuration:\n";
        std::cout << "  Neighbor orders: " << (use_second_order ? "1st + 2nd" : "1st only") << "\n";
        std::cout << "  Weights: " << (use_gaussian_weights ? "Gaussian" : "Uniform") << "\n";
        std::cout << "  Processing: " << (use_kernel_fusion ? "Multi-variable fusion" : "Single variable") << "\n";
        std::cout << "  Reordering: " << (use_recursive_bisection ? "Recursive bisection (" + std::to_string(bisection_levels) + " levels)" : "None") << "\n";
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data.bin> [options]\n";
        std::cerr << "Options:\n";
        std::cerr << "  --second-order    Include second-order neighbors\n";
        std::cerr << "  --uniform         Use uniform weights (default: Gaussian)\n";
        std::cerr << "  --single          Process single variable (default: fusion)\n";
        std::cerr << "  --no-reorder      Disable recursive bisection reordering\n";
        std::cerr << "  --levels N        Set bisection levels (default: 6)\n";
        return 1;
    }

    bool use_second_order = false;
    bool use_gaussian = true;
    bool use_fusion = true;
    bool use_reorder = true;
    int bisection_levels = 6;
    
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--second-order") use_second_order = true;
        if (arg == "--uniform") use_gaussian = false;
        if (arg == "--single") use_fusion = false;
        if (arg == "--no-reorder") use_reorder = false;
        if (arg == "--levels" && i + 1 < argc) {
            bisection_levels = std::stoi(argv[i + 1]);
            i++; // Skip next argument
        }
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << "\n\n";

    HexGridCUDAv5 grid;
    grid.setOptions(use_second_order, use_gaussian, use_fusion, use_reorder, bisection_levels);
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
            bool reorder;
            int levels;
            const char* name;
        };
        
        Config configs[] = {
            {false, true, false, true,  6, "1st-order, Gaussian, Single, Reordered"},
            {false, true, true,  true,  6, "1st-order, Gaussian, Fusion, Reordered"},
            {true,  true, false, true,  6, "Both-orders, Gaussian, Single, Reordered"},
            {true,  true, true,  true,  6, "Both-orders, Gaussian, Fusion, Reordered"},
            {true,  false, true, true,  6, "Both-orders, Uniform, Fusion, Reordered"},
            {false, true, true,  false, 0, "1st-order, Gaussian, Fusion, No Reorder"},
            {true,  true, true,  false, 0, "Both-orders, Gaussian, Fusion, No Reorder"},
        };
        
        for (const auto& config : configs) {
            HexGridCUDAv5 test_grid;
            test_grid.setOptions(config.second_order, config.gaussian, config.fusion, config.reorder, config.levels);
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
