// hex_smooth_optimized.cpp - Fixed compilation errors
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <algorithm>
#include <iomanip>

// For prefetch - check if available
#ifdef __SSE__
#include <xmmintrin.h>
#endif

class HexGridOptimized {
public:  // Move constants to public for access in main
    // Optimization parameters
    static constexpr int CHUNK_SIZE = 1024;
    static constexpr int PREFETCH_DISTANCE = 8;

private:
    // Structure of Arrays for better cache performance
    std::vector<float> ndvi_values;
    std::vector<int> neighbor_indices;  // All neighbors in one list
    std::vector<int> neighbor_offsets;  // Where each hex's neighbors start
    std::vector<int> neighbor_counts;   // How many neighbors per hex
    std::vector<float> smoothed_values;
    int n_hexagons;

public:
    void loadFromBinary(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        file.read(reinterpret_cast<char*>(&n_hexagons), sizeof(int));
        
        // Pre-allocate
        ndvi_values.resize(n_hexagons);
        neighbor_offsets.resize(n_hexagons + 1);
        neighbor_counts.resize(n_hexagons);
        smoothed_values.resize(n_hexagons);
        
        // Read data and build flattened structure
        std::vector<std::vector<int>> temp_neighbors(n_hexagons);
        int total_neighbors = 0;
        
        for (int i = 0; i < n_hexagons; i++) {
            file.read(reinterpret_cast<char*>(&ndvi_values[i]), sizeof(float));
            
            int n_neighbors;
            file.read(reinterpret_cast<char*>(&n_neighbors), sizeof(int));
            neighbor_counts[i] = n_neighbors;
            
            temp_neighbors[i].resize(n_neighbors);
            for (int j = 0; j < n_neighbors; j++) {
                file.read(reinterpret_cast<char*>(&temp_neighbors[i][j]), sizeof(int));
            }
            
            total_neighbors += n_neighbors;
        }
        file.close();
        
        // Build flattened neighbor structure
        neighbor_indices.resize(total_neighbors);
        neighbor_offsets[0] = 0;
        
        int idx = 0;
        for (int i = 0; i < n_hexagons; i++) {
            for (size_t j = 0; j < temp_neighbors[i].size(); j++) {  // Fixed: use size_t
                neighbor_indices[idx++] = temp_neighbors[i][j];
            }
            neighbor_offsets[i + 1] = idx;
        }
        
        std::cout << "Loaded " << n_hexagons << " hexagons with " 
                  << total_neighbors << " total neighbor connections\n";
    }

    void smoothSimpleAverageOptimized() {
        #pragma omp parallel
        {
            // Thread-local storage to avoid false sharing
            std::vector<float> local_sums(CHUNK_SIZE);
            std::vector<int> local_counts(CHUNK_SIZE);
            
            #pragma omp for schedule(guided, CHUNK_SIZE) nowait
            for (int chunk_start = 0; chunk_start < n_hexagons; chunk_start += CHUNK_SIZE) {
                int chunk_end = std::min(chunk_start + CHUNK_SIZE, n_hexagons);
                
                // Process chunk
                for (int i = chunk_start; i < chunk_end; i++) {
                    int local_idx = i - chunk_start;
                    
                    // Prefetch next hexagon data
                    #ifdef __SSE__
                    if (i + PREFETCH_DISTANCE < n_hexagons) {
                        _mm_prefetch((const char*)&ndvi_values[i + PREFETCH_DISTANCE], _MM_HINT_T0);
                    }
                    #endif
                    
                    local_sums[local_idx] = ndvi_values[i];
                    local_counts[local_idx] = 1;
                    
                    // Process neighbors
                    int start = neighbor_offsets[i];
                    int end = neighbor_offsets[i + 1];
                    
                    // Manual unroll for common case (6 neighbors)
                    int n_neighbors = end - start;
                    if (n_neighbors == 6) {
                        // Unrolled loop for exactly 6 neighbors
                        for (int j = 0; j < 6; j++) {
                            int neighbor_idx = neighbor_indices[start + j];
                            if (neighbor_idx >= 0 && neighbor_idx < n_hexagons) {
                                local_sums[local_idx] += ndvi_values[neighbor_idx];
                                local_counts[local_idx]++;
                            }
                        }
                    } else {
                        // General case
                        for (int j = start; j < end; j++) {
                            int neighbor_idx = neighbor_indices[j];
                            if (neighbor_idx >= 0 && neighbor_idx < n_hexagons) {
                                local_sums[local_idx] += ndvi_values[neighbor_idx];
                                local_counts[local_idx]++;
                            }
                        }
                    }
                    
                    // Store result
                    smoothed_values[i] = local_sums[local_idx] / local_counts[local_idx];
                }
            }
        }
    }

    void smoothGaussianOptimized(float sigma) {
        const float weight_center = 1.0f;
        const float weight_neighbor = exp(-1.0f / (2.0f * sigma * sigma));
        
        #pragma omp parallel for schedule(guided, CHUNK_SIZE)
        for (int i = 0; i < n_hexagons; i++) {
            // Prefetch
            #ifdef __SSE__
            if (i + PREFETCH_DISTANCE < n_hexagons) {
                _mm_prefetch((const char*)&ndvi_values[i + PREFETCH_DISTANCE], _MM_HINT_T0);
            }
            #endif
            
            float weighted_sum = ndvi_values[i] * weight_center;
            float weight_sum = weight_center;
            
            // Process neighbors
            int start = neighbor_offsets[i];
            int end = neighbor_offsets[i + 1];
            
            for (int j = start; j < end; j++) {
                int neighbor_idx = neighbor_indices[j];
                if (neighbor_idx >= 0 && neighbor_idx < n_hexagons) {
                    weighted_sum += ndvi_values[neighbor_idx] * weight_neighbor;
                    weight_sum += weight_neighbor;
                }
            }
            
            smoothed_values[i] = weighted_sum / weight_sum;
        }
    }

    double benchmarkSimple(int iterations) {
        // Warm-up
        smoothSimpleAverageOptimized();
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; i++) {
            smoothSimpleAverageOptimized();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count() / (double)iterations;
    }

    double benchmarkGaussian(float sigma, int iterations) {
        // Warm-up
        smoothGaussianOptimized(sigma);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; i++) {
            smoothGaussianOptimized(sigma);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count() / (double)iterations;
    }

    void saveResults(const std::string& filename) {
        std::ofstream file(filename);
        for (int i = 0; i < n_hexagons; i++) {
            file << smoothed_values[i] << "\n";
        }
        file.close();
    }

    void printStats() {
        int min_neighbors = 999, max_neighbors = 0;
        float avg_neighbors = 0;
        
        for (int i = 0; i < n_hexagons; i++) {
            int n = neighbor_counts[i];
            min_neighbors = std::min(min_neighbors, n);
            max_neighbors = std::max(max_neighbors, n);
            avg_neighbors += n;
        }
        avg_neighbors /= n_hexagons;
        
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

    try {
        HexGridOptimized grid;
        grid.loadFromBinary(argv[1]);
        grid.printStats();
        
        const int iterations = 10;
        
        std::cout << "\n=== Optimized CPU Implementation ===\n";
        std::cout << "Using " << omp_get_max_threads() << " OpenMP threads\n";
        std::cout << "Chunk size: " << HexGridOptimized::CHUNK_SIZE << "\n";
        
        // Benchmark simple average
        double time_simple = grid.benchmarkSimple(iterations);
        std::cout << "Simple Average: " << std::fixed << std::setprecision(2) 
                  << time_simple << " μs\n";
        
        // Save result
        grid.smoothSimpleAverageOptimized();
        grid.saveResults("smoothed_optimized.csv");
        
        // Benchmark Gaussian
        double time_gaussian = grid.benchmarkGaussian(1.0f, iterations);
        std::cout << "Gaussian: " << time_gaussian << " μs\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}