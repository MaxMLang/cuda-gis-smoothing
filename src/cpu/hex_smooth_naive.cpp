// hex_smooth_naive.cpp - Fixed version
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cmath>
#include <iomanip>

struct Hexagon {
    float ndvi;
    std::vector<int> neighbors;
};

class HexGridNaive {
private:
    std::vector<Hexagon> hexagons;
    std::vector<float> smoothed_values;

public:
    void loadFromBinary(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        int n_hexagons;
        file.read(reinterpret_cast<char*>(&n_hexagons), sizeof(int));
        
        hexagons.resize(n_hexagons);
        smoothed_values.resize(n_hexagons);

        for (int i = 0; i < n_hexagons; i++) {
            file.read(reinterpret_cast<char*>(&hexagons[i].ndvi), sizeof(float));
            
            int n_neighbors;
            file.read(reinterpret_cast<char*>(&n_neighbors), sizeof(int));
            
            hexagons[i].neighbors.resize(n_neighbors);
            for (int j = 0; j < n_neighbors; j++) {
                file.read(reinterpret_cast<char*>(&hexagons[i].neighbors[j]), sizeof(int));
            }
        }
        file.close();
        
        std::cout << "Loaded " << n_hexagons << " hexagons\n";
    }

    void smoothSimpleAverage() {
        for (size_t i = 0; i < hexagons.size(); i++) {
            float sum = hexagons[i].ndvi;
            int count = 1;
            
            for (size_t j = 0; j < hexagons[i].neighbors.size(); j++) {
                int neighbor_idx = hexagons[i].neighbors[j];
                if (neighbor_idx >= 0 && neighbor_idx < (int)hexagons.size()) {
                    sum += hexagons[neighbor_idx].ndvi;
                    count++;
                }
            }
            
            smoothed_values[i] = sum / count;
        }
    }

    void smoothGaussian(float sigma = 1.0f) {
        float weight_center = 1.0f;
        float weight_neighbor = exp(-1.0f / (2.0f * sigma * sigma));

        for (size_t i = 0; i < hexagons.size(); i++) {
            float weighted_sum = hexagons[i].ndvi * weight_center;
            float weight_sum = weight_center;
            
            for (size_t j = 0; j < hexagons[i].neighbors.size(); j++) {
                int neighbor_idx = hexagons[i].neighbors[j];
                if (neighbor_idx >= 0 && neighbor_idx < (int)hexagons.size()) {
                    weighted_sum += hexagons[neighbor_idx].ndvi * weight_neighbor;
                    weight_sum += weight_neighbor;
                }
            }
            
            smoothed_values[i] = weighted_sum / weight_sum;
        }
    }

    // Fixed: separate benchmark functions for different signatures
    double benchmarkSimple(int iterations) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; i++) {
            smoothSimpleAverage();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count() / (double)iterations;
    }

    double benchmarkGaussian(float sigma, int iterations) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; i++) {
            smoothGaussian(sigma);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count() / (double)iterations;
    }

    void saveResults(const std::string& filename) {
        std::ofstream file(filename);
        for (size_t i = 0; i < smoothed_values.size(); i++) {
            file << smoothed_values[i] << "\n";
        }
        file.close();
    }

    void printStats() {
        int min_neighbors = 999, max_neighbors = 0;
        float avg_neighbors = 0;
        
        for (const auto& hex : hexagons) {
            int n = hex.neighbors.size();
            min_neighbors = std::min(min_neighbors, n);
            max_neighbors = std::max(max_neighbors, n);
            avg_neighbors += n;
        }
        avg_neighbors /= hexagons.size();
        
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
        HexGridNaive grid;
        grid.loadFromBinary(argv[1]);
        grid.printStats();
        
        const int iterations = 10;
        
        std::cout << "\n=== Naive CPU Implementation ===\n";
        
        // Benchmark simple average
        double time_simple = grid.benchmarkSimple(iterations);
        std::cout << "Simple Average: " << std::fixed << std::setprecision(2) 
                  << time_simple << " μs\n";
        
        // Save result
        grid.smoothSimpleAverage();
        grid.saveResults("smoothed_naive.csv");
        
        // Benchmark Gaussian
        double time_gaussian = grid.benchmarkGaussian(1.0f, iterations);
        std::cout << "Gaussian: " << time_gaussian << " μs\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}