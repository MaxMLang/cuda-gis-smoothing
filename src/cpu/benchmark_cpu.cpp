// benchmark_cpu.cpp
#include <iostream>
#include <iomanip>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data.bin>\n";
        return 1;
    }

    const int iterations = 10;
    
    // Test naive implementation
    std::cout << "=== CPU Performance Benchmark ===\n\n";
    
    HexGridNaive naive;
    naive.loadFromBinary(argv[1]);
    
    std::cout << "Naive CPU Implementation:\n";
    double naive_simple = naive.benchmark(&HexGridNaive::smoothSimpleAverage, 1.0f, iterations);
    std::cout << "  Simple Average: " << std::fixed << std::setprecision(2) 
              << naive_simple << " μs\n";
    
    double naive_gaussian = naive.benchmark(&HexGridNaive::smoothGaussian, 1.0f, iterations);
    std::cout << "  Gaussian:       " << naive_gaussian << " μs\n\n";
    
    // Test optimized implementation
    HexGridOptimized optimized;
    optimized.loadFromBinary(argv[1]);
    
    std::cout << "Optimized CPU Implementation:\n";
    
    // Test with different thread counts
    for (int threads : {1, 2, 4, 8, 16}) {
        omp_set_num_threads(threads);
        std::cout << "  Threads: " << threads << "\n";
        
        double opt_simple = optimized.benchmark(&HexGridOptimized::smoothSimpleAverageOptimized, 1.0f, iterations);
        std::cout << "    Simple Average: " << opt_simple << " μs";
        std::cout << " (Speedup: " << naive_simple / opt_simple << "x)\n";
        
        double opt_gaussian = optimized.benchmark(&HexGridOptimized::smoothGaussianOptimized, 1.0f, iterations);
        std::cout << "    Gaussian:       " << opt_gaussian << " μs";
        std::cout << " (Speedup: " << naive_gaussian / opt_gaussian << "x)\n";
    }
    
    return 0;
}