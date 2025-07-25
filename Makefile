# Makefile for hex smoothing project

# Directories
BIN_DIR = bin
SRC_CUDA_DIR = src/cuda
SRC_CPU_DIR = src/cpu

# Compilers
CXX = g++
NVCC = nvcc

# Flags
CXXFLAGS = -O3 -std=c++17 -Wall
OMPFLAGS = -fopenmp
NVCCFLAGS = -O3 -arch=sm_70 -std=c++17 --extended-lambda --expt-relaxed-constexpr

# Targets
all: cpu cuda

cpu: $(BIN_DIR)/hex_smooth_naive $(BIN_DIR)/hex_smooth_optimized

cuda: $(BIN_DIR)/hex_smooth_cuda_v1 $(BIN_DIR)/hex_smooth_cuda_v2 \
      $(BIN_DIR)/hex_smooth_cuda_v3 $(BIN_DIR)/hex_smooth_cuda_v4 \
      $(BIN_DIR)/hex_smooth_cuda_v5

# Create bin directory
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# CPU naive
$(BIN_DIR)/hex_smooth_naive: $(SRC_CPU_DIR)/hex_smooth_naive.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC_CPU_DIR)/hex_smooth_naive.cpp

# CPU optimized  
$(BIN_DIR)/hex_smooth_optimized: $(SRC_CPU_DIR)/hex_smooth_optimized.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $(SRC_CPU_DIR)/hex_smooth_optimized.cpp

# CUDA v1
$(BIN_DIR)/hex_smooth_cuda_v1: $(SRC_CUDA_DIR)/hex_smooth_cuda_v1.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRC_CUDA_DIR)/hex_smooth_cuda_v1.cu

# CUDA v2
$(BIN_DIR)/hex_smooth_cuda_v2: $(SRC_CUDA_DIR)/hex_smooth_cuda_v2.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRC_CUDA_DIR)/hex_smooth_cuda_v2.cu

# CUDA v3
$(BIN_DIR)/hex_smooth_cuda_v3: $(SRC_CUDA_DIR)/hex_smooth_cuda_v3.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRC_CUDA_DIR)/hex_smooth_cuda_v3.cu

# CUDA v4
$(BIN_DIR)/hex_smooth_cuda_v4: $(SRC_CUDA_DIR)/hex_smooth_cuda_v4.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRC_CUDA_DIR)/hex_smooth_cuda_v4.cu

# CUDA v5
$(BIN_DIR)/hex_smooth_cuda_v5: $(SRC_CUDA_DIR)/hex_smooth_cuda_v5.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRC_CUDA_DIR)/hex_smooth_cuda_v5.cu

clean:
	rm -rf $(BIN_DIR)
	rm -f results/outputs/*.csv

.PHONY: all clean cpu cuda
