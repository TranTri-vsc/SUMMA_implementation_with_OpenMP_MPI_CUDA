MPICXX := CC
NVCC   ?= nvcc

# Optional MPI include for NVCC host compilation
MPI_INC :=
ifdef CRAY_MPICH_DIR
  MPI_INC := -I$(CRAY_MPICH_DIR)/include
endif

# Flags
OMP_FLAG     ?= -fopenmp
BASE_FLAGS    = -O3 -std=c++17 -Iinclude $(MPI_INC)

CXXFLAGS_CPU  = $(BASE_FLAGS) $(OMP_FLAG)                  # no CUDA paths
CXXFLAGS_GPU  = $(BASE_FLAGS) -DENABLE_CUDA $(OMP_FLAG)    # enable GPU paths in host files

# NVCC uses the MPI wrapper as host compiler and passes OpenMP to host side
NVCCFLAGS = -O3 -std=c++17 -Iinclude -DENABLE_CUDA $(MPI_INC) \
            -ccbin=$(MPICXX) -Xcompiler="$(OMP_FLAG)"

# Sources
CPU_SRCS  = src/main.cpp src/summa_cpu.cpp src/local_dgemm_cpu.cpp src/verify.cpp
GPU_SRCS  = src/summa_gpu.cu src/local_dgemm_gpu.cu

# Objects (separate dirs so CPU/GPU variants of the same .cpp can coexist)
CPU_OBJS_CPU = $(CPU_SRCS:src/%.cpp=build/cpu/%.o)
CPU_OBJS_GPU = $(CPU_SRCS:src/%.cpp=build/gpu/%.o)
GPU_OBJS     = $(GPU_SRCS:src/%.cu=build/gpu/%.o)

# Target
TARGET ?= summa

# Default build
all: cpu

# Build rules
cpu: $(CPU_OBJS_CPU)
	$(MPICXX) $(CXXFLAGS_CPU) $^ -o $(TARGET)
gpu: $(CPU_OBJS_GPU) $(GPU_OBJS)
	$(MPICXX) $(CXXFLAGS_GPU) $^ -o $(TARGET) -L$(CUDA_HOME)/lib64 -lcudart

# Compile rules
build/cpu/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(MPICXX) $(CXXFLAGS_CPU) -c $< -o $@

build/gpu/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(MPICXX) $(CXXFLAGS_GPU) -c $< -o $@

build/gpu/%.o: src/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(TARGET) src/*.o
