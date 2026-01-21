# KV Cache CUDA Kernels Makefile
#
# Builds libkvcache.a static library for CGO linking.

# CUDA Configuration - auto-detect if not set
NVCC := $(shell which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)

# Detect CUDA paths based on system
ifeq ($(wildcard /usr/local/cuda/include/cuda_runtime.h),)
  # System install (apt/yum) - headers in /usr/include, libs in /usr/lib
  CUDA_INCLUDE := /usr/include
  UNAME_M := $(shell uname -m)
  ifeq ($(UNAME_M),aarch64)
    CUDA_LIB := /usr/lib/aarch64-linux-gnu
  else
    CUDA_LIB := /usr/lib/x86_64-linux-gnu
  endif
else
  # Standard CUDA install
  CUDA_HOME ?= /usr/local/cuda
  CUDA_INCLUDE := $(CUDA_HOME)/include
  CUDA_LIB := $(CUDA_HOME)/lib64
endif

# Compiler flags
NVCC_FLAGS := -O3 -std=c++17 \
              -gencode arch=compute_75,code=sm_75 \
              -gencode arch=compute_80,code=sm_80 \
              -gencode arch=compute_86,code=sm_86 \
              -gencode arch=compute_89,code=sm_89 \
              -gencode arch=compute_90,code=sm_90 \
              --compiler-options -fPIC

# Include paths
INCLUDES := -I$(CUDA_INCLUDE)

# Library paths
LDFLAGS := -L$(CUDA_LIB) -lcudart

# Source files - KV Cache operations
KVCACHE_SRCS := kvcache_memory.cu kvcache_ops.cu
KVCACHE_OBJS := $(KVCACHE_SRCS:.cu=.o)
KVCACHE_DEPS := kvcache.h
KVCACHE_LIB := libkvcache.a

# Source files - Compression
COMPRESS_SRCS := compression_quant.cu compression_sparse.cu compression_delta.cu
COMPRESS_OBJS := $(COMPRESS_SRCS:.cu=.o)
COMPRESS_DEPS := compression.h
COMPRESS_LIB := libkvcache_compression.a

# All sources
SRCS := $(KVCACHE_SRCS) $(COMPRESS_SRCS)
OBJS := $(KVCACHE_OBJS) $(COMPRESS_OBJS)
DEPS := $(KVCACHE_DEPS) $(COMPRESS_DEPS)

# Output libraries
LIBS := $(KVCACHE_LIB) $(COMPRESS_LIB)

# Default target
.PHONY: all clean test

all: $(LIBS)

$(KVCACHE_LIB): $(KVCACHE_OBJS)
	ar rcs $@ $^
	@echo "Built $(KVCACHE_LIB)"

$(COMPRESS_LIB): $(COMPRESS_OBJS)
	ar rcs $@ $^
	@echo "Built $(COMPRESS_LIB)"

# KV Cache objects
kvcache_memory.o: kvcache_memory.cu $(KVCACHE_DEPS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

kvcache_ops.o: kvcache_ops.cu $(KVCACHE_DEPS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Compression objects (need CUB for reductions)
compression_quant.o: compression_quant.cu $(COMPRESS_DEPS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

compression_sparse.o: compression_sparse.cu $(COMPRESS_DEPS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

compression_delta.o: compression_delta.cu $(COMPRESS_DEPS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(LIBS)

# Test compilation
test: $(LIBS)
	@echo "Testing kvcache library linkage..."
	@echo '#include "kvcache.h"\nint main() { return 0; }' | \
		$(NVCC) -x cu - -L. -lkvcache $(LDFLAGS) -o /tmp/kvcache_test && \
		rm /tmp/kvcache_test && \
		echo "KV Cache test passed!"
	@echo "Testing compression library linkage..."
	@echo '#include "compression.h"\nint main() { return 0; }' | \
		$(NVCC) -x cu - -L. -lkvcache_compression $(LDFLAGS) -o /tmp/compress_test && \
		rm /tmp/compress_test && \
		echo "Compression test passed!"

# Install to system (requires sudo)
install: $(LIBS)
	install -m 644 $(KVCACHE_LIB) /usr/local/lib/
	install -m 644 $(COMPRESS_LIB) /usr/local/lib/
	install -m 644 kvcache.h /usr/local/include/
	install -m 644 compression.h /usr/local/include/
	ldconfig

# Print configuration
info:
	@echo "CUDA_HOME: $(CUDA_HOME)"
	@echo "NVCC: $(NVCC)"
	@echo "Sources: $(SRCS)"
	@echo "Objects: $(OBJS)"
