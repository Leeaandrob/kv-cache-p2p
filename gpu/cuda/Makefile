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

# Source files
SRCS := kvcache_memory.cu kvcache_ops.cu
OBJS := $(SRCS:.cu=.o)
DEPS := kvcache.h

# Output library
LIB := libkvcache.a

# Default target
.PHONY: all clean test

all: $(LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^
	@echo "Built $(LIB)"

%.o: %.cu $(DEPS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(LIB)

# Test compilation
test: $(LIB)
	@echo "Testing library linkage..."
	@echo '#include "kvcache.h"\nint main() { return 0; }' | \
		$(NVCC) -x cu - -L. -lkvcache $(LDFLAGS) -o /tmp/kvcache_test && \
		rm /tmp/kvcache_test && \
		echo "Test passed!"

# Install to system (requires sudo)
install: $(LIB)
	install -m 644 $(LIB) /usr/local/lib/
	install -m 644 kvcache.h /usr/local/include/
	ldconfig

# Print configuration
info:
	@echo "CUDA_HOME: $(CUDA_HOME)"
	@echo "NVCC: $(NVCC)"
	@echo "Sources: $(SRCS)"
	@echo "Objects: $(OBJS)"
