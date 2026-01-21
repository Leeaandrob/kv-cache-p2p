.PHONY: all build cuda clean test demo help

# Directories
BIN_DIR := bin
CUDA_DIR := gpu/cuda

# Build flags
CGO_ENABLED := 1
CUDA_LDFLAGS := -L$(CUDA_DIR) -lkvcache -L/usr/local/cuda/lib64 -lcudart -lstdc++
CUDA_CFLAGS := -I$(CUDA_DIR)

all: build

help:
	@echo "KV Cache P2P - Build targets"
	@echo ""
	@echo "  make build    - Build demo (mock mode, no CUDA required)"
	@echo "  make cuda     - Build CUDA kernels + demo with CUDA support"
	@echo "  make test     - Run tests"
	@echo "  make demo     - Run demo in mock mode"
	@echo "  make clean    - Clean build artifacts"
	@echo ""

# Build demo without CUDA (mock mode)
build:
	@mkdir -p $(BIN_DIR)
	go build -o $(BIN_DIR)/demo ./cmd/demo
	@echo "Built: $(BIN_DIR)/demo (mock mode)"
	@echo "Run: ./$(BIN_DIR)/demo --mock --demo"

# Build CUDA kernels
cuda-kernels:
	@echo "Building CUDA kernels..."
	$(MAKE) -C $(CUDA_DIR)

# Build with CUDA support
cuda: cuda-kernels
	@mkdir -p $(BIN_DIR)
	CGO_ENABLED=$(CGO_ENABLED) \
	CGO_CFLAGS="$(CUDA_CFLAGS)" \
	CGO_LDFLAGS="$(CUDA_LDFLAGS)" \
	go build -o $(BIN_DIR)/demo-cuda ./cmd/demo
	@echo "Built: $(BIN_DIR)/demo-cuda (CUDA enabled)"

# Run tests
test:
	go test -v ./...

# Run demo in mock mode
demo: build
	./$(BIN_DIR)/demo --mock --demo

# Clean
clean:
	rm -rf $(BIN_DIR)
	$(MAKE) -C $(CUDA_DIR) clean 2>/dev/null || true
	go clean ./...

# Format code
fmt:
	go fmt ./...

# Lint
lint:
	go vet ./...
