.PHONY: all build cuda clean test demo help build-compress validation-install validation-test validation

# Directories
BIN_DIR := bin
CUDA_DIR := gpu/cuda
VALIDATION_DIR := validation

# Build flags
CGO_ENABLED := 1
CUDA_LDFLAGS := -L$(CUDA_DIR) -lkvcache -L/usr/local/cuda/lib64 -lcudart -lstdc++
CUDA_CFLAGS := -I$(CUDA_DIR)

all: build build-compress

help:
	@echo "KV Cache P2P - Build targets"
	@echo ""
	@echo "  make build           - Build demo (mock mode, no CUDA required)"
	@echo "  make build-compress  - Build KV cache compression CLI"
	@echo "  make cuda            - Build CUDA kernels + demo with CUDA support"
	@echo "  make test            - Run Go tests"
	@echo "  make demo            - Run demo in mock mode"
	@echo "  make clean           - Clean build artifacts"
	@echo ""
	@echo "Validation targets:"
	@echo "  make validation-install  - Install Python validation dependencies"
	@echo "  make validation-test     - Run Python validation tests"
	@echo "  make validation          - Run full perplexity validation"
	@echo "  make validation-quick    - Run quick validation (no model download)"
	@echo ""

# Build demo without CUDA (mock mode)
build:
	@mkdir -p $(BIN_DIR)
	go build -o $(BIN_DIR)/demo ./cmd/demo
	@echo "Built: $(BIN_DIR)/demo (mock mode)"
	@echo "Run: ./$(BIN_DIR)/demo --mock --demo"

# Build compression CLI
build-compress:
	@mkdir -p $(BIN_DIR)
	go build -o $(BIN_DIR)/kv-compress ./cmd/compress
	@echo "Built: $(BIN_DIR)/kv-compress"
	@echo "Run: ./$(BIN_DIR)/kv-compress -h"

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

# Run Go tests
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
	find $(VALIDATION_DIR) -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find $(VALIDATION_DIR) -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

# Format code
fmt:
	go fmt ./...
	@if command -v ruff >/dev/null 2>&1; then \
		cd $(VALIDATION_DIR) && ruff format . 2>/dev/null || true; \
	fi

# Lint
lint:
	go vet ./...
	@if command -v ruff >/dev/null 2>&1; then \
		cd $(VALIDATION_DIR) && ruff check . 2>/dev/null || true; \
	fi

# ============================================================================
# Validation targets
# ============================================================================

# Install Python validation dependencies
validation-install:
	@echo "Installing Python validation dependencies..."
	pip install -r $(VALIDATION_DIR)/requirements.txt
	@echo "Done. Installed to current Python environment."

# Run Python validation tests
validation-test: build-compress
	@echo "Running Python validation tests..."
	cd $(VALIDATION_DIR) && pytest -v test_validation.py

# Run full perplexity validation (requires GPU and model download)
validation: build-compress
	@echo "Running full perplexity validation..."
	@echo "This will download TinyLlama (~1GB) on first run."
	cd $(VALIDATION_DIR) && python validate_perplexity.py

# Run quick validation (synthetic data, no model download)
validation-quick: build-compress
	@echo "Running quick validation with synthetic data..."
	cd $(VALIDATION_DIR) && python validate_perplexity.py --quick-test

# ============================================================================
# CI/CD targets
# ============================================================================

# Run all tests (Go + Python)
test-all: test validation-test
	@echo "All tests passed!"

# Build all binaries
build-all: build build-compress
	@echo "All binaries built!"

# Full CI check
ci: lint test-all
	@echo "CI checks passed!"
