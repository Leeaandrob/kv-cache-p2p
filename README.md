# KV Cache P2P

<p align="center">
  <strong>Distributed KV Cache Layer for LLM Inference Acceleration</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#performance">Performance</a> •
  <a href="#api">API</a> •
  <a href="#license">License</a>
</p>

---

## Overview

**KV Cache P2P** is a high-performance distributed caching system designed to accelerate Large Language Model (LLM) inference by sharing Key-Value (KV) cache data between GPU nodes over a peer-to-peer network.

In transformer-based LLMs, the attention mechanism requires computing K (key) and V (value) tensors for each token. For long sequences or repeated prompts, recomputing these tensors is computationally expensive. This project enables:

- **Cache Reuse**: Share computed KV tensors between nodes instead of recomputing
- **P2P Distribution**: No central server - nodes discover and communicate directly
- **GPU-Native**: Direct GPU memory transfers via CUDA with pinned memory optimization
- **Cross-Platform**: Supports both ARM64 (GH200, Grace Hopper) and x86_64 (RTX, A100, H100)

### Key Benefits

| Without KV Cache P2P | With KV Cache P2P |
|---------------------|-------------------|
| Each node recomputes KV tensors | Nodes share computed KV tensors |
| ~200-500ms per forward pass | ~10-20ms P2P transfer |
| Duplicate GPU compute | Distributed compute once, share everywhere |
| Linear scaling cost | Sub-linear scaling with cache hits |

---

## Features

### Core Capabilities

- **Peer-to-Peer Networking**: Built on [libp2p](https://libp2p.io/) for robust P2P communication
- **Automatic Discovery**: mDNS-based peer discovery on local networks
- **NAT Traversal**: External IP announcement for WAN connectivity
- **CUDA Integration**: CGO bindings for direct GPU memory operations
- **Multi-Architecture**: Supports arm64 (GH200) and x86_64 (RTX/A100/H100)
- **Efficient Serialization**: msgpack-based wire protocol
- **Token-Based Caching**: SHA256 hash keys for prefix-based cache lookup

### GPU Support Matrix

| GPU | Architecture | Compute Capability | Status |
|-----|--------------|-------------------|--------|
| NVIDIA GH200 | arm64 | sm_90 | ✅ Tested |
| NVIDIA H100 | x86_64 | sm_90 | ✅ Supported |
| NVIDIA A100 | x86_64 | sm_80 | ✅ Supported |
| NVIDIA A10 | x86_64 | sm_86 | ✅ Tested |
| NVIDIA RTX 4090 | x86_64 | sm_89 | ✅ Tested |
| NVIDIA RTX 3090 | x86_64 | sm_86 | ✅ Supported |
| NVIDIA V100 | x86_64 | sm_70 | ⚠️ Untested |

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KV Cache P2P Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Node A (Producer)                         Node B (Consumer)               │
│   ┌───────────────────┐                    ┌───────────────────┐           │
│   │   LLM Inference   │                    │   LLM Inference   │           │
│   │   ┌───────────┐   │                    │   ┌───────────┐   │           │
│   │   │ Attention │   │                    │   │ Attention │   │           │
│   │   │  K  │  V  │   │                    │   │  K  │  V  │   │           │
│   │   └──┬──┴──┬──┘   │                    │   └──▲──┴──▲──┘   │           │
│   └──────┼─────┼──────┘                    └──────┼─────┼──────┘           │
│          │     │                                  │     │                   │
│   ┌──────▼─────▼──────┐                    ┌──────┼─────┼──────┐           │
│   │   GPU Memory      │                    │   GPU Memory      │           │
│   │   (Device KV)     │                    │   (Device KV)     │           │
│   └────────┬──────────┘                    └────────▲──────────┘           │
│            │ cudaMemcpyAsync (D2H)                  │ cudaMemcpyAsync (H2D)│
│   ┌────────▼──────────┐                    ┌────────┴──────────┐           │
│   │   Pinned Memory   │                    │   Pinned Memory   │           │
│   │   (Host Buffer)   │                    │   (Host Buffer)   │           │
│   └────────┬──────────┘                    └────────▲──────────┘           │
│            │ CGO                                    │ CGO                   │
│   ┌────────▼──────────┐                    ┌────────┴──────────┐           │
│   │   Go Runtime      │                    │   Go Runtime      │           │
│   │   ┌────────────┐  │                    │   ┌────────────┐  │           │
│   │   │ Cache Mgr  │  │                    │   │ Cache Mgr  │  │           │
│   │   └─────┬──────┘  │                    │   └──────▲─────┘  │           │
│   │         │         │                    │          │        │           │
│   │   ┌─────▼──────┐  │    libp2p stream   │   ┌──────┴─────┐  │           │
│   │   │ P2P Node   │◄─┼────────────────────┼──►│ P2P Node   │  │           │
│   │   └────────────┘  │     msgpack        │   └────────────┘  │           │
│   └───────────────────┘                    └───────────────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **GPU** | CUDA 12.x | KV cache storage, memory transfers |
| **Bridge** | CGO | Go ↔ C/CUDA interface |
| **Network** | libp2p | P2P communication, peer discovery |
| **Protocol** | msgpack | Binary message serialization |
| **Language** | Go 1.21+ | Orchestration, concurrency |

### Project Structure

```
kv-cache-p2p/
├── cmd/
│   └── demo/                  # Demo application
│       └── main.go
├── gpu/
│   ├── cuda/                  # CUDA kernels
│   │   ├── kvcache.h          # C header
│   │   ├── kvcache_memory.cu  # Memory operations
│   │   ├── kvcache_ops.cu     # KV operations
│   │   └── Makefile           # Build system
│   └── bindings/              # CGO bindings
│       └── kvcache.go         # Go ↔ CUDA interface
├── pkg/
│   ├── cache/                 # Cache management
│   │   ├── manager.go         # Local + P2P coordination
│   │   └── types.go           # CacheKey, CacheEntry
│   ├── gpu/                   # GPU abstraction
│   │   ├── types.go           # GPUConnector interface
│   │   ├── connector_cuda.go  # CUDA implementation
│   │   ├── mock.go            # Mock for testing
│   │   └── pool_*.go          # Pinned memory pool
│   ├── integration/           # Engine adapters
│   │   └── neurogrid.go       # neurogrid-engine integration
│   ├── protocol/              # Wire protocol
│   │   └── messages.go        # Message types, serialization
│   ├── storage/               # Storage backends
│   │   └── local.go           # In-memory storage
│   └── transport/             # Networking
│       └── p2p.go             # libp2p P2P node
├── docs/                      # Documentation
├── LICENSE                    # Source-available license
├── CHANGELOG.md               # Version history
└── README.md                  # This file
```

### Data Flow

#### Producer (Node computes KV):

```
1. Forward pass computes K, V tensors per layer
2. K, V stored in GPU device memory
3. cudaMemcpyAsync(pinnedHost, deviceKV, D2H)  // GPU → CPU
4. Go receives []byte from pinned buffer
5. Store in local cache with token hash key
6. Available for P2P requests from other nodes
```

#### Consumer (Node needs KV):

```
1. Generate token hash from input sequence
2. Lookup local cache → MISS
3. P2P Lookup → Query peers for key
4. P2P Get → Receive []byte via libp2p stream
5. Copy to pinned memory buffer
6. cudaMemcpyAsync(deviceKV, pinnedHost, H2D)  // CPU → GPU
7. K, V available in GPU for attention computation
```

---

## Installation

### Prerequisites

- **Go**: 1.21 or later
- **CUDA Toolkit**: 12.0 or later
- **NVIDIA Driver**: 525+ (for CUDA 12.x)
- **GCC/G++**: 11+ (for CUDA compilation)

### Platform-Specific Setup

#### Ubuntu/Debian (x86_64)

```bash
# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-4

# Set environment
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
nvidia-smi
```

#### Ubuntu/Debian (arm64 - GH200/Grace Hopper)

```bash
# CUDA is typically pre-installed on GH200 systems
# Verify installation
nvcc --version
nvidia-smi

# If not installed, use apt
sudo apt install nvidia-cuda-toolkit
```

#### Install Go

```bash
# x86_64
wget https://go.dev/dl/go1.23.4.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.23.4.linux-amd64.tar.gz

# arm64
wget https://go.dev/dl/go1.23.4.linux-arm64.tar.gz
sudo tar -C /usr/local -xzf go1.23.4.linux-arm64.tar.gz

# Add to PATH
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc
```

### Build from Source

```bash
# Clone repository
git clone https://github.com/neurogrid/kv-cache-p2p.git
cd kv-cache-p2p

# Build CUDA kernels
cd gpu/cuda
make clean && make
cd ../..

# Build with CUDA support
CGO_ENABLED=1 go build -tags cuda -o bin/demo-cuda ./cmd/demo

# Build without CUDA (mock mode)
go build -o bin/demo ./cmd/demo
```

### Verify Installation

```bash
# With CUDA
./bin/demo-cuda --role demo --mock  # Mock mode test
./bin/demo-cuda --role demo         # Full CUDA test (requires GPU)

# Without CUDA
./bin/demo --role demo --mock
```

---

## Usage

### Quick Start - Demo Mode

Run a self-contained demo with producer and consumer in the same process:

```bash
./bin/demo-cuda --role demo --mock
```

Expected output:
```
======================================================================
       KV Cache P2P Demo - Go -> libp2p -> CGO -> CUDA
======================================================================

-------------------- Configuration --------------------
   Model: llama-7b-demo
   Layers: 32
   Tokens: 512
   KV size per layer: 2.00 MB
   Total KV size: 64.00 MB

...

   P2P Transfer Results:
      Hits: 32/32 (100.0%)
      Total time: 124ms
      Throughput: 515.91 MB/s

   SPEEDUP: 2.1x faster with P2P KV cache!
```

### Multi-Node Setup

#### Node A - Producer

```bash
./bin/demo-cuda --role producer --port 9100 --gpu 0
```

Output:
```
Node ID: 12D3KooW...
Addresses:
  /ip4/192.168.1.100/tcp/9100/p2p/12D3KooW...

--- Producer Mode ---
Cached 32 layers, ready to serve
```

#### Node B - Consumer

```bash
./bin/demo-cuda --role consumer --port 9101 --gpu 0 \
    --peer /ip4/192.168.1.100/tcp/9100/p2p/12D3KooW...
```

### WAN/NAT Configuration

For nodes behind NAT, use the `--external-ip` flag:

```bash
# Producer with public IP
./bin/demo-cuda --role producer --port 9100 --external-ip 203.0.113.50

# Consumer connects using public IP
./bin/demo-cuda --role consumer --port 9101 \
    --peer /ip4/203.0.113.50/tcp/9100/p2p/12D3KooW...
```

### Configuration Options

| Flag | Default | Description |
|------|---------|-------------|
| `--role` | `demo` | Node role: `producer`, `consumer`, or `demo` |
| `--port` | `9100` | libp2p listen port |
| `--gpu` | `0` | CUDA device ID |
| `--peer` | - | Peer multiaddr for consumer mode |
| `--external-ip` | - | External IP for NAT traversal |
| `--mock` | `false` | Use mock GPU connector |
| `--cache-size` | `1GB` | Local cache size |
| `--model` | `llama-7b-demo` | Model ID for cache keys |
| `--layers` | `32` | Number of transformer layers |
| `--tokens` | `512` | Number of tokens |
| `--kv-heads` | `8` | Number of KV attention heads |
| `--head-dim` | `128` | Head dimension |

---

## Performance

### Benchmark Results (POC Validation)

| Metric | Value |
|--------|-------|
| **P2P Hit Rate** | 100% (32/32 layers) |
| **Total Data Transferred** | 64 MB |
| **Transfer Time** | 124 ms |
| **Throughput** | 515.91 MB/s |
| **Avg Latency per Layer** | 3.88 ms |
| **Speedup vs Recompute** | 2.1x |

### Operation Latency Breakdown

| Operation | Latency | Notes |
|-----------|---------|-------|
| GPU → Pinned (D2H) | ~1-2 ms | 8 MB tensor, async |
| Serialization | <1 ms | msgpack encoding |
| Network Transfer (LAN) | ~5-10 ms | Depends on bandwidth |
| Deserialization | <1 ms | msgpack decoding |
| Pinned → GPU (H2D) | ~1-2 ms | 8 MB tensor, async |
| **Total P2P** | **~10-20 ms** | End-to-end |
| **Recompute** | **~200-500 ms** | Full forward pass |

### Tested Infrastructure

| Node | Hardware | Architecture | CUDA | Result |
|------|----------|--------------|------|--------|
| GH200 #1 | NVIDIA GH200 480GB | arm64 | 12.4 | ✅ Producer working |
| GH200 #2 | NVIDIA GH200 480GB | arm64 | 12.8 | ✅ Consumer connected |
| RTX 4090 | NVIDIA RTX 4090 24GB | x86_64 | 12.1 | ✅ Consumer connected |
| A10 | NVIDIA A10 24GB | x86_64 | - | ✅ Mock mode working |

---

## API

### Integration with Inference Engines

```go
import (
    "github.com/neurogrid/kv-cache-p2p/pkg/integration"
    "github.com/neurogrid/kv-cache-p2p/pkg/cache"
)

// Create adapter
config := integration.Config{
    P2PPort:      9100,
    CacheSize:    5 * 1024 * 1024 * 1024, // 5GB
    EnableP2P:    true,
    EnableMDNS:   true,
}
adapter, err := integration.NewNeuroGridAdapter(ctx, config)
if err != nil {
    log.Fatal(err)
}
defer adapter.Close()

// In attention forward pass
func AttentionForward(modelID string, layerID int, tokens []int32) (K, V Tensor) {
    // Try cache lookup
    k, v, found := adapter.LookupKV(ctx, modelID, layerID, tokens)
    if found {
        return k, v  // Cache hit - use cached tensors
    }

    // Cache miss - compute KV
    k, v = computeAttention(tokens)

    // Store for future use
    adapter.StoreKV(ctx, modelID, layerID, tokens, k, v)

    return k, v
}
```

### Low-Level API

```go
import (
    "github.com/neurogrid/kv-cache-p2p/pkg/cache"
    "github.com/neurogrid/kv-cache-p2p/pkg/storage"
    "github.com/neurogrid/kv-cache-p2p/pkg/transport"
)

// Create storage and P2P node
localStorage := storage.NewLocalStorage(cacheSize)
p2pNode, _ := transport.NewP2PNode(ctx, transport.Config{
    ListenPort: 9100,
    EnableMDNS: true,
    ExternalIP: "203.0.113.50",  // Optional
}, localStorage)

// Create cache manager
manager := cache.NewManager(cache.ManagerConfig{
    LocalMaxSize: cacheSize,
    EnableP2P:    true,
}, localStorage, p2pNode)

// Store entry
key := cache.CacheKey{
    ModelID:   "llama-7b",
    LayerID:   0,
    TokenHash: hasher.HashPrefix(tokens),
    SeqStart:  0,
    SeqEnd:    512,
}
entry := cache.NewCacheEntry(key, kTensor, vTensor)
manager.Store(ctx, entry)

// Lookup (local + P2P)
entry, err := manager.Lookup(ctx, key)
if err == nil {
    // Cache hit
    k, v := entry.K, entry.V
    entry.Unref()
}
```

---

## Contributing

We welcome contributions from the academic and research community!

### Development Setup

```bash
# Clone
git clone https://github.com/neurogrid/kv-cache-p2p.git
cd kv-cache-p2p

# Install development dependencies
go mod download

# Run tests
go test ./...

# Run tests with race detector
go test -race ./...
```

### Code Style

- Follow standard Go conventions (`gofmt`, `golint`)
- Write tests for new functionality
- Update documentation for API changes

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Roadmap

- [ ] Integration with vLLM
- [ ] Integration with llama.cpp
- [ ] FP16 → INT8 quantization for reduced transfer size
- [ ] TLS/noise encryption for P2P communication
- [ ] Distributed hash table (DHT) for large-scale peer discovery
- [ ] Radix attention tree for prefix-based cache optimization
- [ ] Multi-GPU support within single node
- [ ] Kubernetes operator for cloud deployment

---

## License

This project is licensed under the **PolyForm Noncommercial License 1.0.0** with an additional **Academic and Research Use Exception**.

### Summary

| Use Case | Permitted |
|----------|-----------|
| Academic research | ✅ Yes |
| Educational use | ✅ Yes |
| Personal learning | ✅ Yes |
| Non-commercial projects | ✅ Yes |
| Commercial use | ❌ Requires commercial license |

For commercial licensing inquiries, please contact: **leandrobar93@gmail.com**

See [LICENSE](LICENSE) for the full license text.

---

## Acknowledgments

- [libp2p](https://libp2p.io/) - P2P networking stack
- [LMCache](https://github.com/LMCache/LMCache) - Inspiration for KV cache sharing
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) - GPU computing platform

---

## Citation

If you use this project in academic research, please cite:

```bibtex
@software{kvcachep2p2025,
  title = {KV Cache P2P: Distributed KV Cache Layer for LLM Inference},
  author = {Barbosa, Leandro},
  year = {2025},
  institution = {AI Engineering Academy},
  email = {leandrobar93@gmail.com},
  url = {https://github.com/neurogrid/kv-cache-p2p}
}
```

---

<p align="center">
  Made with ❤️ by <a href="mailto:leandrobar93@gmail.com">Leandro Barbosa</a> - AI Engineering Academy
</p>
