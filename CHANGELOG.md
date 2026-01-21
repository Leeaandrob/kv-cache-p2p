# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-21

### Added

#### Core Features
- **P2P Networking**: libp2p-based peer-to-peer communication for KV cache sharing
- **mDNS Discovery**: Automatic peer discovery on local networks
- **NAT Traversal**: External IP announcement support for WAN connectivity
- **CUDA Integration**: CGO bindings for direct GPU memory operations
- **Multi-Architecture Support**: arm64 (GH200) and x86_64 (RTX/A100/H100)

#### GPU Support
- CUDA kernels for KV cache memory operations
  - `kvcache_memory.cu`: Pinned memory allocation, D2H/H2D transfers
  - `kvcache_ops.cu`: KV cache append, extract, merge, clear operations
- Support for compute capabilities: sm_75, sm_80, sm_86, sm_89, sm_90
- Automatic CUDA path detection for different installations (standard vs apt)
- Architecture-specific CGO LDFLAGS (arm64 and amd64)

#### Cache Management
- `CacheKey`: Token hash-based cache key with model/layer identification
- `CacheEntry`: Reference-counted KV tensor storage
- `CacheManager`: Coordinated local + P2P cache lookup
- `TokenHasher`: SHA256-based prefix hashing for cache keys
- LRU eviction policy for local storage

#### Protocol
- msgpack-based wire protocol for efficient serialization
- Message types: Lookup, Get, Put, Ping/Pong
- Request/response pattern with unique request IDs

#### Integration
- `NeuroGridAdapter`: Integration adapter for neurogrid-engine
- `GPUConnector` interface with CUDA and Mock implementations
- `PinnedPool`: Pinned memory pool for efficient GPU transfers

#### Demo Application
- Producer mode: Generate and serve KV cache
- Consumer mode: Fetch KV cache from peers
- Demo mode: Self-contained producer + consumer test
- Configurable model parameters (layers, heads, dimensions)

### Performance Results (POC Validation)

| Metric | Value |
|--------|-------|
| P2P Hit Rate | 100% (32/32 layers) |
| Data Transferred | 64 MB |
| Transfer Time | 124 ms |
| Throughput | 515.91 MB/s |
| Speedup vs Recompute | 2.1x |

### Tested Infrastructure

| Node | Hardware | Architecture | CUDA | Status |
|------|----------|--------------|------|--------|
| GH200 #1 | NVIDIA GH200 480GB | arm64 | 12.4 | ✅ Working |
| GH200 #2 | NVIDIA GH200 480GB | arm64 | 12.8 | ✅ Working |
| RTX 4090 | NVIDIA RTX 4090 24GB | x86_64 | 12.1 | ✅ Working |
| A10 | NVIDIA A10 24GB | x86_64 | - | ✅ Mock mode |

### Fixed

- P2P response handling: Fixed `sendRequest` to read response from same stream instead of waiting on separate channel
- Cross-architecture CGO: Added architecture-specific LDFLAGS for arm64 and amd64
- CUDA path detection: Auto-detect standard vs apt-based CUDA installations
- Peer tracking: Fixed `ConnectPeer` to add peers to internal peer list after connection

### Security

- Source-available license (PolyForm Noncommercial) with academic exception
- No sensitive data in protocol (token hashes only, not actual tokens)

---

## Version History

- **0.1.0** (2026-01-21): Initial release - POC validation complete

---

## Upgrade Guide

### From Pre-release to 0.1.0

This is the initial release. No upgrade path required.

---

## Links

- [Repository](https://github.com/neurogrid/kv-cache-p2p)
- [Documentation](docs/)
- [License](LICENSE)

[Unreleased]: https://github.com/neurogrid/kv-cache-p2p/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/neurogrid/kv-cache-p2p/releases/tag/v0.1.0
