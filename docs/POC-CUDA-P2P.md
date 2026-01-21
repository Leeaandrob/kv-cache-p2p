# POC: KV Cache P2P com CUDA

## Objetivo

Demonstrar comunicação de KV cache entre 2 GPUs remotas usando:

```
GPU A (Node 1)                    GPU B (Node 2)
     │                                 │
     ▼                                 ▼
┌─────────┐                      ┌─────────┐
│  CUDA   │                      │  CUDA   │
│ KV Cache│                      │ KV Cache│
└────┬────┘                      └────┬────┘
     │ cudaMemcpyAsync                │ cudaMemcpyAsync
     ▼                                ▼
┌─────────┐                      ┌─────────┐
│ Pinned  │                      │ Pinned  │
│ Memory  │                      │ Memory  │
└────┬────┘                      └────┬────┘
     │ CGO                            │ CGO
     ▼                                ▼
┌─────────┐                      ┌─────────┐
│   Go    │◄────── libp2p ──────►│   Go    │
│  Node   │        stream        │  Node   │
└─────────┘                      └─────────┘
```

## Stack Técnico

| Camada | Tecnologia | Função |
|--------|------------|--------|
| GPU | CUDA 12.x | KV cache storage, operações de tensor |
| Bridge | CGO | Interface Go ↔ C/CUDA |
| Network | libp2p | P2P communication, discovery |
| Protocol | msgpack | Serialização de mensagens |
| Language | Go 1.21+ | Orquestração, concorrência |

## Estrutura de Arquivos

```
kv-cache-p2p/
├── gpu/
│   ├── cuda/
│   │   ├── kvcache.h           # Header C para CGO
│   │   ├── kvcache_memory.cu   # Alloc, copy, streams
│   │   ├── kvcache_ops.cu      # Append, extract, merge
│   │   ├── kvcache_transfer.cu # P2P GPU transfers (NVLink)
│   │   └── Makefile
│   └── bindings/
│       └── kvcache.go          # CGO bindings
├── pkg/
│   ├── gpu/
│   │   ├── connector.go        # GPUConnector interface
│   │   ├── cuda_connector.go   # CUDA implementation
│   │   └── pool.go             # Pinned memory pool
│   ├── cache/                  # ✅ Implementado
│   ├── protocol/               # ✅ Implementado
│   ├── transport/              # ✅ Implementado
│   └── storage/                # ✅ Implementado
└── cmd/
    └── gpu-demo/               # Demo com CUDA real
```

## Compilação

### Pré-requisitos

```bash
# CUDA Toolkit
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verificar
nvcc --version
nvidia-smi
```

### Build CUDA Kernels

```bash
cd gpu/cuda
make clean && make
# Gera: libkvcache.a
```

### Build Go com CGO

```bash
# CGO flags para linkar com CUDA
export CGO_ENABLED=1
export CGO_CFLAGS="-I${PWD}/gpu/cuda"
export CGO_LDFLAGS="-L${PWD}/gpu/cuda -lkvcache -L/usr/local/cuda/lib64 -lcudart -lstdc++"

go build -o bin/gpu-demo ./cmd/gpu-demo
```

## Cenários de Teste

### Cenário 1: Single Machine, 2 GPUs

```bash
# Terminal 1 - GPU 0
CUDA_VISIBLE_DEVICES=0 ./bin/gpu-demo --role=producer --port=9100 --gpu=0

# Terminal 2 - GPU 1
CUDA_VISIBLE_DEVICES=1 ./bin/gpu-demo --role=consumer --port=9101 --gpu=1 --peer=<NODE1_ADDR>
```

### Cenário 2: 2 Machines via Network

```bash
# Machine A (192.168.1.100)
./bin/gpu-demo --role=producer --port=9100 --gpu=0

# Machine B (192.168.1.101)
./bin/gpu-demo --role=consumer --port=9100 --gpu=0 \
    --peer=/ip4/192.168.1.100/tcp/9100/p2p/<PEER_ID>
```

### Cenário 3: Same GPU, 2 Processes (Testing)

```bash
# Terminal 1
./bin/gpu-demo --role=producer --port=9100

# Terminal 2
./bin/gpu-demo --role=consumer --port=9101
```

## Fluxo de Dados

### Producer (GPU A computa KV)

```
1. Forward pass computa K, V para cada layer
2. K, V estão em GPU memory (device pointer)
3. cudaMemcpyAsync(pinnedHost, deviceKV, D2H)
4. Go recebe []byte do pinned buffer
5. Store no cache local + broadcast para peers
```

### Consumer (GPU B precisa KV)

```
1. Gera hash dos tokens
2. Lookup local → MISS
3. P2P Lookup → Peer A tem!
4. P2P Get → Recebe []byte via libp2p stream
5. Copia para pinned buffer
6. cudaMemcpyAsync(deviceKV, pinnedHost, H2D)
7. K, V disponíveis em GPU B para attention
```

## Métricas Esperadas

| Operação | Latência Esperada |
|----------|-------------------|
| GPU → Pinned (D2H) | ~1-2ms (8MB) |
| Pinned → libp2p serialize | <1ms |
| Network transfer (LAN) | ~5-10ms |
| libp2p → Pinned deserialize | <1ms |
| Pinned → GPU (H2D) | ~1-2ms |
| **Total P2P transfer** | **~10-20ms** |
| **Recompute (baseline)** | **~200-500ms** |
| **Speedup** | **10-25x** |

## Integração com neurogrid-engine

O `KVCacheProvider` interface conecta com o Engine:

```go
// neurogrid-engine/pkg/inference/engine.go
type Engine struct {
    kvProvider kvcache.KVCacheProvider
    // ...
}

func (e *Engine) forwardLayer(ctx context.Context, layer int, hidden []byte) ([]byte, error) {
    // 1. Try P2P cache
    k, v, found := e.kvProvider.LookupKV(ctx, e.modelID, layer, e.tokens)

    if found {
        // Cache hit - copy to GPU and use
        e.gpuConnector.ToDevice(ctx, k, e.kvPtrK[layer], kvConfig)
        e.gpuConnector.ToDevice(ctx, v, e.kvPtrV[layer], kvConfig)
        return e.cudaExecutor.ForwardWithCachedKV(ctx, layer, hidden)
    }

    // 2. Cache miss - compute
    output, newK, newV := e.cudaExecutor.Forward(ctx, layer, hidden)

    // 3. Save to P2P cache (async)
    go e.saveToCache(layer, newK, newV)

    return output, nil
}
```

## Próximos Passos

1. [ ] Compilar CUDA kernels (`make` no `gpu/cuda/`)
2. [ ] Testar CGO bindings isoladamente
3. [ ] Criar `cmd/gpu-demo` com CUDA real
4. [ ] Testar 2 processos na mesma máquina
5. [ ] Testar 2 máquinas via rede
6. [ ] Integrar com neurogrid-engine
7. [ ] Benchmarks completos
