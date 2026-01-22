# Safetensors Format Specification

## Overview
The safetensors format is used for cross-language tensor exchange between Python and Go in the TinyLLaMA Perplexity Validation system.

## File Structure

```
+------------------------+
| Header Size (8 bytes)  |  <- uint64, little-endian
+------------------------+
| Header JSON (variable) |  <- UTF-8 encoded JSON
+------------------------+
| Tensor Data (variable) |  <- Concatenated binary data
+------------------------+
```

## Header Structure

### Header Size
- First 8 bytes of the file
- Unsigned 64-bit integer, little-endian
- Specifies the length of the JSON header in bytes

### Header JSON
```json
{
  "__metadata__": {
    "compression": "INT4",
    "group_size": "64",
    "original_dtype": "float32"
  },
  "layer_0_key": {
    "dtype": "F32",
    "shape": [1, 4, 128, 64],
    "data_offsets": [0, 131072]
  },
  "layer_0_value": {
    "dtype": "F32",
    "shape": [1, 4, 128, 64],
    "data_offsets": [131072, 262144]
  }
}
```

### Tensor Entry Fields
| Field | Type | Description |
|-------|------|-------------|
| `dtype` | string | Data type: F16, F32, BF16, etc. |
| `shape` | int[] | Tensor dimensions |
| `data_offsets` | [int, int] | Start and end byte offsets in data section |

## Data Types

| DType | Bytes | Description |
|-------|-------|-------------|
| F16 | 2 | IEEE 754 half-precision (binary16) |
| F32 | 4 | IEEE 754 single-precision (binary32) |
| F64 | 8 | IEEE 754 double-precision (binary64) |
| BF16 | 2 | Brain floating-point (bfloat16) |
| I8 | 1 | Signed 8-bit integer |
| I16 | 2 | Signed 16-bit integer |
| I32 | 4 | Signed 32-bit integer |
| I64 | 8 | Signed 64-bit integer |
| U8 | 1 | Unsigned 8-bit integer |
| U16 | 2 | Unsigned 16-bit integer |
| U32 | 4 | Unsigned 32-bit integer |
| U64 | 8 | Unsigned 64-bit integer |
| BOOL | 1 | Boolean |

## KV Cache Tensor Naming Convention

```
layer_{i}_key    # Key cache for layer i
layer_{i}_value  # Value cache for layer i
```

### TinyLLaMA Tensor Shapes
```
Tensor Name          Shape               Elements    Bytes (F32)
-----------------------------------------------------------------
layer_0_key          [1, 4, seq, 64]     4*seq*64    16*seq*64
layer_0_value        [1, 4, seq, 64]     4*seq*64    16*seq*64
layer_1_key          [1, 4, seq, 64]     4*seq*64    16*seq*64
...
layer_21_key         [1, 4, seq, 64]     4*seq*64    16*seq*64
layer_21_value       [1, 4, seq, 64]     4*seq*64    16*seq*64
-----------------------------------------------------------------
Total (22 layers):                       22*2*4*seq*64 elements
```

## Byte Order
- All multi-byte values are **little-endian**
- Applies to header size, tensor data, and all numeric types

## FP16 Binary Format (IEEE 754 Half-Precision)

```
Bit:  15 | 14-10  | 9-0
      S  | EEEEE  | MMMMMMMMMM
      |      |        |
      |      |        +-- Mantissa (10 bits)
      |      +----------- Exponent (5 bits, bias 15)
      +------------------ Sign (0=positive, 1=negative)
```

### Special Values
| Value | Sign | Exponent | Mantissa |
|-------|------|----------|----------|
| +0 | 0 | 00000 | 0000000000 |
| -0 | 1 | 00000 | 0000000000 |
| +Inf | 0 | 11111 | 0000000000 |
| -Inf | 1 | 11111 | 0000000000 |
| NaN | X | 11111 | != 0 |

### Conversion to Float32

```go
func FP16ToFloat32(fp16 uint16) float32 {
    sign := uint32(fp16>>15) & 0x1
    exp := uint32(fp16>>10) & 0x1F
    frac := uint32(fp16) & 0x3FF

    var f32Bits uint32
    if exp == 0 {
        if frac == 0 {
            // Zero
            f32Bits = sign << 31
        } else {
            // Subnormal -> normalize
            exp = 1
            for (frac & 0x400) == 0 {
                frac <<= 1
                exp--
            }
            frac &= 0x3FF
            f32Bits = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
        }
    } else if exp == 31 {
        // Infinity or NaN
        f32Bits = (sign << 31) | (0xFF << 23) | (frac << 13)
    } else {
        // Normal number
        f32Bits = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    }
    return math.Float32frombits(f32Bits)
}
```

## BF16 Binary Format (Brain Floating-Point)

```
Bit:  15 | 14-7   | 6-0
      S  | EEEEEEEE | MMMMMMM
      |       |         |
      |       |         +-- Mantissa (7 bits)
      |       +------------ Exponent (8 bits, bias 127)
      +-------------------- Sign (0=positive, 1=negative)
```

BF16 is simply the top 16 bits of a float32.

### Conversion to Float32
```go
func BF16ToFloat32(bf16 uint16) float32 {
    return math.Float32frombits(uint32(bf16) << 16)
}
```

## Example: Reading a Safetensors File

### Python
```python
from safetensors.torch import load_file

tensors = load_file("kv_cache.safetensors")
# tensors["layer_0_key"].shape -> [1, 4, 128, 64]
```

### Go
```go
import "github.com/neurogrid/kv-cache-p2p/pkg/safetensors"

file, err := safetensors.ReadFile("kv_cache.safetensors")
if err != nil {
    log.Fatal(err)
}

// Get tensor as float32
data, err := file.GetTensorFloat32("layer_0_key")
// data is []float32 with 4*128*64 = 32768 elements
```

## Example: Writing a Safetensors File

### Python
```python
from safetensors.torch import save_file
import torch

tensors = {
    "layer_0_key": torch.randn(1, 4, 128, 64),
    "layer_0_value": torch.randn(1, 4, 128, 64),
}
save_file(tensors, "kv_cache.safetensors")
```

### Go
```go
import "github.com/neurogrid/kv-cache-p2p/pkg/safetensors"

tensors := []*safetensors.TensorData{
    {
        Name:  "layer_0_key",
        DType: safetensors.F32,
        Shape: []int64{1, 4, 128, 64},
        Data:  keyBytes,  // []byte
    },
    {
        Name:  "layer_0_value",
        DType: safetensors.F32,
        Shape: []int64{1, 4, 128, 64},
        Data:  valueBytes,
    },
}

metadata := map[string]string{
    "compression": "INT4",
    "group_size": "64",
}

err := safetensors.WriteFile("kv_cache.safetensors", tensors, metadata)
```

## Validation

### Check File Integrity
```bash
# Python
python -c "from safetensors.torch import load_file; print(load_file('file.safetensors').keys())"

# Quick header check
head -c 8 file.safetensors | xxd -e -g 8  # Shows header size
```

### Verify Tensor Shapes
```python
from safetensors.torch import load_file

tensors = load_file("kv_cache.safetensors")
for name, tensor in tensors.items():
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}")
```

## Security Note
Safetensors was designed as a safe alternative to pickle-based formats (like .pt files). It does not allow arbitrary code execution and only stores tensor data and metadata.

## References
- [Safetensors GitHub](https://github.com/huggingface/safetensors)
- [Safetensors Documentation](https://huggingface.co/docs/safetensors/)
- [IEEE 754 Standard](https://en.wikipedia.org/wiki/IEEE_754)
