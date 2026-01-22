// Package safetensors provides reading and writing of safetensors files.
// Safetensors is a binary format for storing tensors safely (no pickle).
// Reference: https://github.com/huggingface/safetensors
package safetensors

import "errors"

// Errors
var (
	ErrInvalidHeader     = errors.New("invalid safetensors header")
	ErrInvalidTensor     = errors.New("invalid tensor data")
	ErrUnsupportedDType  = errors.New("unsupported data type")
	ErrTensorNotFound    = errors.New("tensor not found")
	ErrHeaderTooLarge    = errors.New("header exceeds maximum size")
	ErrDataSizeMismatch  = errors.New("data size does not match tensor shape")
)

// DType represents the data type of tensor elements
type DType string

const (
	F16  DType = "F16"  // IEEE 754 half-precision float (2 bytes)
	F32  DType = "F32"  // IEEE 754 single-precision float (4 bytes)
	F64  DType = "F64"  // IEEE 754 double-precision float (8 bytes)
	BF16 DType = "BF16" // Brain floating-point (2 bytes)
	I8   DType = "I8"   // Signed 8-bit integer
	I16  DType = "I16"  // Signed 16-bit integer
	I32  DType = "I32"  // Signed 32-bit integer
	I64  DType = "I64"  // Signed 64-bit integer
	U8   DType = "U8"   // Unsigned 8-bit integer
	U16  DType = "U16"  // Unsigned 16-bit integer
	U32  DType = "U32"  // Unsigned 32-bit integer
	U64  DType = "U64"  // Unsigned 64-bit integer
	BOOL DType = "BOOL" // Boolean (1 byte)
)

// BytesPerElement returns the number of bytes per element for a dtype
func (d DType) BytesPerElement() int {
	switch d {
	case F16, BF16, I16, U16:
		return 2
	case F32, I32, U32:
		return 4
	case F64, I64, U64:
		return 8
	case I8, U8, BOOL:
		return 1
	default:
		return 0
	}
}

// TensorInfo describes a tensor stored in a safetensors file
type TensorInfo struct {
	Name   string  // Tensor name (key)
	DType  DType   // Data type
	Shape  []int64 // Tensor dimensions
	Offset int64   // Byte offset in data section
	Size   int64   // Size in bytes
}

// NumElements returns the total number of elements in the tensor
func (t *TensorInfo) NumElements() int64 {
	if len(t.Shape) == 0 {
		return 0
	}
	n := int64(1)
	for _, dim := range t.Shape {
		n *= dim
	}
	return n
}

// File represents a safetensors file
type File struct {
	// Metadata contains key-value pairs from the header
	Metadata map[string]string
	// Tensors maps tensor names to their info
	Tensors map[string]*TensorInfo
	// Data contains the raw tensor data
	Data []byte
	// HeaderSize is the size of the JSON header in bytes
	HeaderSize int64
}

// GetTensor returns the tensor info for a given name
func (f *File) GetTensor(name string) (*TensorInfo, error) {
	t, ok := f.Tensors[name]
	if !ok {
		return nil, ErrTensorNotFound
	}
	return t, nil
}

// GetTensorData returns the raw bytes for a tensor
func (f *File) GetTensorData(name string) ([]byte, error) {
	t, err := f.GetTensor(name)
	if err != nil {
		return nil, err
	}

	end := t.Offset + t.Size
	if end > int64(len(f.Data)) {
		return nil, ErrDataSizeMismatch
	}

	return f.Data[t.Offset:end], nil
}

// GetTensorFloat32 returns tensor data converted to float32 slice
func (f *File) GetTensorFloat32(name string) ([]float32, error) {
	t, err := f.GetTensor(name)
	if err != nil {
		return nil, err
	}

	data, err := f.GetTensorData(name)
	if err != nil {
		return nil, err
	}

	return bytesToFloat32(data, t.DType)
}

// TensorNames returns all tensor names in the file
func (f *File) TensorNames() []string {
	names := make([]string, 0, len(f.Tensors))
	for name := range f.Tensors {
		names = append(names, name)
	}
	return names
}

// headerTensorInfo is the JSON representation in the header
type headerTensorInfo struct {
	DType       string  `json:"dtype"`
	Shape       []int64 `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// header is the JSON header structure
type header struct {
	Metadata map[string]string             `json:"__metadata__,omitempty"`
	Tensors  map[string]headerTensorInfo   `json:"-"`
}

// MaxHeaderSize is the maximum allowed header size (100MB)
const MaxHeaderSize = 100 * 1024 * 1024
