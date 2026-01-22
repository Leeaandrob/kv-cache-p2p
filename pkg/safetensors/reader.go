package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// ReadFile reads a safetensors file from the given path
func ReadFile(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open file: %w", err)
	}
	defer f.Close()

	return Read(f)
}

// Read reads a safetensors file from the given reader
func Read(r io.Reader) (*File, error) {
	// Step 1: Read header size (8 bytes, little-endian uint64)
	var headerSize uint64
	if err := binary.Read(r, binary.LittleEndian, &headerSize); err != nil {
		return nil, fmt.Errorf("read header size: %w", err)
	}

	if headerSize > MaxHeaderSize {
		return nil, ErrHeaderTooLarge
	}

	// Step 2: Read header JSON
	headerBytes := make([]byte, headerSize)
	if _, err := io.ReadFull(r, headerBytes); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}

	// Step 3: Parse header JSON
	// The header is a flat map where keys are tensor names
	// and values are {dtype, shape, data_offsets}
	// __metadata__ is a special key for metadata
	var rawHeader map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &rawHeader); err != nil {
		return nil, fmt.Errorf("parse header: %w", err)
	}

	file := &File{
		Metadata:   make(map[string]string),
		Tensors:    make(map[string]*TensorInfo),
		HeaderSize: int64(headerSize),
	}

	// Calculate total data size from tensor offsets
	var maxOffset int64

	for name, rawValue := range rawHeader {
		if name == "__metadata__" {
			// Parse metadata
			var meta map[string]string
			if err := json.Unmarshal(rawValue, &meta); err != nil {
				// Metadata might be more complex, try to parse as-is
				continue
			}
			file.Metadata = meta
			continue
		}

		// Parse tensor info
		var info headerTensorInfo
		if err := json.Unmarshal(rawValue, &info); err != nil {
			return nil, fmt.Errorf("parse tensor %s: %w", name, err)
		}

		dtype := DType(info.DType)
		if dtype.BytesPerElement() == 0 {
			return nil, fmt.Errorf("tensor %s: %w: %s", name, ErrUnsupportedDType, info.DType)
		}

		tensor := &TensorInfo{
			Name:   name,
			DType:  dtype,
			Shape:  info.Shape,
			Offset: info.DataOffsets[0],
			Size:   info.DataOffsets[1] - info.DataOffsets[0],
		}

		// Validate size matches shape
		expectedSize := tensor.NumElements() * int64(dtype.BytesPerElement())
		if tensor.Size != expectedSize {
			return nil, fmt.Errorf("tensor %s: %w: expected %d bytes, got %d",
				name, ErrDataSizeMismatch, expectedSize, tensor.Size)
		}

		file.Tensors[name] = tensor

		if info.DataOffsets[1] > maxOffset {
			maxOffset = info.DataOffsets[1]
		}
	}

	// Step 4: Read all tensor data
	file.Data = make([]byte, maxOffset)
	if _, err := io.ReadFull(r, file.Data); err != nil {
		return nil, fmt.Errorf("read tensor data: %w", err)
	}

	return file, nil
}

// bytesToFloat32 converts raw bytes to float32 based on dtype
func bytesToFloat32(data []byte, dtype DType) ([]float32, error) {
	switch dtype {
	case F32:
		return bytesToFloat32Native(data), nil
	case F16:
		return fp16BytesToFloat32(data), nil
	case BF16:
		return bf16BytesToFloat32(data), nil
	default:
		return nil, fmt.Errorf("%w: cannot convert %s to float32", ErrUnsupportedDType, dtype)
	}
}

// bytesToFloat32Native converts F32 bytes to float32 slice
func bytesToFloat32Native(data []byte) []float32 {
	n := len(data) / 4
	result := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint32(data[i*4:])
		result[i] = Float32FromBits(bits)
	}
	return result
}

// fp16BytesToFloat32 converts F16 bytes to float32 slice
func fp16BytesToFloat32(data []byte) []float32 {
	n := len(data) / 2
	result := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint16(data[i*2:])
		result[i] = FP16ToFloat32(bits)
	}
	return result
}

// bf16BytesToFloat32 converts BF16 bytes to float32 slice
func bf16BytesToFloat32(data []byte) []float32 {
	n := len(data) / 2
	result := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint16(data[i*2:])
		result[i] = BF16ToFloat32(bits)
	}
	return result
}
