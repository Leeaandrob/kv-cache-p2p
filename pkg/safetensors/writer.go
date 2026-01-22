package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
)

// TensorData represents tensor data to be written
type TensorData struct {
	Name  string
	DType DType
	Shape []int64
	Data  []byte
}

// WriteFile writes tensors to a safetensors file
func WriteFile(path string, tensors []*TensorData, metadata map[string]string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer f.Close()

	return Write(f, tensors, metadata)
}

// Write writes tensors to an io.Writer
func Write(w io.Writer, tensors []*TensorData, metadata map[string]string) error {
	// Sort tensors by name for deterministic output
	sortedTensors := make([]*TensorData, len(tensors))
	copy(sortedTensors, tensors)
	sort.Slice(sortedTensors, func(i, j int) bool {
		return sortedTensors[i].Name < sortedTensors[j].Name
	})

	// Build header
	header := make(map[string]interface{})

	// Add metadata if present
	if len(metadata) > 0 {
		header["__metadata__"] = metadata
	}

	// Calculate offsets and add tensor info to header
	var offset int64
	for _, t := range sortedTensors {
		expectedSize := numElements(t.Shape) * int64(t.DType.BytesPerElement())
		if int64(len(t.Data)) != expectedSize {
			return fmt.Errorf("tensor %s: %w: expected %d bytes, got %d",
				t.Name, ErrDataSizeMismatch, expectedSize, len(t.Data))
		}

		header[t.Name] = map[string]interface{}{
			"dtype":        string(t.DType),
			"shape":        t.Shape,
			"data_offsets": [2]int64{offset, offset + int64(len(t.Data))},
		}
		offset += int64(len(t.Data))
	}

	// Encode header to JSON
	headerBytes, err := json.Marshal(header)
	if err != nil {
		return fmt.Errorf("encode header: %w", err)
	}

	// Write header size (8 bytes, little-endian)
	headerSize := uint64(len(headerBytes))
	if err := binary.Write(w, binary.LittleEndian, headerSize); err != nil {
		return fmt.Errorf("write header size: %w", err)
	}

	// Write header
	if _, err := w.Write(headerBytes); err != nil {
		return fmt.Errorf("write header: %w", err)
	}

	// Write tensor data
	for _, t := range sortedTensors {
		if _, err := w.Write(t.Data); err != nil {
			return fmt.Errorf("write tensor %s: %w", t.Name, err)
		}
	}

	return nil
}

// WriteFloat32Tensors is a convenience function to write float32 tensors
func WriteFloat32Tensors(path string, tensors map[string][]float32, shapes map[string][]int64, metadata map[string]string) error {
	tensorData := make([]*TensorData, 0, len(tensors))

	for name, data := range tensors {
		shape, ok := shapes[name]
		if !ok {
			return fmt.Errorf("missing shape for tensor %s", name)
		}

		// Convert float32 slice to bytes
		bytes := float32ToBytes(data)

		tensorData = append(tensorData, &TensorData{
			Name:  name,
			DType: F32,
			Shape: shape,
			Data:  bytes,
		})
	}

	return WriteFile(path, tensorData, metadata)
}

// float32ToBytes converts a float32 slice to bytes (uses exported Float32ToBytes)
func float32ToBytes(data []float32) []byte {
	return Float32ToBytes(data)
}

// float32ToFP16Bytes converts float32 slice to FP16 bytes (uses exported Float32ToFP16Bytes)
func float32ToFP16Bytes(data []float32) []byte {
	return Float32ToFP16Bytes(data)
}

// numElements calculates total elements from shape
func numElements(shape []int64) int64 {
	if len(shape) == 0 {
		return 0
	}
	n := int64(1)
	for _, dim := range shape {
		n *= dim
	}
	return n
}

// WriteFP16Tensors writes float32 data as FP16 tensors
func WriteFP16Tensors(path string, tensors map[string][]float32, shapes map[string][]int64, metadata map[string]string) error {
	tensorData := make([]*TensorData, 0, len(tensors))

	for name, data := range tensors {
		shape, ok := shapes[name]
		if !ok {
			return fmt.Errorf("missing shape for tensor %s", name)
		}

		// Convert float32 to FP16 bytes
		bytes := float32ToFP16Bytes(data)

		tensorData = append(tensorData, &TensorData{
			Name:  name,
			DType: F16,
			Shape: shape,
			Data:  bytes,
		})
	}

	return WriteFile(path, tensorData, metadata)
}
