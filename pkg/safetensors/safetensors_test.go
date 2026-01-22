package safetensors

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// TestReadWriteRoundtrip tests that we can write and read back tensors
func TestReadWriteRoundtrip(t *testing.T) {
	// Create temporary directory
	tmpDir, err := os.MkdirTemp("", "safetensors_test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create test data
	testPath := filepath.Join(tmpDir, "test.safetensors")

	// Test tensors
	tensors := map[string][]float32{
		"tensor1": {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
		"tensor2": {0.5, -0.5, 0.25, -0.25},
	}
	shapes := map[string][]int64{
		"tensor1": {2, 3},
		"tensor2": {2, 2},
	}
	metadata := map[string]string{
		"format":  "pt",
		"version": "1.0",
	}

	// Write
	if err := WriteFloat32Tensors(testPath, tensors, shapes, metadata); err != nil {
		t.Fatalf("WriteFloat32Tensors failed: %v", err)
	}

	// Read
	file, err := ReadFile(testPath)
	if err != nil {
		t.Fatalf("ReadFile failed: %v", err)
	}

	// Verify metadata
	if file.Metadata["format"] != "pt" {
		t.Errorf("metadata format = %q, want %q", file.Metadata["format"], "pt")
	}

	// Verify tensors
	for name, expected := range tensors {
		got, err := file.GetTensorFloat32(name)
		if err != nil {
			t.Errorf("GetTensorFloat32(%q) error: %v", name, err)
			continue
		}

		if len(got) != len(expected) {
			t.Errorf("tensor %q: len = %d, want %d", name, len(got), len(expected))
			continue
		}

		for i := range expected {
			if math.Abs(float64(got[i]-expected[i])) > 1e-6 {
				t.Errorf("tensor %q[%d] = %f, want %f", name, i, got[i], expected[i])
			}
		}
	}

	// Verify shapes
	for name, expectedShape := range shapes {
		tensor, err := file.GetTensor(name)
		if err != nil {
			t.Errorf("GetTensor(%q) error: %v", name, err)
			continue
		}

		if len(tensor.Shape) != len(expectedShape) {
			t.Errorf("tensor %q: shape len = %d, want %d", name, len(tensor.Shape), len(expectedShape))
			continue
		}

		for i := range expectedShape {
			if tensor.Shape[i] != expectedShape[i] {
				t.Errorf("tensor %q: shape[%d] = %d, want %d", name, i, tensor.Shape[i], expectedShape[i])
			}
		}
	}
}

// TestFP16Conversion tests IEEE 754 half-precision conversion
func TestFP16Conversion(t *testing.T) {
	testCases := []struct {
		name     string
		f32      float32
		expected float32 // Expected value after round-trip (FP16 has less precision)
	}{
		{"zero", 0.0, 0.0},
		{"one", 1.0, 1.0},
		{"negative_one", -1.0, -1.0},
		{"half", 0.5, 0.5},
		{"small", 0.001, 0.001},
		{"large", 1000.0, 1000.0},
		{"negative", -3.14159, -3.14062}, // FP16 precision limits
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			fp16 := Float32ToFP16(tc.f32)
			result := FP16ToFloat32(fp16)

			diff := math.Abs(float64(result - tc.expected))
			maxDiff := 0.01 // Allow 1% error due to FP16 precision
			if diff > float64(math.Abs(float64(tc.expected)))*maxDiff && diff > 0.001 {
				t.Errorf("roundtrip(%f) = %f, want ~%f (diff=%f)", tc.f32, result, tc.expected, diff)
			}
		})
	}
}

// TestFP16WriteRead tests FP16 tensor write/read
func TestFP16WriteRead(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "safetensors_fp16_test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testPath := filepath.Join(tmpDir, "fp16.safetensors")

	// Test data
	tensors := map[string][]float32{
		"weights": {0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4},
	}
	shapes := map[string][]int64{
		"weights": {2, 4},
	}

	// Write as FP16
	if err := WriteFP16Tensors(testPath, tensors, shapes, nil); err != nil {
		t.Fatalf("WriteFP16Tensors failed: %v", err)
	}

	// Read back
	file, err := ReadFile(testPath)
	if err != nil {
		t.Fatalf("ReadFile failed: %v", err)
	}

	// Verify dtype
	tensor, _ := file.GetTensor("weights")
	if tensor.DType != F16 {
		t.Errorf("dtype = %s, want F16", tensor.DType)
	}

	// Verify data (with FP16 precision tolerance)
	got, err := file.GetTensorFloat32("weights")
	if err != nil {
		t.Fatalf("GetTensorFloat32 failed: %v", err)
	}

	for i, expected := range tensors["weights"] {
		diff := math.Abs(float64(got[i] - expected))
		if diff > 0.001 {
			t.Errorf("weights[%d] = %f, want ~%f (diff=%f)", i, got[i], expected, diff)
		}
	}
}

// TestTensorNotFound tests error handling for missing tensors
func TestTensorNotFound(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "safetensors_notfound_test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testPath := filepath.Join(tmpDir, "test.safetensors")

	// Write minimal file
	tensors := map[string][]float32{"exists": {1.0}}
	shapes := map[string][]int64{"exists": {1}}
	WriteFloat32Tensors(testPath, tensors, shapes, nil)

	file, _ := ReadFile(testPath)

	// Try to get non-existent tensor
	_, err = file.GetTensor("does_not_exist")
	if err != ErrTensorNotFound {
		t.Errorf("expected ErrTensorNotFound, got %v", err)
	}
}

// TestDTypeBytes tests dtype byte size calculation
func TestDTypeBytes(t *testing.T) {
	testCases := []struct {
		dtype DType
		bytes int
	}{
		{F16, 2},
		{BF16, 2},
		{F32, 4},
		{F64, 8},
		{I8, 1},
		{I16, 2},
		{I32, 4},
		{I64, 8},
		{U8, 1},
		{U16, 2},
		{U32, 4},
		{U64, 8},
		{BOOL, 1},
	}

	for _, tc := range testCases {
		t.Run(string(tc.dtype), func(t *testing.T) {
			if got := tc.dtype.BytesPerElement(); got != tc.bytes {
				t.Errorf("BytesPerElement() = %d, want %d", got, tc.bytes)
			}
		})
	}
}

// TestTensorInfo tests TensorInfo methods
func TestTensorInfo(t *testing.T) {
	info := &TensorInfo{
		Name:  "test",
		DType: F32,
		Shape: []int64{2, 3, 4},
	}

	expected := int64(2 * 3 * 4)
	if got := info.NumElements(); got != expected {
		t.Errorf("NumElements() = %d, want %d", got, expected)
	}
}

// TestEmptyShape tests handling of scalar tensors
func TestEmptyShape(t *testing.T) {
	info := &TensorInfo{
		Name:  "scalar",
		DType: F32,
		Shape: []int64{},
	}

	if got := info.NumElements(); got != 0 {
		t.Errorf("NumElements() for empty shape = %d, want 0", got)
	}
}

// TestWriteReadBuffer tests write/read with io.Writer/Reader
func TestWriteReadBuffer(t *testing.T) {
	var buf bytes.Buffer

	// Create test tensor
	tensors := []*TensorData{
		{
			Name:  "data",
			DType: F32,
			Shape: []int64{2, 2},
			Data:  float32ToBytes([]float32{1.0, 2.0, 3.0, 4.0}),
		},
	}

	// Write to buffer
	if err := Write(&buf, tensors, nil); err != nil {
		t.Fatalf("Write failed: %v", err)
	}

	// Read from buffer
	file, err := Read(&buf)
	if err != nil {
		t.Fatalf("Read failed: %v", err)
	}

	// Verify
	got, err := file.GetTensorFloat32("data")
	if err != nil {
		t.Fatalf("GetTensorFloat32 failed: %v", err)
	}

	expected := []float32{1.0, 2.0, 3.0, 4.0}
	for i := range expected {
		if got[i] != expected[i] {
			t.Errorf("data[%d] = %f, want %f", i, got[i], expected[i])
		}
	}
}

// TestKVCacheFormat tests the format we'll use for KV cache
func TestKVCacheFormat(t *testing.T) {
	// This test validates the format we'll use for Python-Go interop
	tmpDir, err := os.MkdirTemp("", "kvcache_format_test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testPath := filepath.Join(tmpDir, "kv_cache.safetensors")

	// Simulate TinyLlama KV cache structure
	// TinyLlama: 22 layers, 4 KV heads, 64 head_dim
	numLayers := 22
	numKVHeads := 4
	seqLen := 128
	headDim := 64

	tensors := make(map[string][]float32)
	shapes := make(map[string][]int64)

	// Create key and value tensors for each layer
	for layer := 0; layer < numLayers; layer++ {
		keyName := layerKeyName(layer)
		valueName := layerValueName(layer)

		// Shape: [batch=1, num_kv_heads, seq_len, head_dim]
		shape := []int64{1, int64(numKVHeads), int64(seqLen), int64(headDim)}
		numElements := 1 * numKVHeads * seqLen * headDim

		// Generate dummy data
		keyData := make([]float32, numElements)
		valueData := make([]float32, numElements)
		for i := range keyData {
			keyData[i] = float32(i) * 0.001
			valueData[i] = float32(i) * -0.001
		}

		tensors[keyName] = keyData
		tensors[valueName] = valueData
		shapes[keyName] = shape
		shapes[valueName] = shape
	}

	// Add metadata
	metadata := map[string]string{
		"num_layers":   "22",
		"num_kv_heads": "4",
		"seq_len":      "128",
		"head_dim":     "64",
		"model":        "TinyLlama-1.1B-Chat-v1.0",
	}

	// Write
	if err := WriteFloat32Tensors(testPath, tensors, shapes, metadata); err != nil {
		t.Fatalf("WriteFloat32Tensors failed: %v", err)
	}

	// Read and verify
	file, err := ReadFile(testPath)
	if err != nil {
		t.Fatalf("ReadFile failed: %v", err)
	}

	// Verify we have all tensors
	expectedTensors := numLayers * 2 // key + value for each layer
	if len(file.Tensors) != expectedTensors {
		t.Errorf("tensor count = %d, want %d", len(file.Tensors), expectedTensors)
	}

	// Verify metadata
	if file.Metadata["model"] != "TinyLlama-1.1B-Chat-v1.0" {
		t.Errorf("metadata model = %q, want %q", file.Metadata["model"], "TinyLlama-1.1B-Chat-v1.0")
	}

	// Verify first layer key shape
	tensor, _ := file.GetTensor("layer_0_key")
	expectedShape := []int64{1, 4, 128, 64}
	for i := range expectedShape {
		if tensor.Shape[i] != expectedShape[i] {
			t.Errorf("layer_0_key shape[%d] = %d, want %d", i, tensor.Shape[i], expectedShape[i])
		}
	}

	t.Logf("Successfully created KV cache with %d tensors", len(file.Tensors))
	t.Logf("File size: %d bytes", file.HeaderSize+int64(len(file.Data)))
}

// layerKeyName returns the tensor name for a layer's key cache
func layerKeyName(layer int) string {
	return "layer_" + strconv.Itoa(layer) + "_key"
}

// layerValueName returns the tensor name for a layer's value cache
func layerValueName(layer int) string {
	return "layer_" + strconv.Itoa(layer) + "_value"
}

// BenchmarkFP16Conversion benchmarks FP16 conversion
func BenchmarkFP16Conversion(b *testing.B) {
	data := make([]float32, 1024*1024)
	for i := range data {
		data[i] = float32(i) * 0.001
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Float32ToFP16Bytes(data)
	}
}

// BenchmarkReadLargeFile benchmarks reading a large safetensors file
func BenchmarkReadLargeFile(b *testing.B) {
	// Create a large test file
	tmpDir, err := os.MkdirTemp("", "safetensors_bench")
	if err != nil {
		b.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testPath := filepath.Join(tmpDir, "large.safetensors")

	// Create large tensor (1M elements)
	data := make([]float32, 1024*1024)
	for i := range data {
		data[i] = float32(i)
	}

	tensors := map[string][]float32{"large": data}
	shapes := map[string][]int64{"large": {1024, 1024}}
	WriteFloat32Tensors(testPath, tensors, shapes, nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = ReadFile(testPath)
	}
}
