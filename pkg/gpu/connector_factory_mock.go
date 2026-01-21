//go:build !cuda
// +build !cuda

package gpu

// NewConnector creates a GPU connector.
// Without CUDA, this creates a mock connector.
func NewConnector(poolSize int64) (GPUConnector, error) {
	return NewMockConnector(), nil
}

// IsCUDAEnabled returns true when CUDA support is compiled in.
func IsCUDAEnabled() bool {
	return false
}
