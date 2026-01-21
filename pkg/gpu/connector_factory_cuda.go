//go:build cuda
// +build cuda

package gpu

// NewConnector creates a GPU connector.
// With CUDA enabled, this creates a real CUDA connector.
func NewConnector(poolSize int64) (GPUConnector, error) {
	return NewCUDAConnector(poolSize)
}

// IsCUDAEnabled returns true when CUDA support is compiled in.
func IsCUDAEnabled() bool {
	return true
}
