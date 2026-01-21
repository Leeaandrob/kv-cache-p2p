package compression

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash/crc32"

	"github.com/pierrec/lz4/v4"
)

// LZ4Compressor handles network-level compression
type LZ4Compressor struct {
	level int
}

// NewLZ4Compressor creates a new LZ4 compressor
func NewLZ4Compressor(level int) *LZ4Compressor {
	if level < 1 {
		level = 1
	}
	if level > 12 {
		level = 12
	}
	return &LZ4Compressor{level: level}
}

// Compress compresses data using LZ4
func (c *LZ4Compressor) Compress(data []byte) ([]byte, error) {
	if len(data) == 0 {
		return nil, ErrInvalidInput
	}

	// Create buffer with header space
	// Header: [magic(4)] [original_size(4)] [checksum(4)]
	var buf bytes.Buffer

	// Write header
	buf.Write([]byte("LZ4C")) // Magic
	binary.Write(&buf, binary.LittleEndian, uint32(len(data)))
	binary.Write(&buf, binary.LittleEndian, crc32.ChecksumIEEE(data))

	// Compress using block mode for simplicity
	compressed := make([]byte, lz4.CompressBlockBound(len(data)))
	n, err := lz4.CompressBlock(data, compressed, nil)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCompressFailed, err)
	}

	buf.Write(compressed[:n])
	return buf.Bytes(), nil
}

// Decompress decompresses LZ4 data
func (c *LZ4Compressor) Decompress(data []byte) ([]byte, error) {
	if len(data) < 12 {
		return nil, ErrInvalidInput
	}

	// Read header
	magic := string(data[:4])
	if magic != "LZ4C" {
		return nil, fmt.Errorf("%w: invalid magic", ErrDecompressFailed)
	}

	originalSize := binary.LittleEndian.Uint32(data[4:8])
	expectedChecksum := binary.LittleEndian.Uint32(data[8:12])

	// Decompress using block mode
	result := make([]byte, originalSize)
	n, err := lz4.UncompressBlock(data[12:], result)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrDecompressFailed, err)
	}

	// Verify checksum
	actualChecksum := crc32.ChecksumIEEE(result[:n])
	if actualChecksum != expectedChecksum {
		return nil, fmt.Errorf("%w: checksum mismatch", ErrDecompressFailed)
	}

	return result[:n], nil
}

// CompressedSize returns the size after compression (for stats)
func (c *LZ4Compressor) CompressedSize(compressed []byte) int {
	return len(compressed)
}

// OriginalSize returns the original size from header
func (c *LZ4Compressor) OriginalSize(compressed []byte) (int, error) {
	if len(compressed) < 12 {
		return 0, ErrInvalidInput
	}
	return int(binary.LittleEndian.Uint32(compressed[4:8])), nil
}

// CompressionRatio returns the compression ratio
func (c *LZ4Compressor) CompressionRatio(original, compressed []byte) float32 {
	if len(compressed) == 0 {
		return 0
	}
	return float32(len(original)) / float32(len(compressed))
}
