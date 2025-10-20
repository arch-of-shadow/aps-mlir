/**
 * DECA Decompression - Simplified C Reference Implementation
 *
 * This is a simplified version matching deca_decompress.cadl
 * Designed for polygeist MLIR conversion
 *
 * Key features:
 * - 32-element tile (fits in 64-byte burst)
 * - u8 bitmask storage (4 bytes for 32 bits)
 * - Single global scale factor (Q8.8 fixed-point)
 * - Two-stage processing: sparse expansion + decompression
 */

#include <stdint.h>

// ============================================================================
// Core Decompression Function
// ============================================================================

/**
 * Decompress a 32-element tile
 *
 * Memory layout:
 *   base_addr+0:  bitmask (4 bytes, u8[4])
 *   base_addr+4:  sparse values (32 bytes, i8[32])
 *   base_addr+36: decompressed output (64 bytes, i16[32])
 *
 * @param base_addr  Base address of compressed data
 * @param scale      Global scale factor (Q8.8 fixed-point i16)
 * @return           Number of non-zero elements
 */
uint32_t deca_decompress_u1(uint32_t base_addr, int16_t scale) {
    // Local buffers (simulating static arrays in CADL)
    uint8_t bitmask[4];
    int8_t values[32];
    int8_t dense_values[32];
    int16_t decompressed_weights[32];

    // Cast base_addr to pointer for memory access
    uint8_t* mem_u8 = (uint8_t*)base_addr;
    int16_t* out_i16 = (int16_t*)(base_addr + 36);

    // ========================================================================
    // Burst Read: Load compressed data
    // ========================================================================

    // Read bitmask (4 bytes)
    for (int i = 0; i < 4; i++) {
        bitmask[i] = mem_u8[i];
    }

    // Read sparse values (32 bytes)
    for (int i = 0; i < 32; i++) {
        values[i] = (int8_t)mem_u8[4 + i];
    }

    // ========================================================================
    // Stage 1: Sparse-to-Dense Expansion (bitmask-driven)
    // ========================================================================

    uint32_t vidx = 0;  // Sparse value index

    for (uint32_t idx = 0; idx < 32; idx++) {
        // Extract bit from bitmask
        uint32_t byte_idx = idx / 8;           // Which byte (0..3)
        uint8_t bit_pos = idx & 0x7;           // Bit position within byte (0..7)
        uint8_t mask_byte = bitmask[byte_idx]; // Read the byte
        uint8_t bit_shifted = mask_byte >> bit_pos;
        uint8_t is_nonzero = bit_shifted & 0x1; // Extract bit 0

        // Conditionally read from sparse values
        int8_t sparse_val = is_nonzero ? values[vidx] : 0;

        // Write to dense array
        dense_values[idx] = sparse_val;

        // Update sparse index counter
        vidx += is_nonzero ? 1 : 0;
    }

    uint32_t nnz = vidx;  // Number of non-zero elements

    // ========================================================================
    // Stage 2: Decompression with Scale (Q8.8 multiplication)
    // ========================================================================

    for (uint32_t idx = 0; idx < 32; idx++) {
        // Read from dense values
        int8_t val = dense_values[idx];

        // Q8.8 multiplication with global scale
        int32_t mul_result = (int32_t)val * (int32_t)scale;
        int16_t dequant = (int16_t)(mul_result >> 8);

        // Write result
        decompressed_weights[idx] = dequant;
    }

    // ========================================================================
    // Burst Write: Store decompressed output
    // ========================================================================

    for (int i = 0; i < 32; i++) {
        out_i16[i] = decompressed_weights[i];
    }

    // Return number of non-zero elements
    return nnz;
}
