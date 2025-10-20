/**
 * DECA Decompression - Ultra-Simplified for Polygeist
 *
 * This version removes all memory operations and focuses on
 * the core computational pattern for instruction matching.
 *
 * Corresponds to the two-stage pattern in deca_decompress.cadl:
 * 1. Sparse-to-dense expansion (bitmask driven)
 * 2. Dequantization with scale (Q8.8 fixed-point)
 */

#include <stdint.h>

/**
 * Simplified decompression kernel
 *
 * @param bitmask         Pointer to 4-byte bitmask
 * @param sparse_values   Pointer to sparse i8 values (max 32)
 * @param scale           Q8.8 fixed-point scale factor
 * @param output          Pointer to output i16 array (32 elements)
 * @return                Number of non-zero elements processed
 */
uint32_t deca_decompress_simple(
    const uint8_t* bitmask,
    const int8_t* sparse_values,
    int16_t scale,
    int16_t* output)
{
    int8_t dense_values[32];
    uint32_t vidx = 0;

    // ========================================================================
    // Stage 1: Sparse-to-Dense Expansion
    // ========================================================================
    for (uint32_t idx = 0; idx < 32; idx++) {
        // Extract bit from bitmask
        uint32_t byte_idx = idx >> 3;          // idx / 8
        uint8_t bit_pos = idx & 0x7;           // idx % 8
        uint8_t mask_byte = bitmask[byte_idx];
        uint8_t is_nonzero = (mask_byte >> bit_pos) & 0x1;

        // Conditionally read from sparse values
        int8_t val = is_nonzero ? sparse_values[vidx] : 0;
        dense_values[idx] = val;

        // Update sparse index
        vidx += is_nonzero;
    }

    // ========================================================================
    // Stage 2: Dequantization (Q8.8)
    // ========================================================================
    for (uint32_t idx = 0; idx < 32; idx++) {
        int8_t val = dense_values[idx];
        int32_t mul_result = (int32_t)val * (int32_t)scale;
        output[idx] = (int16_t)(mul_result >> 8);
    }

    return vidx;
}


/**
 * Even simpler version - single loop fused
 *
 * This version fuses both stages into one loop for better
 * instruction matching potential.
 */
uint32_t deca_decompress_fused(
    const uint8_t* bitmask,
    const int8_t* sparse_values,
    int16_t scale,
    int16_t* output)
{
    uint32_t vidx = 0;

    for (uint32_t idx = 0; idx < 32; idx++) {
        // Extract bit from bitmask
        uint32_t byte_idx = idx >> 3;
        uint8_t bit_pos = idx & 0x7;
        uint8_t is_nonzero = (bitmask[byte_idx] >> bit_pos) & 0x1;

        // Get value (sparse or zero)
        int8_t val = is_nonzero ? sparse_values[vidx] : 0;

        // Dequantize (Q8.8)
        int32_t mul_result = (int32_t)val * (int32_t)scale;
        output[idx] = (int16_t)(mul_result >> 8);

        // Update sparse index
        vidx += is_nonzero;
    }

    return vidx;
}


/**
 * Core pattern - just the bit extraction logic
 *
 * This isolates the bitmask bit extraction pattern which is
 * a good candidate for custom instruction matching.
 */
uint8_t extract_bit_from_bitmask(const uint8_t* bitmask, uint32_t idx) {
    uint32_t byte_idx = idx >> 3;      // idx / 8
    uint8_t bit_pos = idx & 0x7;       // idx % 8
    return (bitmask[byte_idx] >> bit_pos) & 0x1;
}


/**
 * Core pattern - just the Q8.8 dequantization
 *
 * This isolates the Q8.8 multiplication pattern.
 */
int16_t dequantize_q88(int8_t value, int16_t scale) {
    int32_t mul_result = (int32_t)value * (int32_t)scale;
    return (int16_t)(mul_result >> 8);
}
