/**
 * Test program for DECA C reference implementations
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "deca_decompress_simple.c"

// ============================================================================
// Test utilities
// ============================================================================

void print_hex_array(const char* name, const void* data, size_t size) {
    const uint8_t* bytes = (const uint8_t*)data;
    printf("%s: ", name);
    for (size_t i = 0; i < size; i++) {
        printf("%02X ", bytes[i]);
    }
    printf("\n");
}

void print_i8_array(const char* name, const int8_t* data, size_t size) {
    printf("%s: ", name);
    for (size_t i = 0; i < size; i++) {
        printf("%4d ", data[i]);
        if ((i + 1) % 8 == 0) printf("\n           ");
    }
    printf("\n");
}

void print_i16_array(const char* name, const int16_t* data, size_t size) {
    printf("%s: ", name);
    for (size_t i = 0; i < size; i++) {
        printf("%6d ", data[i]);
        if ((i + 1) % 8 == 0) printf("\n           ");
    }
    printf("\n");
}

int count_bits(const uint8_t* bitmask, size_t num_bits) {
    int count = 0;
    for (size_t i = 0; i < num_bits; i++) {
        uint32_t byte_idx = i >> 3;
        uint8_t bit_pos = i & 0x7;
        if ((bitmask[byte_idx] >> bit_pos) & 0x1) {
            count++;
        }
    }
    return count;
}

// ============================================================================
// Test cases
// ============================================================================

void test_extract_bit() {
    printf("\n========================================\n");
    printf("TEST 1: Bit Extraction\n");
    printf("========================================\n");

    uint8_t bitmask[4] = {0xFF, 0x0F, 0xAA, 0x55};
    print_hex_array("bitmask", bitmask, 4);

    printf("\nBit values:\n");
    for (int i = 0; i < 32; i++) {
        uint8_t bit = extract_bit_from_bitmask(bitmask, i);
        printf("%d", bit);
        if ((i + 1) % 8 == 0) printf(" ");
    }
    printf("\n");

    int total = count_bits(bitmask, 32);
    printf("Total bits set: %d\n", total);
}

void test_dequantize_q88() {
    printf("\n========================================\n");
    printf("TEST 2: Q8.8 Dequantization\n");
    printf("========================================\n");

    struct {
        int8_t value;
        int16_t scale;
        int16_t expected;
    } test_cases[] = {
        {1, 0x0100, 1},      // 1 * 1.0 = 1
        {2, 0x0100, 2},      // 2 * 1.0 = 2
        {1, 0x0200, 2},      // 1 * 2.0 = 2
        {1, 0x0080, 0},      // 1 * 0.5 = 0 (truncated)
        {10, 0x0080, 5},     // 10 * 0.5 = 5
        {-1, 0x0100, -1},    // -1 * 1.0 = -1
        {5, 0x0300, 15},     // 5 * 3.0 = 15
    };

    for (size_t i = 0; i < sizeof(test_cases) / sizeof(test_cases[0]); i++) {
        int16_t result = dequantize_q88(test_cases[i].value, test_cases[i].scale);
        printf("dequantize(%3d, 0x%04X) = %6d (expected %6d) %s\n",
               test_cases[i].value,
               test_cases[i].scale,
               result,
               test_cases[i].expected,
               result == test_cases[i].expected ? "✓" : "✗");
    }
}

void test_simple_decompression() {
    printf("\n========================================\n");
    printf("TEST 3: Simple Decompression\n");
    printf("========================================\n");

    // Test case: First 16 bits set, last 16 bits clear
    uint8_t bitmask[4] = {0xFF, 0xFF, 0x00, 0x00};
    int8_t sparse_values[16] = {1, 2, 3, 4, 5, 6, 7, 8,
                                 9, 10, 11, 12, 13, 14, 15, 16};
    int16_t scale = 0x0100;  // 1.0 in Q8.8
    int16_t output[32];

    print_hex_array("bitmask", bitmask, 4);
    print_i8_array("sparse_values", sparse_values, 16);
    printf("scale: 0x%04X (%.2f)\n", scale, scale / 256.0);

    uint32_t nnz = deca_decompress_simple(bitmask, sparse_values, scale, output);

    printf("\nResults:\n");
    printf("Non-zero count: %u\n", nnz);
    print_i16_array("output", output, 32);

    // Verify
    printf("\nVerification:\n");
    int errors = 0;
    for (int i = 0; i < 16; i++) {
        if (output[i] != sparse_values[i]) {
            printf("  ERROR at index %d: got %d, expected %d\n",
                   i, output[i], sparse_values[i]);
            errors++;
        }
    }
    for (int i = 16; i < 32; i++) {
        if (output[i] != 0) {
            printf("  ERROR at index %d: got %d, expected 0\n", i, output[i]);
            errors++;
        }
    }
    if (errors == 0) {
        printf("  ✓ All values correct!\n");
    }
}

void test_fused_decompression() {
    printf("\n========================================\n");
    printf("TEST 4: Fused Decompression\n");
    printf("========================================\n");

    // Sparse pattern: alternating bits
    uint8_t bitmask[4] = {0xAA, 0xAA, 0xAA, 0xAA};  // 10101010...
    int8_t sparse_values[16] = {10, 20, 30, 40, 50, 60, 70, 80,
                                 90, 100, 110, 120, -10, -20, -30, -40};
    int16_t scale = 0x0200;  // 2.0 in Q8.8
    int16_t output[32];

    print_hex_array("bitmask", bitmask, 4);
    print_i8_array("sparse_values", sparse_values, 16);
    printf("scale: 0x%04X (%.2f)\n", scale, scale / 256.0);

    uint32_t nnz = deca_decompress_fused(bitmask, sparse_values, scale, output);

    printf("\nResults:\n");
    printf("Non-zero count: %u\n", nnz);
    print_i16_array("output", output, 32);

    // Verify pattern
    printf("\nVerification:\n");
    int expected_nnz = count_bits(bitmask, 32);
    printf("  Expected non-zero count: %d, got: %u %s\n",
           expected_nnz, nnz, expected_nnz == nnz ? "✓" : "✗");
}

void test_comparison() {
    printf("\n========================================\n");
    printf("TEST 5: Compare Simple vs Fused\n");
    printf("========================================\n");

    uint8_t bitmask[4] = {0x0F, 0xF0, 0x33, 0xCC};
    int8_t sparse_values[16] = {1, 2, 3, 4, 5, 6, 7, 8,
                                 9, 10, 11, 12, 13, 14, 15, 16};
    int16_t scale = 0x0180;  // 1.5 in Q8.8
    int16_t output_simple[32];
    int16_t output_fused[32];

    uint32_t nnz_simple = deca_decompress_simple(
        bitmask, sparse_values, scale, output_simple);
    uint32_t nnz_fused = deca_decompress_fused(
        bitmask, sparse_values, scale, output_fused);

    printf("Non-zero counts: simple=%u, fused=%u %s\n",
           nnz_simple, nnz_fused,
           nnz_simple == nnz_fused ? "✓" : "✗");

    int differences = 0;
    for (int i = 0; i < 32; i++) {
        if (output_simple[i] != output_fused[i]) {
            printf("  DIFF at index %d: simple=%d, fused=%d\n",
                   i, output_simple[i], output_fused[i]);
            differences++;
        }
    }

    if (differences == 0) {
        printf("✓ Simple and Fused produce identical results!\n");
    } else {
        printf("✗ Found %d differences!\n", differences);
    }
}

void test_edge_cases() {
    printf("\n========================================\n");
    printf("TEST 6: Edge Cases\n");
    printf("========================================\n");

    int16_t output[32];

    // Edge case 1: All zeros
    printf("\n--- Case 1: All zeros ---\n");
    uint8_t bitmask_zeros[4] = {0, 0, 0, 0};
    int8_t sparse_zeros[1] = {0};
    uint32_t nnz1 = deca_decompress_fused(
        bitmask_zeros, sparse_zeros, 0x0100, output);
    printf("Non-zero count: %u (expected 0) %s\n",
           nnz1, nnz1 == 0 ? "✓" : "✗");

    // Edge case 2: All ones
    printf("\n--- Case 2: All ones ---\n");
    uint8_t bitmask_ones[4] = {0xFF, 0xFF, 0xFF, 0xFF};
    int8_t sparse_ones[32];
    for (int i = 0; i < 32; i++) sparse_ones[i] = i + 1;
    uint32_t nnz2 = deca_decompress_fused(
        bitmask_ones, sparse_ones, 0x0100, output);
    printf("Non-zero count: %u (expected 32) %s\n",
           nnz2, nnz2 == 32 ? "✓" : "✗");

    // Edge case 3: Single bit
    printf("\n--- Case 3: Single bit (first) ---\n");
    uint8_t bitmask_single[4] = {0x01, 0, 0, 0};
    int8_t sparse_single[1] = {42};
    uint32_t nnz3 = deca_decompress_fused(
        bitmask_single, sparse_single, 0x0100, output);
    printf("Non-zero count: %u (expected 1) %s\n",
           nnz3, nnz3 == 1 ? "✓" : "✗");
    printf("output[0]: %d (expected 42) %s\n",
           output[0], output[0] == 42 ? "✓" : "✗");

    // Edge case 4: Last bit only
    printf("\n--- Case 4: Single bit (last) ---\n");
    uint8_t bitmask_last[4] = {0, 0, 0, 0x80};
    int8_t sparse_last[1] = {99};
    uint32_t nnz4 = deca_decompress_fused(
        bitmask_last, sparse_last, 0x0100, output);
    printf("Non-zero count: %u (expected 1) %s\n",
           nnz4, nnz4 == 1 ? "✓" : "✗");
    printf("output[31]: %d (expected 99) %s\n",
           output[31], output[31] == 99 ? "✓" : "✗");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("========================================\n");
    printf("DECA C Reference Implementation Tests\n");
    printf("========================================\n");

    test_extract_bit();
    test_dequantize_q88();
    test_simple_decompression();
    test_fused_decompression();
    test_comparison();
    test_edge_cases();

    printf("\n========================================\n");
    printf("All tests completed!\n");
    printf("========================================\n");

    return 0;
}
