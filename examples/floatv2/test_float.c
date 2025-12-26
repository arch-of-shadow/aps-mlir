/**
 * Float32 Operations Test with CPU Verification
 * Operation: result = sqrt(a) + (a * b) - (a / b)
 *
 * This version compares accelerator results with CPU float operations.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define NUM_TESTS 10

// Custom instruction: opcode=0x0B, funct7=0x2B
static inline uint32_t float_ops_custom(uint32_t a_bits, uint32_t b_bits) {
    uint32_t rd = 0;
    asm volatile(
        ".insn r 0x0B, 0b111, 0x2B, %0, %1, %2"
        : "=r"(rd) : "r"(a_bits), "r"(b_bits)
    );
    return rd;
}

static inline uint32_t float_to_bits(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return bits;
}

static inline float bits_to_float(uint32_t bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// Simple LCG random number generator (no stdlib dependency)
static uint32_t rand_state = 12345;

static uint32_t simple_rand(void) {
    rand_state = rand_state * 1103515245 + 12345;
    return rand_state;
}

// Generate random float in range [min, max]
static float random_float(float min, float max) {
    uint32_t r = simple_rand();
    float normalized = (float)(r & 0xFFFFFF) / (float)0xFFFFFF;  // 0.0 ~ 1.0
    return min + normalized * (max - min);
}

// CPU reference: sqrt(a) + a*b - a/b
static float compute_expected(float a, float b) {
    float sqrt_a = sqrtf(a);
    float mul_ab = a * b;
    float div_ab = a / b;
    return sqrt_a + mul_ab - div_ab;
}

// Compute absolute error
static float abs_error(float expected, float actual) {
    float diff = expected - actual;
    return diff < 0 ? -diff : diff;
}

// Compute relative error (percentage)
static float rel_error(float expected, float actual) {
    if (expected == 0.0f) {
        return actual == 0.0f ? 0.0f : 100.0f;
    }
    float diff = expected - actual;
    if (diff < 0) diff = -diff;
    if (expected < 0) expected = -expected;
    return (diff / expected) * 100.0f;
}

int main(void) {
    float a_vals[NUM_TESTS];
    float b_vals[NUM_TESTS];

    // Generate random test data
    // a: positive values for sqrt (1.0 ~ 100.0)
    // b: non-zero values to avoid division by zero (0.5 ~ 50.0)
    printf("=== FLOAT_TEST_BEGIN ===\n");
    printf("Operation: sqrt(a) + a*b - a/b\n");
    printf("Generating %d random test cases...\n\n", NUM_TESTS);

    for (int i = 0; i < NUM_TESTS; i++) {
        a_vals[i] = random_float(1.0f, 100.0f);
        b_vals[i] = random_float(0.5f, 50.0f);
    }

    float total_abs_err = 0.0f;
    float total_rel_err = 0.0f;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int pass_count = 0;

    printf("%-4s %-12s %-12s %-14s %-14s %-12s %-10s\n",
           "No.", "a", "b", "Expected", "Actual", "AbsErr", "RelErr%");
    printf("--------------------------------------------------------------------------------\n");

    for (int i = 0; i < NUM_TESTS; i++) {
        float a = a_vals[i];
        float b = b_vals[i];

        uint32_t a_bits = float_to_bits(a);
        uint32_t b_bits = float_to_bits(b);

        // Accelerator result
        uint32_t result_bits = float_ops_custom(a_bits, b_bits);
        float actual = bits_to_float(result_bits);

        // CPU reference result
        float expected = compute_expected(a, b);

        // Calculate errors
        float abs_e = abs_error(expected, actual);
        float rel_e = rel_error(expected, actual);

        total_abs_err += abs_e;
        total_rel_err += rel_e;
        if (abs_e > max_abs_err) max_abs_err = abs_e;
        if (rel_e > max_rel_err) max_rel_err = rel_e;

        // Consider pass if relative error < 0.01%
        int pass = (rel_e < 0.01f);
        if (pass) pass_count++;

        printf("%-4d %-12.4f %-12.4f %-14.6f %-14.6f %-12.6f %-10.6f %s\n",
               i, a, b, expected, actual, abs_e, rel_e, pass ? "PASS" : "FAIL");
    }

    printf("--------------------------------------------------------------------------------\n\n");

    // Summary
    printf("=== SUMMARY ===\n");
    printf("Total tests:     %d\n", NUM_TESTS);
    printf("Passed:          %d\n", pass_count);
    printf("Failed:          %d\n", NUM_TESTS - pass_count);
    printf("Avg Abs Error:   %.6f\n", total_abs_err / NUM_TESTS);
    printf("Avg Rel Error:   %.6f%%\n", total_rel_err / NUM_TESTS);
    printf("Max Abs Error:   %.6f\n", max_abs_err);
    printf("Max Rel Error:   %.6f%%\n", max_rel_err);
    printf("\n=== FLOAT_TEST_END ===\n");

    return (pass_count == NUM_TESTS) ? 0 : 1;
}
