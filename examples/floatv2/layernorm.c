/**
 * Float32 LayerNorm Implementation with CPU Verification
 *
 * LayerNorm Formula:
 *   y = (x - mean) / sqrt(var + epsilon)
 *   mean = (1/n) * sum(x_i)
 *   var = (1/n) * sum((x_i - mean)^2)
 *
 * For a matrix, LayerNorm is computed row-wise.
 * Input: M x N matrix of random floats
 * Output: M x N matrix where each row has mean=0 and std=1
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Matrix dimensions
#define ROWS 4
#define COLS 4
#define MATRIX_SIZE (ROWS * COLS)

// Epsilon for numerical stability
#define EPSILON 1e-5f

// Custom instruction for layernorm accelerator
// opcode=0x0B, funct7=0x2C
static inline uint32_t layernorm_custom(uint32_t addr_in, uint32_t addr_out) {
    uint32_t rd = 0;
    asm volatile(
        ".insn r 0x0B, 0b111, 0x2B, %0, %1, %2"
        : "=r"(rd) : "r"(addr_in), "r"(addr_out)
    );
    asm volatile("fence");
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

// Simple LCG random number generator
static uint32_t rand_state = 42;

static uint32_t simple_rand(void) {
    rand_state = rand_state * 1103515245 + 12345;
    return rand_state;
}

static float random_float(float min, float max) {
    uint32_t r = simple_rand();
    float normalized = (float)(r & 0xFFFFFF) / (float)0xFFFFFF;
    return min + normalized * (max - min);
}

// CPU reference: LayerNorm for a single row
static void layernorm_row_cpu(const float* input, float* output, int n) {
    // Step 1: Compute mean
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += input[i];
    }
    float mean = sum / n;

    // Step 2: Compute variance
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = input[i] - mean;
        sum_sq += diff * diff;
    }
    float variance = sum_sq / n;

    // Step 3: Normalize
    float std_inv = 1.0f / sqrtf(variance + EPSILON);
    for (int i = 0; i < n; i++) {
        output[i] = (input[i] - mean) * std_inv;
    }
}

// CPU reference: LayerNorm for entire matrix (row-wise)
static void layernorm_matrix_cpu(const float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        layernorm_row_cpu(&input[i * cols], &output[i * cols], cols);
    }
}

static float abs_error(float expected, float actual) {
    float diff = expected - actual;
    return diff < 0 ? -diff : diff;
}

static float rel_error(float expected, float actual) {
    if (expected == 0.0f) {
        return actual == 0.0f ? 0.0f : 100.0f;
    }
    float diff = expected - actual;
    if (diff < 0) diff = -diff;
    if (expected < 0) expected = -expected;
    return (diff / expected) * 100.0f;
}

static void print_matrix(const char* name, const float* mat, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        printf("  [");
        for (int j = 0; j < cols; j++) {
            printf("%8.4f", mat[i * cols + j]);
            if (j < cols - 1) printf(", ");
        }
        printf("]\n");
    }
}

int main(void) {
    // Input and output matrices
    float input_matrix[MATRIX_SIZE] __attribute__((aligned(128)));
    float expected_output[MATRIX_SIZE] __attribute__((aligned(128)));
    float actual_output[MATRIX_SIZE] __attribute__((aligned(128)));

    // For accelerator: bit representation
    uint32_t input_bits[MATRIX_SIZE] __attribute__((aligned(128)));
    uint32_t output_bits[MATRIX_SIZE] __attribute__((aligned(128)));

    // Generate random input matrix
    for (int i = 0; i < MATRIX_SIZE; i++) {
        input_matrix[i] = random_float(-2.0f, 2.0f);
        input_bits[i] = float_to_bits(input_matrix[i]);
    }

    // Compute CPU reference
    layernorm_matrix_cpu(input_matrix, expected_output, ROWS, COLS);
    print_matrix("Expected Output (CPU)", expected_output, ROWS, COLS);

    // Call accelerator
    printf("Calling layernorm accelerator...\n");
    (void)layernorm_custom((uint32_t)(uintptr_t)input_bits,
                           (uint32_t)(uintptr_t)output_bits);

    // Convert results back to float
    for (int i = 0; i < MATRIX_SIZE; i++) {
        actual_output[i] = bits_to_float(output_bits[i]);
    }

    print_matrix("Actual Output (Accelerator)", actual_output, ROWS, COLS);
    printf("\n");

    // Compare results
    printf("=== ELEMENT-WISE COMPARISON ===\n");
    printf("%-8s %-12s %-12s %-12s %-10s\n",
           "Index", "Expected", "Actual", "AbsErr", "RelErr%");
    printf("--------------------------------------------------------------\n");

    float total_abs_err = 0.0f;
    float total_rel_err = 0.0f;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int pass_count = 0;

    for (int i = 0; i < MATRIX_SIZE; i++) {
        float exp_val = expected_output[i];
        float act_val = actual_output[i];
        float abs_e = abs_error(exp_val, act_val);
        float rel_e = rel_error(exp_val, act_val);

        total_abs_err += abs_e;
        total_rel_err += rel_e;
        if (abs_e > max_abs_err) max_abs_err = abs_e;
        if (rel_e > max_rel_err) max_rel_err = rel_e;

        int pass = (rel_e < 1.0f);
        if (pass) pass_count++;

        printf("[%d,%d]   %-12.6f %-12.6f %-12.6f %-10.4f %s\n",
               i / COLS, i % COLS, exp_val, act_val, abs_e, rel_e,
               pass ? "PASS" : "FAIL");
    }

    printf("--------------------------------------------------------------\n\n");

    // Summary
    printf("=== SUMMARY ===\n");
    printf("Total elements:  %d\n", MATRIX_SIZE);
    printf("Passed:          %d\n", pass_count);
    printf("Failed:          %d\n", MATRIX_SIZE - pass_count);
    printf("Avg Abs Error:   %.6f\n", total_abs_err / MATRIX_SIZE);
    printf("Avg Rel Error:   %.4f%%\n", total_rel_err / MATRIX_SIZE);
    printf("Max Abs Error:   %.6f\n", max_abs_err);
    printf("Max Rel Error:   %.4f%%\n", max_rel_err);

    printf("\n=== LAYERNORM_TEST_END ===\n");

    return (pass_count == MATRIX_SIZE) ? 0 : 1;
}
