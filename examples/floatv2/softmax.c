/**
 * Float32 Softmax Implementation with CPU Verification
 *
 * Softmax Formula (numerically stable):
 *   softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
 *
 * For a matrix, softmax is computed row-wise.
 * Input: M x N matrix of random floats
 * Output: M x N matrix where each row sums to 1.0
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Matrix dimensions (small matrix for testing)
#define ROWS 4
#define COLS 4
#define MATRIX_SIZE (ROWS * COLS)

// Custom instruction for softmax accelerator
// opcode=0x0B, funct7=0x2C (different from test_float)
static inline uint32_t softmax_custom(uint32_t addr_in, uint32_t addr_out) {
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

// Generate random float in range [min, max]
static float random_float(float min, float max) {
    uint32_t r = simple_rand();
    float normalized = (float)(r & 0xFFFFFF) / (float)0xFFFFFF;
    return min + normalized * (max - min);
}

// CPU reference: numerically stable softmax for a single row
static void softmax_row_cpu(const float* input, float* output, int n) {
    // Step 1: Find max for numerical stability
    float max_val = input[0];
    for (int i = 1; i < n; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    // Step 2: Compute exp(x_i - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    // Step 3: Normalize by sum
    for (int i = 0; i < n; i++) {
        output[i] /= sum;
    }
}

// CPU reference: softmax for entire matrix (row-wise)
static void softmax_matrix_cpu(const float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        softmax_row_cpu(&input[i * cols], &output[i * cols], cols);
    }
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

// Print matrix
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

    // printf("=== SOFTMAX_TEST_BEGIN ===\n");
    // printf("Softmax Formula: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))\n");
    // printf("Matrix size: %d x %d\n\n", ROWS, COLS);

    // Generate random input matrix
    // Use range [-2.0, 2.0] to test with both positive and negative values
    // printf("Generating random input matrix...\n");
    for (int i = 0; i < MATRIX_SIZE; i++) {
        input_matrix[i] = random_float(-2.0f, 2.0f);
        input_bits[i] = float_to_bits(input_matrix[i]);
    }

    // print_matrix("Input Matrix", input_matrix, ROWS, COLS);
    // printf("\n");

    // Compute CPU reference (expected)
    softmax_matrix_cpu(input_matrix, expected_output, ROWS, COLS);
    print_matrix("Expected Output (CPU)", expected_output, ROWS, COLS);

    // // Verify each row sums to 1.0
    // printf("\nRow sum verification (should be 1.0):\n");
    // for (int i = 0; i < ROWS; i++) {
    //     float row_sum = 0.0f;
    //     for (int j = 0; j < COLS; j++) {
    //         row_sum += expected_output[i * COLS + j];
    //     }
    //     printf("  Row %d sum: %.6f\n", i, row_sum);
    // }
    // printf("\n");

    // Call accelerator
    printf("Calling softmax accelerator...\n");
    uint32_t status = softmax_custom((uint32_t)(uintptr_t)input_bits,
                                      (uint32_t)(uintptr_t)output_bits);

    // Convert results back to float
    for (int i = 0; i < MATRIX_SIZE; i++) {
        actual_output[i] = bits_to_float(output_bits[i]);
    }

    print_matrix("Actual Output (Accelerator)", actual_output, ROWS, COLS);
    printf("\n");

    // Compare results element by element
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

        // Pass if relative error < 1% (softmax involves exp, may have larger errors)
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
    printf("\n=== SOFTMAX_TEST_END ===\n");

    return (pass_count == MATRIX_SIZE) ? 0 : 1;
}
