/**
 * Float32 Sigmoid Implementation with CPU Verification
 *
 * Sigmoid Formula:
 *   sigmoid(x) = 1 / (1 + exp(-x))
 *
 * Input: 4x4 matrix of random floats
 * Output: 4x4 matrix with sigmoid applied element-wise
 * Output range: (0, 1)
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define ROWS 4
#define COLS 4
#define MATRIX_SIZE (ROWS * COLS)

// Custom instruction for sigmoid accelerator
// opcode=0x0B, funct7=0x2D
static inline uint32_t sigmoid_custom(uint32_t addr_in, uint32_t addr_out) {
    uint32_t rd = 0;
    asm volatile(
        ".insn r 0x0B, 0b111, 0x2D, %0, %1, %2"
        : "=r"(rd) : "r"(addr_in), "r"(addr_out)
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

// CPU reference: sigmoid(x) = 1 / (1 + exp(-x))
static float sigmoid_cpu(float x) {
    return 1.0f / (1.0f + expf(-x));
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
    float input_matrix[MATRIX_SIZE];
    float expected_output[MATRIX_SIZE];
    float actual_output[MATRIX_SIZE];

    uint32_t input_bits[MATRIX_SIZE];
    uint32_t output_bits[MATRIX_SIZE];

    printf("=== SIGMOID_TEST_BEGIN ===\n");
    printf("Sigmoid Formula: sigmoid(x) = 1 / (1 + exp(-x))\n");
    printf("Matrix size: %d x %d\n\n", ROWS, COLS);

    // Generate random input matrix in range [-5.0, 5.0]
    printf("Generating random input matrix...\n");
    for (int i = 0; i < MATRIX_SIZE; i++) {
        input_matrix[i] = random_float(-5.0f, 5.0f);
        input_bits[i] = float_to_bits(input_matrix[i]);
    }

    print_matrix("Input Matrix", input_matrix, ROWS, COLS);
    printf("\n");

    // Compute CPU reference
    for (int i = 0; i < MATRIX_SIZE; i++) {
        expected_output[i] = sigmoid_cpu(input_matrix[i]);
    }
    print_matrix("Expected Output (CPU)", expected_output, ROWS, COLS);
    printf("\n");

    // Call accelerator
    printf("Calling sigmoid accelerator...\n");
    uint32_t status = sigmoid_custom((uint32_t)(uintptr_t)input_bits,
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

        int pass = (rel_e < 0.1f);
        if (pass) pass_count++;

        printf("[%d,%d]   %-12.6f %-12.6f %-12.6f %-10.4f %s\n",
               i / COLS, i % COLS, exp_val, act_val, abs_e, rel_e,
               pass ? "PASS" : "FAIL");
    }

    printf("--------------------------------------------------------------\n\n");

    printf("=== SUMMARY ===\n");
    printf("Total elements:  %d\n", MATRIX_SIZE);
    printf("Passed:          %d\n", pass_count);
    printf("Failed:          %d\n", MATRIX_SIZE - pass_count);
    printf("Avg Abs Error:   %.6f\n", total_abs_err / MATRIX_SIZE);
    printf("Avg Rel Error:   %.4f%%\n", total_rel_err / MATRIX_SIZE);
    printf("Max Abs Error:   %.6f\n", max_abs_err);
    printf("Max Rel Error:   %.4f%%\n", max_rel_err);
    printf("\n=== SIGMOID_TEST_END ===\n");

    return (pass_count == MATRIX_SIZE) ? 0 : 1;
}
