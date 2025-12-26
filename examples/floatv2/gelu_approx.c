/**
 * GELU - Gaussian Error Linear Unit (Approximate Version)
 * Formula: gelu(x) ≈ x * sigmoid(1.702 * x)
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define ROWS 4
#define COLS 4
#define MATRIX_SIZE (ROWS * COLS)

#define GELU_APPROX_COEFF 1.702f

static inline uint32_t gelu_approx_custom(uint32_t addr_in, uint32_t addr_out) {
    uint32_t rd = 0;
    asm volatile(".insn r 0x0B, 0b111, 0x2E, %0, %1, %2" : "=r"(rd) : "r"(addr_in), "r"(addr_out));
    asm volatile("fence");
    return rd;
}

static inline uint32_t float_to_bits(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }
static inline float bits_to_float(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }

static uint32_t rand_state = 42;
static float random_float(float min, float max) {
    rand_state = rand_state * 1103515245 + 12345;
    return min + ((float)(rand_state & 0xFFFFFF) / 0xFFFFFF) * (max - min);
}

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float gelu_approx_cpu(float x) { return x * sigmoid(GELU_APPROX_COEFF * x); }

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
    float input[MATRIX_SIZE] __attribute__((aligned(128)));
    float expected[MATRIX_SIZE] __attribute__((aligned(128)));
    float actual[MATRIX_SIZE] __attribute__((aligned(128)));
    uint32_t input_bits[MATRIX_SIZE] __attribute__((aligned(128)));
    uint32_t output_bits[MATRIX_SIZE] __attribute__((aligned(128)));

    for (int i = 0; i < MATRIX_SIZE; i++) {
        input[i] = random_float(-3.0f, 3.0f);
        input_bits[i] = float_to_bits(input[i]);
        expected[i] = gelu_approx_cpu(input[i]);
    }

    print_matrix("Expected (CPU)", expected, ROWS, COLS);

    printf("Calling GELU Approx accelerator...\n");
    (void)gelu_approx_custom((uint32_t)(uintptr_t)input_bits, (uint32_t)(uintptr_t)output_bits);

    for (int i = 0; i < MATRIX_SIZE; i++)
        actual[i] = bits_to_float(output_bits[i]);

    print_matrix("Actual (Accelerator)", actual, ROWS, COLS);

    int pass = 0;
    float max_err = 0;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        float err = fabsf(expected[i] - actual[i]);
        if (err > max_err) max_err = err;
        if (err < 0.01f) pass++;
    }

    printf("\n=== SUMMARY ===\nPassed: %d/%d, Max Error: %.6f\n", pass, MATRIX_SIZE, max_err);
    return (pass == MATRIX_SIZE) ? 0 : 1;
}
