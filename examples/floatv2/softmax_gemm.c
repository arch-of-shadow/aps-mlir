/**
 * Simplified Attention: Softmax + MatMul with Delayed Normalization
 *
 * Standard flow:
 *   Y = softmax(X) @ V
 *   where softmax(X)[i,j] = exp(X[i,j] - max_i) / sum_j(exp(X[i,j] - max_i))
 *
 * Delayed normalization flow (mathematically equivalent):
 *   E = exp(X - max)           // unnormalized exp
 *   Y_unnorm = E @ V           // matrix multiply first
 *   Y = Y_unnorm / sum(E)      // normalize after
 *
 * Mathematical equivalence proof:
 *   Standard:  Y = (E / sum) @ V = (1/sum) * (E @ V)
 *   Delayed:   Y = (E @ V) / sum = (1/sum) * (E @ V)
 *   They are identical because sum is a scalar (per row).
 *
 * Matrix dimensions:
 *   X: 4x4 (input to softmax)
 *   V: 4x2 (value matrix, stored as V^T 2x4 for contiguous row access)
 *   Y: 4x2 (output)
 *
 * Note: V is pre-transposed to V^T before sending to accelerator.
 * This allows row-row dot products instead of row-column access.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define ROWS 4
#define COLS 4
#define V_COLS 2

// Custom instruction for softmax_gemm accelerator
// opcode=0x0B, funct7=0x2E
static inline uint32_t softmax_gemm_custom(uint32_t addr_in, uint32_t addr_out) {
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

// ============================================================================
// CPU Reference: Standard Softmax + MatMul
// ============================================================================

static void softmax_row_cpu(const float* input, float* output, int n) {
    float max_val = input[0];
    for (int i = 1; i < n; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < n; i++) {
        output[i] /= sum;
    }
}

// Standard: Y = softmax(X) @ V
static void standard_softmax_gemm(const float* X, const float* V, float* Y,
                                   int m, int n, int p) {
    // m=4, n=4, p=2: X is 4x4, V is 4x2, Y is 4x2
    float softmax_X[ROWS * COLS];

    // Step 1: Compute softmax row-wise
    for (int i = 0; i < m; i++) {
        softmax_row_cpu(&X[i * n], &softmax_X[i * n], n);
    }

    // Step 2: Matrix multiply softmax(X) @ V
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += softmax_X[i * n + k] * V[k * p + j];
            }
            Y[i * p + j] = sum;
        }
    }
}

// ============================================================================
// CPU Reference: Delayed Normalization (for CADL verification)
// ============================================================================

// Delayed: E = exp(X - max), Y_unnorm = E @ V, Y = Y_unnorm / sum(E)
// static void delayed_softmax_gemm(const float* X, const float* V, float* Y,
//                                   int m, int n, int p) {
//     float exp_X[ROWS * COLS];
//     float sum_exp[ROWS];

//     // Step 1: Compute exp(X - max) and sum (but don't divide yet)
//     for (int i = 0; i < m; i++) {
//         // Find max
//         float max_val = X[i * n];
//         for (int j = 1; j < n; j++) {
//             if (X[i * n + j] > max_val) max_val = X[i * n + j];
//         }

//         // Compute exp and sum
//         sum_exp[i] = 0.0f;
//         for (int j = 0; j < n; j++) {
//             exp_X[i * n + j] = expf(X[i * n + j] - max_val);
//             sum_exp[i] += exp_X[i * n + j];
//         }
//         // Note: NOT dividing by sum yet!
//     }

//     // Step 2: Matrix multiply E @ V (unnormalized)
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < p; j++) {
//             float dot = 0.0f;
//             for (int k = 0; k < n; k++) {
//                 dot += exp_X[i * n + k] * V[k * p + j];
//             }
//             // Step 3: Divide by sum AFTER matmul
//             Y[i * p + j] = dot / sum_exp[i];
//         }
//     }
// }

// ============================================================================
// Error computation
// ============================================================================

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
            printf("%10.6f", mat[i * cols + j]);
            if (j < cols - 1) printf(", ");
        }
        printf("]\n");
    }
}

int main(void) {
    // Input matrices
    float X[ROWS * COLS] __attribute__((aligned(128)));      // 4x4 input to softmax
    float V[COLS * V_COLS] __attribute__((aligned(128)));    // 4x2 value matrix


    // For accelerator: bit representation
    uint32_t input_bits[ROWS * COLS+COLS * V_COLS] __attribute__((aligned(128)));
    uint32_t output_bits[COLS * V_COLS] __attribute__((aligned(128)));

    // Output matrices
    float Y_standard[ROWS * V_COLS] __attribute__((aligned(128)));  // Standard result
    float Y_delayed[ROWS * V_COLS] __attribute__((aligned(128)));   // Delayed normalization result

    // Generate random input matrices
    for (int i = 0; i < ROWS * COLS; i++) {
        X[i] = random_float(-2.0f, 2.0f);
    }
    for (int i = 0; i < COLS * V_COLS; i++) {
        V[i] = random_float(-1.0f, 1.0f);
    }

    // Convert inputs to bit representation for accelerator
    for (int i = 0; i < ROWS * COLS; i++) {
        input_bits[i] = float_to_bits(X[i]);
    }
    // Store V^T (transposed V) into input_bits
    // V is 4x2 (COLS x V_COLS), V^T is 2x4 (V_COLS x COLS)
    // V[k,j] at index k*V_COLS+j -> V^T[j,k] at index j*COLS+k
    for (int j = 0; j < V_COLS; j++) {
        for (int k = 0; k < COLS; k++) {
            input_bits[ROWS * COLS + j * COLS + k] = float_to_bits(V[k * V_COLS + j]);
        }
    }

    // Compute CPU reference
    standard_softmax_gemm(X, V, Y_standard, ROWS, COLS, V_COLS);

    // Call accelerator
    printf("Calling softmax_gemm accelerator...\n");
    uint32_t status = softmax_gemm_custom((uint32_t)(uintptr_t)input_bits, (uint32_t)(uintptr_t)output_bits);

    // Convert accelerator output back to float
    for (int i = 0; i < ROWS * V_COLS; i++) {
        Y_delayed[i] = bits_to_float(output_bits[i]);
    }

    print_matrix("Expected Output (CPU)", Y_standard, ROWS, V_COLS);
    printf("\n");
    print_matrix("Actual Output (Accelerator)", Y_delayed, ROWS, V_COLS);
    printf("\n");

    // Compare CPU vs Accelerator
    printf("=== ELEMENT-WISE COMPARISON ===\n");
    printf("%-8s %-14s %-14s %-12s %-10s\n",
           "Index", "Expected", "Actual", "AbsErr", "RelErr%");
    printf("--------------------------------------------------------------\n");

    float total_abs_err = 0.0f;
    float total_rel_err = 0.0f;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int pass_count = 0;
    int total = ROWS * V_COLS;

    for (int i = 0; i < total; i++) {
        float exp_val = Y_standard[i];
        float act_val = Y_delayed[i];
        float abs_e = abs_error(exp_val, act_val);
        float rel_e = rel_error(exp_val, act_val);

        total_abs_err += abs_e;
        total_rel_err += rel_e;
        if (abs_e > max_abs_err) max_abs_err = abs_e;
        if (rel_e > max_rel_err) max_rel_err = rel_e;

        int pass = (rel_e < 1.0f);
        if (pass) pass_count++;

        printf("[%d,%d]   %-14.6f %-14.6f %-12.6f %-10.4f %s\n",
               i / V_COLS, i % V_COLS, exp_val, act_val, abs_e, rel_e,
               pass ? "PASS" : "FAIL");
    }

    printf("--------------------------------------------------------------\n\n");

    // Summary
    printf("=== SUMMARY ===\n");
    printf("Total elements:  %d\n", total);
    printf("Passed:          %d\n", pass_count);
    printf("Failed:          %d\n", total - pass_count);
    printf("Avg Abs Error:   %.6f\n", total_abs_err / total);
    printf("Avg Rel Error:   %.4f%%\n", total_rel_err / total);
    printf("Max Abs Error:   %.6f\n", max_abs_err);
    printf("Max Rel Error:   %.4f%%\n", max_rel_err);

    printf("\n=== SOFTMAX_GEMM_TEST_END ===\n");

    return (pass_count == total) ? 0 : 1;
}
