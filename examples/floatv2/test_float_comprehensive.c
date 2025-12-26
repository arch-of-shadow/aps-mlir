/**
 * Comprehensive Float32 Operations Test
 * Tests ALL floating-point operations in a single chained expression:
 * add, sub, mul, div, sqrt, exp, log, fptosi, sitofp, cmpf
 *
 * Formula: result = select(a > b, exp(log(sqrt(a) + b)), (a*b) - (a/b)) + float(int(a * b))
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Custom instruction: opcode=0x0B, funct7=0x2C (0101100)
static inline uint32_t float_comprehensive_custom(uint32_t a_bits, uint32_t b_bits) {
    uint32_t rd = 0;
    asm volatile(
        ".insn r 0x0B, 0b111, 0x2C, %0, %1, %2"
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

/**
 * Compute expected result using standard C math functions
 * This mirrors the CADL implementation exactly:
 *
 * Branch 1: sqrt(a) -> add b -> log -> exp
 * Branch 2: a*b -> a/b -> sub
 * Branch 3: fptosi(a*b) -> sitofp
 * Branch 4: cmp(a > b) -> select
 * Final: select_val + float_back
 */
static float compute_expected(float a, float b) {
    // Branch 1: sqrt -> add -> log -> exp
    float sqrt_a = sqrtf(a);
    float sum = sqrt_a + b;
    float ln_sum = logf(sum);
    float exp_ln = expf(ln_sum);  // exp(log(x)) ~= x

    // Branch 2: mul -> div -> sub
    float prod = a * b;
    float quot = a / b;
    float diff = prod - quot;

    // Branch 3: type conversion
    int32_t int_val = (int32_t)prod;  // fptosi: truncate toward zero
    float float_back = (float)int_val;  // sitofp

    // Branch 4: comparison and select
    int cmp_gt = (a > b) ? 1 : 0;
    float select_val = cmp_gt ? exp_ln : diff;

    // Final result
    float result = select_val + float_back;

    return result;
}

/**
 * Print detailed breakdown of computation
 */
static void print_breakdown(float a, float b) {
    printf("  Breakdown for a=%.2f, b=%.2f:\n", a, b);

    // Branch 1
    float sqrt_a = sqrtf(a);
    float sum = sqrt_a + b;
    float ln_sum = logf(sum);
    float exp_ln = expf(ln_sum);
    printf("    sqrt(a)=%.4f, sum=%.4f, log=%.4f, exp=%.4f\n",
           sqrt_a, sum, ln_sum, exp_ln);

    // Branch 2
    float prod = a * b;
    float quot = a / b;
    float diff = prod - quot;
    printf("    prod=%.4f, quot=%.4f, diff=%.4f\n", prod, quot, diff);

    // Branch 3
    int32_t int_val = (int32_t)prod;
    float float_back = (float)int_val;
    printf("    int_val=%d, float_back=%.4f\n", int_val, float_back);

    // Branch 4
    int cmp_gt = (a > b);
    float select_val = cmp_gt ? exp_ln : diff;
    printf("    cmp(a>b)=%d, select_val=%.4f\n", cmp_gt, select_val);

    // Final
    float result = select_val + float_back;
    printf("    final result=%.4f\n", result);
}

int main(void) {
    // Test cases designed to exercise all code paths:
    // - Some with a > b (uses exp_ln path)
    // - Some with a <= b (uses diff path)
    // - Various magnitudes to test precision
    float test_cases[][2] = {
        // a > b cases (select exp_ln)
        { 4.0f,   2.0f   },  // sqrt(4)=2, sum=4, exp(log(4))=4, result=4+8=12
        { 9.0f,   1.0f   },  // sqrt(9)=3, sum=4, exp(log(4))=4, result=4+9=13
        { 16.0f,  2.0f   },  // sqrt(16)=4, sum=6
        { 25.0f,  3.0f   },  // sqrt(25)=5, sum=8

        // a <= b cases (select diff)
        { 2.0f,   4.0f   },  // diff = 8 - 0.5 = 7.5, result=7.5+8=15.5
        { 1.0f,   3.0f   },  // diff = 3 - 0.333 = 2.667
        { 3.0f,   3.0f   },  // a == b, diff = 9 - 1 = 8

        // Edge cases
        { 1.0f,   1.0f   },  // sqrt=1, all simple values
        { 100.0f, 10.0f  },  // larger values
    };
    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);
    int passed = 0;

    printf("=== Comprehensive Float32 Operations Test ===\n");
    printf("Formula: select(a>b, exp(log(sqrt(a)+b)), a*b-a/b) + float(int(a*b))\n");
    printf("Operations: sqrt, add, log, exp, mul, div, sub, fptosi, sitofp, cmpf, select\n\n");

    for (int i = 0; i < num_tests; i++) {
        float a = test_cases[i][0];
        float b = test_cases[i][1];
        float expected = compute_expected(a, b);

        uint32_t result_bits = float_comprehensive_custom(float_to_bits(a), float_to_bits(b));
        float actual = bits_to_float(result_bits);

        float diff = fabsf(actual - expected);
        float rel_err = (expected != 0.0f) ? fabsf(diff / expected) : diff;

        // Allow 0.1% relative error or 1e-3 absolute error for floating point
        int ok = (rel_err < 0.001f) || (diff < 1e-3f);

        printf("[%d] a=%.1f b=%.1f: got %.4f, expect %.4f (err=%.2e) [%s]\n",
               i, a, b, actual, expected, diff, ok ? "PASS" : "FAIL");

        if (!ok) {
            print_breakdown(a, b);
        }

        if (ok) passed++;
    }

    printf("\n=== Result: %d/%d passed ===\n", passed, num_tests);
    return passed == num_tests ? 0 : 1;
}
