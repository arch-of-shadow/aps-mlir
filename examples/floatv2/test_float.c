/**
 * Comprehensive Float32 Operations Test
 * Operation: result = sqrt(a) + (a * b) - (a / b)
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Custom instruction: opcode=0x0B, funct7=0x00
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

static float compute_expected(float a, float b) {
    return sqrtf(a) + (a * b) - (a / b);
}

int main(void) {
    float test_cases[][2] = {
        { 4.0f,   2.0f   },
        { 9.0f,   3.0f   },
        { 1.0f,   1.0f   },
        { 16.0f,  4.0f   },
        { 100.0f, 10.0f  },
    };
    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);
    int passed = 0;

    printf("Float ops test: sqrt(a) + a*b - a/b\n");

    for (int i = 0; i < num_tests; i++) {
        float a = test_cases[i][0];
        float b = test_cases[i][1];
        float expected = compute_expected(a, b);

        uint32_t result_bits = float_ops_custom(float_to_bits(a), float_to_bits(b));
        float actual = bits_to_float(result_bits);

        float diff = fabsf(actual - expected);
        int ok = (diff < 1e-4f);

        printf("[%d] a=%.1f b=%.1f: got %.2f, expect %.2f [%s]\n",
               i, a, b, actual, expected, ok ? "PASS" : "FAIL");
        if (ok) passed++;
    }

    printf("Result: %d/%d passed\n", passed, num_tests);
    return passed == num_tests ? 0 : 1;
}
