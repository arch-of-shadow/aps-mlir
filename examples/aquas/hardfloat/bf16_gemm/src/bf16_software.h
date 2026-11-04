#ifndef BF16_GEMM_SOFTWARE_H
#define BF16_GEMM_SOFTWARE_H
#include <stdint.h>

/* BF16 operands widen exactly to FP32. For separate add/multiply of two
 * BF16 operands, FP32 provides sufficient precision for BF16 RNE results.
 * Canonicalize NaNs to the same encoding as the CADL HardFloat operators. */
static inline float bf16_widen(uint16_t value) {
    union { uint32_t bits; float value; } v = {.bits = (uint32_t)value << 16};
    return v.value;
}
static inline uint16_t bf16_round(float value) {
    union { float value; uint32_t bits; } v = {.value = value};
    if (__builtin_expect((v.bits & 0x7fffffffU) > 0x7f800000U, 0)) return 0x7fc0;
    return (uint16_t)((v.bits + 0x7fffU + ((v.bits >> 16) & 1)) >> 16);
}
static inline uint16_t bf16_add(uint16_t a, uint16_t b) {
    return bf16_round(bf16_widen(a) + bf16_widen(b));
}
static inline uint16_t bf16_mul(uint16_t a, uint16_t b) {
    return bf16_round(bf16_widen(a) * bf16_widen(b));
}

/* Eight tiles in the same DMA row layout as the accelerator; B transposed.
 * alpha=1, beta=0, every multiply and add rounds separately to BF16. */
void bf16_gemm_software(const uint16_t *restrict a, const uint16_t *restrict bt,
                        uint16_t *restrict d);
#endif
