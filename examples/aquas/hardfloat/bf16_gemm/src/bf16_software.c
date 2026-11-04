#include "bf16_software.h"

/* Separate translation unit, no LTO: inputs cannot be constant-folded.
 * Two independent columns and eight unrolled terms, like the CADL kernel. */
__attribute__((noinline))
void bf16_gemm_software(const uint16_t *restrict a, const uint16_t *restrict bt,
                        uint16_t *restrict d) {
    for (unsigned tile = 0; tile < 8; ++tile)
        for (unsigned i = 0; i < 8; ++i)
            for (unsigned j = 0; j < 8; j += 2) {
                uint16_t acc0 = 0, acc1 = 0;
#pragma GCC unroll 8
                for (unsigned k = 0; k < 8; ++k) {
                    uint16_t av = a[i * 64 + tile * 8 + k];
                    acc0 = bf16_add(acc0, bf16_mul(av, bt[j * 64 + tile * 8 + k]));
                    acc1 = bf16_add(acc1, bf16_mul(av, bt[(j + 1) * 64 + tile * 8 + k]));
                }
                d[i * 64 + tile * 8 + j] = bf16_add(acc0, 0);
                d[i * 64 + tile * 8 + j + 1] = bf16_add(acc1, 0);
            }
}
