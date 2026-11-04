#include <stdint.h>

// Ordinary software has no CSR operations. MEGG supplies gain/bias when
// this loop is replaced with the instruction described in csr_scale.cadl.
#pragma megg optimize
void scale(uint32_t *src, uint32_t *dst, uint32_t gain, uint32_t bias) {
    for (uint32_t i = 0; i < 4; ++i)
        dst[i] = src[i] * gain + bias;
}

int main(void) {
    uint32_t src[4] = {1, 2, 3, 4};
    uint32_t dst[4] = {0};
    scale(src, dst, 3, 7);
    for (uint32_t i = 0; i < 4; ++i)
        if (dst[i] != src[i] * 3 + 7)
            return 1;
    scale(src, dst, 5, 11);
    for (uint32_t i = 0; i < 4; ++i)
        if (dst[i] != src[i] * 5 + 11)
            return 1;
    return 0;
}
