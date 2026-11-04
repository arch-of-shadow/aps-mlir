/* Native Rocket FPU baseline: FP32 inputs, accumulation and outputs.
 * Separate translation unit, no LTO; allow the target's native FMA.
 * Same tile layout and two column accumulators as the CADL kernel.
 * Leave K-loop unrolling to the compiler's ordinary optimization choices. */
__attribute__((noinline))
void fp32_gemm_software(const float *restrict a, const float *restrict bt,
                        float *restrict d) {
    for (unsigned tile = 0; tile < 8; ++tile)
        for (unsigned i = 0; i < 8; ++i)
            for (unsigned j = 0; j < 8; j += 2) {
                float acc0 = 0, acc1 = 0;
                for (unsigned k = 0; k < 8; ++k) {
                    float av = a[i * 64 + tile * 8 + k];
                    acc0 += av * bt[j * 64 + tile * 8 + k];
                    acc1 += av * bt[(j + 1) * 64 + tile * 8 + k];
                }
                d[i * 64 + tile * 8 + j] = acc0;
                d[i * 64 + tile * 8 + j + 1] = acc1;
            }
}
