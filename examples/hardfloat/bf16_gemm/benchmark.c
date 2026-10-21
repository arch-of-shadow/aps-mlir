/* Both implementations execute on the same RV32F Rocket RTL, in one ELF. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "bf16_software.h"
#include "bench_vectors.h"

/* Consume each returned rd before subsequent counters: a RoCC response may
 * arrive asynchronously even though Rocket issues instructions in order. */
#define ROCC_COMMAND(name, opcode, funct)                                     \
    static inline uint32_t name(uint32_t rs1, uint32_t rs2) {                   \
        uint32_t rd;                                                         \
        __asm__ volatile (".insn r " #opcode ", 7, " #funct ", %0, %1, %2\n" \
                          "addi %0, %0, 0"                                   \
                          : "=r"(rd) : "r"(rs1), "r"(rs2) : "memory");        \
        return rd;                                                           \
    }
ROCC_COMMAND(compute, 0x0b, 0)
ROCC_COMMAND(load_a, 0x7b, 0)
ROCC_COMMAND(load_b, 0x7b, 1)
ROCC_COMMAND(store_c, 0x7b, 2)

static uint16_t a[512] __attribute__((aligned(128)));
static uint16_t bt[512] __attribute__((aligned(128)));
static uint16_t d[512] __attribute__((aligned(128)));
static float fa[512] __attribute__((aligned(128)));
static float fbt[512] __attribute__((aligned(128)));
static float fd[512] __attribute__((aligned(128)));
void fp32_gemm_software(const float *restrict, const float *restrict,
                        float *restrict);

/* Regions are far shorter than 2^32 cycles; unsigned subtraction handles wrap.
 * Fences and memory clobbers keep accesses inside their measured regions. */
static inline uint32_t cycles(void) {
    uint32_t value;
    __asm__ volatile ("fence rw, rw\ncsrr %0, mcycle" : "=r"(value) :: "memory");
    return value;
}

static void prepare(void) {
    memcpy(a, bench_a, sizeof(a));
    memcpy(bt, bench_bt, sizeof(bt));
    memset(d, 0xa5, sizeof(d));
    __asm__ volatile ("fence rw, rw" ::: "memory");
}

static int verify(const char *label, unsigned trial) {
    for (unsigned i = 0; i < 512; ++i)
        if (d[i] != bench_expected[i]) {
            printf("BF16 BENCH FAIL %s trial=%u index=%u expected=%04x got=%04x\n",
                   label, trial, i, bench_expected[i], d[i]);
            return 1;
        }
    return 0;
}

static int run_software(unsigned trial) {
    /* Exact widening and output initialization are outside the timed kernel. */
    for (unsigned i = 0; i < 512; ++i) {
        fa[i] = bf16_widen(bench_a[i]);
        fbt[i] = bf16_widen(bench_bt[i]);
    }
    memset(fd, 0xa5, sizeof(fd));
    uint32_t start = cycles();
    fp32_gemm_software(fa, fbt, fd);
    uint32_t end = cycles();
    for (unsigned i = 0; i < 512; ++i) {
        union { float value; uint32_t bits; } got = {.value = fd[i]};
        if (got.bits != bench_fp32_expected[i]) {
            printf("BF16 BENCH FAIL fp32 trial=%u index=%u expected=%08lx got=%08lx\n",
                   trial, i, (unsigned long)bench_fp32_expected[i], (unsigned long)got.bits);
            return 1;
        }
    }
    printf("BF16_BENCH trial=%u engine=fp32 tiles=8 elements=512 cycles=%lu\n",
           trial, (unsigned long)(end - start));
    return 0;
}

static int run_isax(unsigned trial) {
    prepare();
    unsigned ack = 0;
    const unsigned a_tile = 8, b_tile = 24, d_tile = 40;
    uint32_t start = cycles();
    ack |= load_a((uint32_t)(uintptr_t)a, (a_tile * 64 << 16) | 128);
    ack |= load_b((uint32_t)(uintptr_t)bt, (b_tile * 64 << 16) | 128);
    uint32_t loaded = cycles();
    for (unsigned tile = 0; tile < 8; ++tile)
        ack |= compute(((a_tile + tile) << 16) | (b_tile + tile),
                       ((d_tile + tile) << 16) | (d_tile + tile));
    uint32_t computed = cycles();
    ack |= store_c((uint32_t)(uintptr_t)d, (d_tile * 64 << 16) | 128);
    uint32_t end = cycles();
    if (ack || verify("isax", trial)) {
        printf("BF16 BENCH FAIL RoCC acknowledgement=%u\n", ack);
        return 1;
    }
    printf("BF16_BENCH trial=%u engine=isax tiles=8 elements=512 load=%lu compute=%lu store=%lu total=%lu\n",
           trial, (unsigned long)(loaded - start), (unsigned long)(computed - loaded),
           (unsigned long)(end - computed), (unsigned long)(end - start));
    return 0;
}

int main(void) {
    uint32_t misa;
    __asm__ volatile ("csrr %0, misa" : "=r"(misa));
    if (!(misa & (1U << 5))) {
        puts("BF16 BENCH FAIL: F extension unavailable");
        return 1;
    }
    __asm__ volatile ("csrw fcsr, zero"); /* RNE */
    puts("BF16 BENCH Rocket RV32F: 8 tiles, native FP32 FMA vs BF16 ISAX, O3, cycles=mcycle");
    uint32_t start = cycles(), end = cycles();
    printf("BF16_BENCH counter_pair_cycles=%lu\n", (unsigned long)(end - start));
    /* First invocation and repeat; reverse order to expose ordering/cache effects.
     * Input setup, B transposition, checks and printing are outside all timings. */
    if (run_software(0) || run_isax(0) || run_isax(1) || run_software(1)) return 1;
    puts("BF16 BENCH PASS trials=2 checked=2048");
    return 0;
}
