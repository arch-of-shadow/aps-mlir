/* Full RV32 Rocket program: tiled DRAM -> DMA -> partitioned ISAX -> DMA -> DRAM. */
#include <stdint.h>
#include <stdio.h>
#include "vectors.h"

#define ROCC_COMMAND(name, opcode, funct)                                     \
    static inline uint32_t name(uint32_t rs1, uint32_t rs2) {                   \
        uint32_t rd;                                                         \
        __asm__ volatile (".insn r " #opcode ", 7, " #funct ", %0, %1, %2"    \
                          : "=r"(rd) : "r"(rs1), "r"(rs2) : "memory");        \
        return rd;                                                           \
    }
ROCC_COMMAND(compute, 0x0b, 0)
ROCC_COMMAND(load_a, 0x7b, 0)
ROCC_COMMAND(load_b, 0x7b, 1)
ROCC_COMMAND(store_c, 0x7b, 2)

/* 128-byte alignment matches the DMA burst and Rocket cache block size.
 * The second batch has an extra 128 bytes of padding between DRAM rows. */
struct dma_buffer {
    uint16_t before[64];
    uint16_t data[8 * 128];
    uint16_t after[64];
};
static volatile struct dma_buffer a_buffer __attribute__((aligned(128)));
static volatile struct dma_buffer b_buffer __attribute__((aligned(128)));
static volatile struct dma_buffer d_buffer __attribute__((aligned(128)));
_Static_assert(sizeof(cases) / sizeof(cases[0]) == 16, "two batches of eight tiles");

static void fence_memory(void) { __asm__ volatile ("fence rw, rw" ::: "memory"); }

static void fill(volatile struct dma_buffer *buffer) {
    for (unsigned i = 0; i < 64; ++i) buffer->before[i] = buffer->after[i] = 0xdead;
    for (unsigned i = 0; i < 8 * 128; ++i) buffer->data[i] = 0xdead;
}

static uint32_t dma_args(unsigned tile, unsigned row_words) {
    /* SPM address is in elements, DRAM pitch is in bytes. */
    return ((tile * 64) << 16) | (row_words * sizeof(uint16_t));
}

static int check_output(unsigned batch, unsigned pass, unsigned row_words) {
    for (unsigned i = 0; i < 64; ++i)
        if (d_buffer.before[i] != 0xdead || d_buffer.after[i] != 0xdead) {
            printf("BF16 GEMM FAIL: DMA guard batch=%u pass=%u\n", batch, pass);
            return 1;
        }
    for (unsigned index = 0; index < 8 * 128; ++index) {
        unsigned row = index / row_words, col = index % row_words;
        unsigned expected = 0xdead;
        if (row < 8 && col < 64) {
            const struct gemm_case *v = &cases[batch * 8 + col / 8];
            expected = (pass ? v->accumulated : v->first)[row * 8 + col % 8];
        }
#ifdef GEMM_FORCE_FAIL
        if (batch == 0 && pass == 0 && index == 0) expected ^= 1;
#endif
        unsigned got = d_buffer.data[index];
        if (got != expected) {
            printf("BF16 GEMM FAIL: batch=%u pass=%u row=%u col=%u "
                   "expected=%04x got=%04x\n", batch, pass, row, col, expected, got);
            return 1;
        }
    }
    return 0;
}

int main(void) {
    printf("BF16 GEMM Rocket/RoCC: 8x8, partition=8, unroll=2/8, tiled DMA\n");
    for (unsigned batch = 0; batch < 2; ++batch) {
        unsigned row_words = batch ? 128 : 64;
        /* Nonzero, distinct A/B/C tile offsets; second batch near the DMA
         * packed-address limit. Full 131072-element arrays remain in RTL. */
        unsigned a_tile = batch ? 1008 : 8;
        unsigned b_tile = batch ? 1016 : 24;
        unsigned c_tile = batch ? 1000 : 40;
        fill(&a_buffer);
        fill(&b_buffer);
        for (unsigned tile = 0; tile < 8; ++tile)
            for (unsigned i = 0; i < 8; ++i)
                for (unsigned k = 0; k < 8; ++k) {
                    const struct gemm_case *v = &cases[batch * 8 + tile];
                    a_buffer.data[i * row_words + tile * 8 + k] = v->a[i * 8 + k];
                    b_buffer.data[i * row_words + tile * 8 + k] = v->b[k * 8 + i];
                }
        fence_memory();
        if (load_a((uint32_t)(uintptr_t)a_buffer.data, dma_args(a_tile, row_words)) ||
            load_b((uint32_t)(uintptr_t)b_buffer.data, dma_args(b_tile, row_words))) {
            printf("BF16 GEMM FAIL: DMA load acknowledgement\n");
            return 1;
        }
        for (unsigned pass = 0; pass < 2; ++pass) {
            /* Second pass adds the first result, separately in batch 0 and
             * in-place (offsetC == offsetD) in batch 1. */
            unsigned d_tile = c_tile + (pass && !batch ? 16 : 0);
            for (unsigned tile = 0; tile < 8; ++tile) {
                uint32_t rs1 = ((a_tile + tile) << 16) | (b_tile + tile) | (pass << 15);
                uint32_t rs2 = ((c_tile + tile) << 16) | (d_tile + tile);
                if (compute(rs1, rs2)) {
                    printf("BF16 GEMM FAIL: compute acknowledgement\n");
                    return 1;
                }
            }
            fill(&d_buffer);
            fence_memory();
            if (store_c((uint32_t)(uintptr_t)d_buffer.data, dma_args(d_tile, row_words))) {
                printf("BF16 GEMM FAIL: DMA store acknowledgement\n");
                return 1;
            }
            fence_memory();
            if (check_output(batch, pass, row_words)) return 1;
            printf("BF16 GEMM batch=%u pass=%u: 8 tiles + DMA guards PASS\n", batch, pass);
        }
    }
    printf("BF16 GEMM PASS tiles=32 elements=2048 DMA=8tiles partition=8 unroll=2/8\n");
    return 0;
}
