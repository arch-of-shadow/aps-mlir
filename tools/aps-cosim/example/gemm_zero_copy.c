#include <stdint.h>
#include <stdio.h>

static int16_t tile_a[16] __attribute__((aligned(128)));

struct GemmWorkspace {
  int16_t tile_b[16];
  int16_t output[16];
} __attribute__((aligned(128)));

static struct GemmWorkspace workspace;

static uint32_t gemm_4x4_custom(uint32_t rs1, uint32_t rs2) {
  uint32_t rd = 0;
  asm volatile(".insn r 0x2B, 0b111, 0x38, %0, %1, %2"
               : "=r"(rd)
               : "r"(rs1), "r"(rs2)
               : "memory");
  return rd;
}

int main(void) {
  static const int16_t expected[16] = {
      0x0800, 0x0800, 0x0800, 0x0000,
      0x1800, 0x1800, 0x1800, 0x0000,
      0x0800, 0x0800, 0x0800, 0x0000,
      0x3800, 0x3800, 0x3800, 0x0000,
  };

  uint8_t *a_bytes = (uint8_t *)tile_a;
  uint8_t *b_bytes = (uint8_t *)workspace.tile_b;
  for (unsigned i = 0; i < 32; ++i) {
    a_bytes[i] = i + 1;
    b_bytes[i] = i + 33;
  }
  for (unsigned i = 0; i < 16; ++i)
    workspace.output[i] = 0;

  gemm_4x4_custom((uint32_t)(uintptr_t)tile_a,
                  (uint32_t)(uintptr_t)&workspace);

  for (unsigned i = 0; i < 16; ++i) {
    if (workspace.output[i] != expected[i]) {
      printf("APS GEMM EXAMPLE FAIL index=%u expected=%d got=%d\n", i,
             expected[i], workspace.output[i]);
      return 1;
    }
  }

  printf("APS GEMM EXAMPLE PASS\n");
  return 0;
}
