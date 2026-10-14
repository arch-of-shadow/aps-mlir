#include <stdint.h>
#include <stdio.h>

static uint32_t rng = 0x12345678;
static uint32_t random_word(void) {
  rng ^= rng << 13;
  rng ^= rng >> 17;
  rng ^= rng << 5;
  return rng;
}

#ifdef STORAGE_GEMV
static uint32_t input[20] __attribute__((aligned(128)));
static volatile uint32_t output[4] __attribute__((aligned(128)));
static uint32_t execute(uint32_t a, uint32_t b) {
  uint32_t rd;
  asm volatile(".insn r 0x7b, 7, 0x31, %0, %1, %2"
               : "=r"(rd) : "r"(a), "r"(b) : "memory");
  return rd;
}
#else
static uint32_t execute(uint32_t a, uint32_t b) {
  uint32_t rd;
  asm volatile(".insn r 0x2b, 7, 0, %0, %1, %2"
               : "=r"(rd) : "r"(a), "r"(b) : "memory");
  return rd;
}
#endif

int main(void) {
  static const uint32_t edges[] = {
      0, 1, 0xffffffffu, 0x80000000u, 0x7fffffffu, 0x55555555u,
      0xaaaaaaaau, 0xfffffffeu};
  for (unsigned trial = 0; trial < 64; ++trial) {
#ifdef STORAGE_GEMV
    for (unsigned i = 0; i < 20; ++i)
      input[i] = trial < 8 ? edges[(trial + i) % 8] : random_word();
    for (unsigned i = 0; i < 4; ++i)
      output[i] = 0xdeadbeefu;
    uint32_t rd = execute((uint32_t)(uintptr_t)input,
                          (uint32_t)(uintptr_t)output);
    if (rd != 0) {
      printf("FAIL trial=%u response=%u\n", trial, rd);
      return 1;
    }
    for (unsigned row = 0; row < 4; ++row) {
      uint32_t expected = 0;
      for (unsigned col = 0; col < 4; ++col)
        expected += input[row * 4 + col] * input[16 + col];
      if (output[row] != expected) {
        printf("FAIL trial=%u row=%u expected=%08x got=%08x\n",
               trial, row, expected, output[row]);
        return 1;
      }
    }
#else
    uint32_t a = trial < 8 ? edges[trial] : random_word();
    uint32_t b = trial < 8 ? edges[7 - trial] : random_word();
    uint32_t actual = execute(a, b);
    if (actual != a + b) {
      printf("FAIL trial=%u expected=%08x got=%08x\n", trial, a + b, actual);
      return 1;
    }
#endif
  }
#ifdef STORAGE_GEMV
  printf("PASS gemv trials=64 checked_outputs=256\n");
#else
  printf("PASS hello trials=64 checked_responses=64\n");
#endif
  return 0;
}
