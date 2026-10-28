// Counter, HTIF output and the matching loop-overhead measurement.
#define main rocket_arithmetic_calibration_main
#include "rocket_calibration.cpp"
#undef main

static float sign_result;
static uint32_t compare_result;

__attribute__((noinline)) static uint32_t run_sign() {
  float x = 1.25f;
  uint32_t start = cycles();
#define SIGN "fsgnjn.s %0,%0,%0\n"
  asm volatile("li t0,128\n1:\n" SIGN SIGN SIGN SIGN SIGN SIGN SIGN SIGN
               "addi t0,t0,-1\nbnez t0,1b"
               : "+f"(x) : : "t0", "memory");
#undef SIGN
  uint32_t elapsed = cycles() - start;
  sign_result = x;
  return elapsed;
}

#define CROSSING_CHAIN(NAME, OP) \
__attribute__((noinline)) static uint32_t NAME() { \
  uint32_t x = 0; float y; float limit = 1.0f; \
  uint32_t start = cycles(); \
  asm volatile("li t0,128\n1:\n" OP OP OP OP OP OP OP OP \
               "addi t0,t0,-1\nbnez t0,1b" \
               : "+r"(x), "=&f"(y) : "f"(limit) : "t0", "memory"); \
  uint32_t elapsed = cycles() - start; \
  compare_result = x; \
  return elapsed; \
}
CROSSING_CHAIN(run_moves, "fmv.w.x %1,%0\nfmv.x.w %0,%1\n")
CROSSING_CHAIN(run_compare, "fmv.w.x %1,%0\nflt.s %0,%1,%2\n")
#undef CROSSING_CHAIN

extern "C" int main() {
  run_empty(); uint32_t empty = run_empty();
  run_sign(); uint32_t sign = run_sign();
  run_moves(); uint32_t moves = run_moves();
  run_compare(); uint32_t compare = run_compare();
  print("APS_FP_empty", empty);
  print("APS_FP_sign", sign);
  print("APS_FP_moves", moves);
  print("APS_FP_compare", compare);
  return sign_result == 1.25f && compare_result == 1 ? 0 : 1;
}
