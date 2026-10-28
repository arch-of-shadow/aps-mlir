// Reuse the counter, HTIF output and loop-overhead measurement fixture.
#define main rocket_arithmetic_calibration_main
#include "rocket_calibration.cpp"
#undef main

#define CONVERSION_CHAIN(NAME, PAIR) \
__attribute__((noinline)) static uint32_t NAME() { \
  uint32_t x = 42; float y; \
  uint32_t start = cycles(); \
  asm volatile("li t0,128\n1:\n" PAIR PAIR PAIR PAIR PAIR PAIR PAIR PAIR \
               "addi t0,t0,-1\nbnez t0,1b" \
               : "+r"(x), "=&f"(y) : : "t0", "memory"); \
  return cycles() - start; \
}
CONVERSION_CHAIN(run_moves, "fmv.w.x %1,%0\nfmv.x.w %0,%1\n")
CONVERSION_CHAIN(run_signed_to_float, "fcvt.s.w %1,%0\nfmv.x.w %0,%1\n")
CONVERSION_CHAIN(run_float_to_signed, "fmv.w.x %1,%0\nfcvt.w.s %0,%1,rtz\n")
CONVERSION_CHAIN(run_conversion_pair, "fcvt.s.w %1,%0\nfcvt.w.s %0,%1,rtz\n")
#undef CONVERSION_CHAIN

extern "C" int main() {
  run_empty(); uint32_t empty = run_empty();
  run_moves(); uint32_t moves = run_moves();
  run_signed_to_float(); uint32_t signedToFloat = run_signed_to_float();
  run_float_to_signed(); uint32_t floatToSigned = run_float_to_signed();
  run_conversion_pair(); uint32_t pair = run_conversion_pair();
  print("APS_CVT_empty", empty);
  print("APS_CVT_moves", moves);
  print("APS_CVT_signed_to_float", signedToFloat);
  print("APS_CVT_float_to_signed", floatToSigned);
  print("APS_CVT_pair", pair);
  return 0;
}
