#define main morphological_reference_unused
#include "reference.cpp"
#undef main
#include <stdint.h>

static float expected_base[HEIGHT][WIDTH], expected_tmp0[HEIGHT][WIDTH];
static float expected_tmp1[HEIGHT][WIDTH], expected_out[HEIGHT][WIDTH];
volatile int test_status, checked_outputs, validation_calls, mismatches;
volatile int first_mismatch = -1;
volatile uint32_t first_actual, first_expected;
extern "C" int pcl_selected_main();

static uint32_t bits(float value) {
  uint32_t result;
  __builtin_memcpy(&result, &value, 4);
  return result;
}

static void check(float actual, float expected) {
  if (bits(actual) != bits(expected)) {
    if (first_mismatch < 0) {
      first_mismatch = checked_outputs;
      first_actual = bits(actual);
      first_expected = bits(expected);
    }
    ++mismatches;
  }
  ++checked_outputs;
}

extern "C" void pcl_validate_outputs(float *base, float *tmp0,
                                      float *tmp1, float *out) {
  ++validation_calls;
  for (int y = 0; y < HEIGHT; ++y)
    for (int x = 0; x < WIDTH; ++x) {
      int i = y * WIDTH + x;
      check(base[i], expected_base[y][x]);
      check(tmp0[i], expected_tmp0[y][x]);
      check(tmp1[i], expected_tmp1[y][x]);
      check(out[i], expected_out[y][x]);
    }
}

extern "C" int main() {
  init_height(expected_base);
  open_close_pipeline(expected_base, expected_tmp0, expected_tmp1, expected_out);
  int expected = summarize(expected_out);
  int actual = pcl_selected_main();
  if (validation_calls != 1 || checked_outputs != HEIGHT * WIDTH * 4) return 3;
  if (mismatches) return 1;
  if (actual != expected) return 2;
  return 0;
}
