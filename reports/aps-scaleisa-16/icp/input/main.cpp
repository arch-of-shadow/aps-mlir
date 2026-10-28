#define main icp_reference_unused
#include "reference.cpp"
#undef main
#include <stdint.h>

static float expected_src[POINTS][3], expected_dst[POINTS][3];
static float expected_transform[3], expected_residuals[ITERS];
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
extern "C" void pcl_validate_outputs(float *src, float *dst,
                                      float *transform, float *residuals) {
  ++validation_calls;
  for (int i = 0; i < POINTS; ++i)
    for (int k = 0; k < 3; ++k) {
      check(src[i*3+k], expected_src[i][k]);
      check(dst[i*3+k], expected_dst[i][k]);
    }
  for (int k = 0; k < 3; ++k) check(transform[k], expected_transform[k]);
  for (int i = 0; i < ITERS; ++i) check(residuals[i], expected_residuals[i]);
}
extern "C" int main() {
  init_clouds(expected_src, expected_dst);
  icp_align(expected_src, expected_dst, expected_transform, expected_residuals);
  int expected = summarize(expected_transform, expected_residuals);
  int actual = pcl_selected_main();
  if (validation_calls != 1 || checked_outputs != POINTS*6+3+ITERS) return 3;
  if (mismatches) return 1;
  if (actual != expected) return 2;
  return 0;
}
