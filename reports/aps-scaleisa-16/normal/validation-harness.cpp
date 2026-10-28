#define main normal_reference_unused
#include "reference.cpp"
#undef main
#include <stdint.h>

static float expected_points[HEIGHT][WIDTH][3];
static float expected_normals[HEIGHT][WIDTH][3];
static float expected_smoothed[HEIGHT][WIDTH][3];
static float expected_curvature[HEIGHT][WIDTH];
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

extern "C" void pcl_validate_outputs(float *points, float *normals,
                                      float *smoothed, float *curvature) {
  ++validation_calls;
  for (int y = 0; y < HEIGHT; ++y)
    for (int x = 0; x < WIDTH; ++x) {
      for (int k = 0; k < 3; ++k) {
        int i = (y * WIDTH + x) * 3 + k;
        check(points[i], expected_points[y][x][k]);
        check(normals[i], expected_normals[y][x][k]);
        check(smoothed[i], expected_smoothed[y][x][k]);
      }
      check(curvature[y * WIDTH + x], expected_curvature[y][x]);
    }
}

extern "C" int main() {
  for (int y = 0; y < HEIGHT; ++y)
    for (int x = 0; x < WIDTH; ++x) {
      float fx = (float)x * 0.05f;
      float fy = (float)y * 0.04f;
      expected_points[y][x][0] = fx;
      expected_points[y][x][1] = fy;
      expected_points[y][x][2] =
          0.15f * fx + 0.07f * fy + 0.01f * (float)((x * y) % 9);
    }
  compute_normals(expected_points, expected_normals, expected_curvature);
  smooth_normals(expected_normals, expected_smoothed);
  int expected = accumulate_signature(expected_smoothed, expected_curvature);
  int actual = pcl_selected_main();
  if (validation_calls != 1 || checked_outputs != HEIGHT * WIDTH * 10) return 3;
  if (mismatches) return 1;
  if (actual != expected) return 2;
  return 0;
}
