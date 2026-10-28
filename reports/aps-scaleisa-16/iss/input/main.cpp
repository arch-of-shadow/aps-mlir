#define main iss_reference_unused
#include "iss_reference.cpp"
#undef main
#include <stdint.h>

static float expected_points[HEIGHT][WIDTH][3];
static float expected_response[HEIGHT][WIDTH];
static int expected_keypoints[HEIGHT][WIDTH];
volatile int test_status, checked_outputs, validation_calls, mismatches;
volatile int first_mismatch = -1;
volatile uint32_t first_actual, first_expected;
extern "C" int pcl_selected_main();

static uint32_t bits(float value) {
  uint32_t result;
  __builtin_memcpy(&result, &value, 4);
  return result;
}

static void check(uint32_t actual, uint32_t expected) {
  if (actual != expected) {
    if (first_mismatch < 0) {
      first_mismatch = checked_outputs;
      first_actual = actual;
      first_expected = expected;
    }
    ++mismatches;
  }
  ++checked_outputs;
}

extern "C" void pcl_validate_outputs(float *points, float *response,
                                       int *keypoints) {
 ++validation_calls;
  for (int i = 0; i < HEIGHT * WIDTH * 3; ++i)
    check(bits(points[i]), bits(expected_points[i / (WIDTH * 3)][(i / 3) % WIDTH][i % 3]));
  for (int i = 0; i < HEIGHT * WIDTH; ++i) {
    check(bits(response[i]), bits(expected_response[i / WIDTH][i % WIDTH]));
    check(keypoints[i], expected_keypoints[i / WIDTH][i % WIDTH]);
  }
}

extern "C" int main() {
  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      expected_points[y][x][0] = (float)x * 0.03f;
      expected_points[y][x][1] = (float)y * 0.03f;
      expected_points[y][x][2] = 0.02f * (float)((x * 5 + y * 9 + (x ^ y)) % 19);
    }
  }
  compute_iss_response(expected_points, expected_response);
  nonmax_suppress(expected_response, expected_keypoints);
  int expected = summarize(expected_response, expected_keypoints);
 int actual=pcl_selected_main();
  if (validation_calls != 1 || checked_outputs != HEIGHT * WIDTH * 5) return 3;
  if (mismatches) return 1;
  if (actual != expected) return 2;
  return 0;
}
