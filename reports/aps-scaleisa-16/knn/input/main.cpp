#define main knn_reference_unused
#include "reference.cpp"
#undef main
#include <stdint.h>

static float expected_points[POINTS][3], expected_queries[QUERIES][3];
static float expected_dist[QUERIES][TOPK];
static int expected_idx[QUERIES][TOPK];
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
extern "C" void pcl_validate_outputs(float *points, float *queries,
                                      float *dist, int *idx) {
  ++validation_calls;
  for (int i = 0; i < POINTS; ++i)
    for (int k = 0; k < 3; ++k)
      check(bits(points[i*3+k]), bits(expected_points[i][k]));
  for (int q = 0; q < QUERIES; ++q) {
    for (int k = 0; k < 3; ++k)
      check(bits(queries[q*3+k]), bits(expected_queries[q][k]));
    for (int k = 0; k < TOPK; ++k) {
      check(bits(dist[q*TOPK+k]), bits(expected_dist[q][k]));
      check(idx[q*TOPK+k], expected_idx[q][k]);
    }
  }
}
extern "C" int main() {
  for (int i = 0; i < POINTS; ++i) {
    expected_points[i][0] = (float)((i*11+3)%101)*0.031f;
    expected_points[i][1] = (float)((i*17+5)%97)*0.029f;
    expected_points[i][2] = (float)((i*19+7)%89)*0.027f;
  }
  for (int i = 0; i < QUERIES; ++i) {
    expected_queries[i][0] = (float)((i*23+9)%101)*0.031f;
    expected_queries[i][1] = (float)((i*29+13)%97)*0.029f;
    expected_queries[i][2] = (float)((i*31+17)%89)*0.027f;
  }
  knn_search(expected_points, expected_queries, expected_dist, expected_idx);
  int expected = summarize(expected_dist, expected_idx);
  int actual = pcl_selected_main();
  if (validation_calls != 1 || checked_outputs != POINTS*3+QUERIES*(3+2*TOPK)) return 3;
  if (mismatches) return 1;
  if (actual != expected) return 2;
  return 0;
}
