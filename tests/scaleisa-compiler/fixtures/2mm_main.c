static float a[64][64], b[64][64], c[64][64], d[64][64], out[64][64];
#ifndef TEST_ROWS
#define TEST_ROWS 64
#endif
#ifndef TEST_COLS
#define TEST_COLS 64
#endif
volatile int test_status;
extern void two_stage_mm(float *, float *, float *, float *, float *);
int main(void) {
  for (unsigned i = 0; i < 64; ++i)
    for (unsigned j = 0; j < 64; ++j)
      a[i][j] = b[i][j] = c[i][j] = d[i][j] = 1.0f;
  two_stage_mm(&a[0][0], &b[0][0], &c[0][0], &d[0][0], &out[0][0]);
  for (unsigned i = 0; i < TEST_ROWS; ++i)
    for (unsigned j = 0; j < TEST_COLS; ++j)
      if (out[i][j] != 4097.0f) return 1;
  return 0;
}
