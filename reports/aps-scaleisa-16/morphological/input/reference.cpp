static const int HEIGHT = 12;
static const int WIDTH = 12;
static const int RADIUS = 2;

static int clamp_i(int x, int lo, int hi) {
  if (x < lo) {
    return lo;
  }
  if (x > hi) {
    return hi;
  }
  return x;
}

static void init_height(float grid[HEIGHT][WIDTH]) {
  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      int ridge = ((x - 31) * (x - 31) + (y - 53) * (y - 53)) % 17;
      grid[y][x] = 0.02f * (float)((x * 7 + y * 13 + ridge) % 51);
    }
  }
}

static void morph_pass(
    float src[HEIGHT][WIDTH],
    float dst[HEIGHT][WIDTH],
    int mode) {
  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      float best = src[y][x];
      for (int ky_idx = 0; ky_idx < 2 * RADIUS + 1; ++ky_idx) {
        int ky = ky_idx - RADIUS;
        for (int kx_idx = 0; kx_idx < 2 * RADIUS + 1; ++kx_idx) {
          int kx = kx_idx - RADIUS;
          int yy = clamp_i(y + ky, 0, HEIGHT - 1);
          int xx = clamp_i(x + kx, 0, WIDTH - 1);
          float value = src[yy][xx];
          if (mode == 0) {
            if (value > best) {
              best = value;
            }
          } else {
            if (value < best) {
              best = value;
            }
          }
        }
      }
      dst[y][x] = best;
    }
  }
}

static void open_close_pipeline(
    float base[HEIGHT][WIDTH],
    float tmp0[HEIGHT][WIDTH],
    float tmp1[HEIGHT][WIDTH],
    float out[HEIGHT][WIDTH]) {
  morph_pass(base, tmp0, 1);
  morph_pass(tmp0, tmp1, 0);
  morph_pass(tmp1, tmp0, 0);
  morph_pass(tmp0, out, 1);
}

static int summarize(float grid[HEIGHT][WIDTH]) {
  int total = 0;
  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      total += (int)(grid[y][x] * 128.0f) + ((x + y) & 3);
    }
  }
  return total;
}

int main() {
  float base[HEIGHT][WIDTH];
  float tmp0[HEIGHT][WIDTH];
  float tmp1[HEIGHT][WIDTH];
  float out[HEIGHT][WIDTH];

  init_height(base);
  open_close_pipeline(base, tmp0, tmp1, out);
  return summarize(out);
}
