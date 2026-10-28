static const int HEIGHT = 8;
static const int WIDTH = 8;
static const int ITERS = 2;

static float abs_f(float x) {
  return x < 0.0f ? -x : x;
}

static int clamp_i(int x, int lo, int hi) {
  if (x < lo) {
    return lo;
  }
  if (x > hi) {
    return hi;
  }
  return x;
}

static void init_scene(float depth[HEIGHT][WIDTH], int labels[HEIGHT][WIDTH]) {
  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      int region = (x > 24) + (x > 56) + (y > 30);
      depth[y][x] = 0.1f * (float)region + 0.005f * (float)((x * 7 + y * 5) % 9);
      labels[y][x] = y * WIDTH + x;
    }
  }
}

static void compute_gradient(float depth[HEIGHT][WIDTH], float grad[HEIGHT][WIDTH]) {
  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      int xl = clamp_i(x - 1, 0, WIDTH - 1);
      int xr = clamp_i(x + 1, 0, WIDTH - 1);
      int yu = clamp_i(y - 1, 0, HEIGHT - 1);
      int yd = clamp_i(y + 1, 0, HEIGHT - 1);
      grad[y][x] = abs_f(depth[y][xr] - depth[y][xl]) + abs_f(depth[yd][x] - depth[yu][x]);
    }
  }
}

static void region_grow(
    float depth[HEIGHT][WIDTH],
    float grad[HEIGHT][WIDTH],
    int labels[HEIGHT][WIDTH]) {
  for (int iter = 0; iter < ITERS; ++iter) {
    for (int y = 0; y < HEIGHT; ++y) {
      for (int x = 0; x < WIDTH; ++x) {
        int best = labels[y][x];
        float best_cost = 1.0e30f;
        for (int dy_idx = 0; dy_idx < 3; ++dy_idx) {
          int dy = dy_idx - 1;
          for (int dx_idx = 0; dx_idx < 3; ++dx_idx) {
            int dx = dx_idx - 1;
            int yy = clamp_i(y + dy, 0, HEIGHT - 1);
            int xx = clamp_i(x + dx, 0, WIDTH - 1);
            float cost = abs_f(depth[y][x] - depth[yy][xx]) + grad[yy][xx];
            if (cost < best_cost || (cost == best_cost && labels[yy][xx] < best)) {
              best_cost = cost;
              best = labels[yy][xx];
            }
          }
        }
        labels[y][x] = best;
      }
    }
  }
}

static int summarize(int labels[HEIGHT][WIDTH]) {
  int total = 0;
  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      total += (labels[y][x] & 255);
    }
  }
  return total;
}

int reference_main() {
  float depth[HEIGHT][WIDTH];
  float grad[HEIGHT][WIDTH];
  int labels[HEIGHT][WIDTH];

  init_scene(depth, labels);
  compute_gradient(depth, grad);
  region_grow(depth, grad, labels);
  __builtin_memcpy(expected_0, depth, sizeof(expected_0));
  __builtin_memcpy(expected_1, grad, sizeof(expected_1));
  __builtin_memcpy(expected_2, labels, sizeof(expected_2));
  return summarize(labels);
}
