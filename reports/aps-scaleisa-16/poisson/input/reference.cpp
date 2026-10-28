static const int SAMPLES = 24;
static const int GRID = 6;
static const int ITERS = 2;
static const int CELLS = GRID * GRID * GRID;

static int clamp_i(int x, int lo, int hi) {
  if (x < lo) {
    return lo;
  }
  if (x > hi) {
    return hi;
  }
  return x;
}

static float abs_f(float x) {
  return x < 0.0f ? -x : x;
}

static void splat_divergence(
    float points[SAMPLES][6],
    float rhs[CELLS]) {
  for (int i = 0; i < SAMPLES; ++i) {
    int x = clamp_i((int)(points[i][0] * (float)GRID), 0, GRID - 1);
    int y = clamp_i((int)(points[i][1] * (float)GRID), 0, GRID - 1);
    int z = clamp_i((int)(points[i][2] * (float)GRID), 0, GRID - 1);
    int linear = (x * GRID + y) * GRID + z;
    rhs[linear] += points[i][3] + points[i][4] + points[i][5];
  }
}

static void solve_poisson(
    float rhs[CELLS],
    float grid[CELLS],
    float tmp[CELLS]) {
  for (int iter = 0; iter < ITERS; ++iter) {
    for (int x = 1; x < GRID - 1; ++x) {
      for (int y = 1; y < GRID - 1; ++y) {
        for (int z = 1; z < GRID - 1; ++z) {
          int c = (x * GRID + y) * GRID + z;
          int xm = ((x - 1) * GRID + y) * GRID + z;
          int xp = ((x + 1) * GRID + y) * GRID + z;
          int ym = (x * GRID + (y - 1)) * GRID + z;
          int yp = (x * GRID + (y + 1)) * GRID + z;
          int zm = (x * GRID + y) * GRID + (z - 1);
          int zp = (x * GRID + y) * GRID + (z + 1);
          float nb = grid[xm] + grid[xp] + grid[ym] + grid[yp] + grid[zm] + grid[zp];
          tmp[c] = (rhs[c] + nb) / 6.0f;
        }
      }
    }
    for (int x = 1; x < GRID - 1; ++x) {
      for (int y = 1; y < GRID - 1; ++y) {
        for (int z = 1; z < GRID - 1; ++z) {
          int c = (x * GRID + y) * GRID + z;
          grid[c] = tmp[c];
        }
      }
    }
  }
}

static int extract_surface_proxy(float grid[CELLS]) {
  int crossings = 0;
  for (int x = 1; x < GRID - 1; ++x) {
    for (int y = 1; y < GRID - 1; ++y) {
      for (int z = 1; z < GRID - 1; ++z) {
        int c = (x * GRID + y) * GRID + z;
        int xp = ((x + 1) * GRID + y) * GRID + z;
        int yp = (x * GRID + (y + 1)) * GRID + z;
        int zp = (x * GRID + y) * GRID + (z + 1);
        float v = grid[c];
        crossings += (v * grid[xp] < 0.0f);
        crossings += (v * grid[yp] < 0.0f);
        crossings += (v * grid[zp] < 0.0f);
      }
    }
  }
  return crossings;
}

int reference_main() {
  float points[SAMPLES][6];
  float rhs[CELLS];
  float grid[CELLS];
  float tmp[CELLS];

  for (int i = 0; i < CELLS; ++i) {
    rhs[i] = 0.0f;
    grid[i] = 0.0f;
    tmp[i] = 0.0f;
  }
  for (int i = 0; i < SAMPLES; ++i) {
    points[i][0] = (float)((i * 7 + 3) % 97) / 97.0f;
    points[i][1] = (float)((i * 11 + 5) % 89) / 89.0f;
    points[i][2] = (float)((i * 13 + 7) % 83) / 83.0f;
    points[i][3] = 0.5f - points[i][0];
    points[i][4] = 0.5f - points[i][1];
    points[i][5] = 0.5f - points[i][2];
  }
  splat_divergence(points, rhs);
  solve_poisson(rhs, grid, tmp);
  __builtin_memcpy(expected_0, points, sizeof(expected_0));
  __builtin_memcpy(expected_1, rhs, sizeof(expected_1));
  __builtin_memcpy(expected_2, grid, sizeof(expected_2));
  __builtin_memcpy(expected_3, tmp, sizeof(expected_3));
  return extract_surface_proxy(grid);
}
