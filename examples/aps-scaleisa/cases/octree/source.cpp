static const int POINTS = 2048;
static const int LEVELS = 4;
static const int GRID = 16;
static const int QUERIES = 128;
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

static void build_level_counts(
    float points[POINTS][3],
    int counts[LEVELS][CELLS]) {
  for (int level = 0; level < LEVELS; ++level) {
    int cells = GRID >> level;
    for (int i = 0; i < POINTS; ++i) {
      int x = clamp_i((int)(points[i][0] * (float)cells), 0, cells - 1);
      int y = clamp_i((int)(points[i][1] * (float)cells), 0, cells - 1);
      int z = clamp_i((int)(points[i][2] * (float)cells), 0, cells - 1);
      int linear = (x * GRID + y) * GRID + z;
      counts[level][linear] += 1;
    }
  }
}

static void query_neighbors(
    float points[POINTS][3],
    int counts[LEVELS][CELLS],
    int scores[QUERIES]) {
  for (int q = 0; q < QUERIES; ++q) {
    int score = 0;
    float qx = points[q][0];
    float qy = points[q][1];
    float qz = points[q][2];
    for (int level = 0; level < LEVELS; ++level) {
      int cells = GRID >> level;
      int cx = clamp_i((int)(qx * (float)cells), 0, cells - 1);
      int cy = clamp_i((int)(qy * (float)cells), 0, cells - 1);
      int cz = clamp_i((int)(qz * (float)cells), 0, cells - 1);
      for (int dz_idx = 0; dz_idx < 3; ++dz_idx) {
        int dz = dz_idx - 1;
        for (int dy_idx = 0; dy_idx < 3; ++dy_idx) {
          int dy = dy_idx - 1;
          for (int dx_idx = 0; dx_idx < 3; ++dx_idx) {
            int dx = dx_idx - 1;
            int xx = clamp_i(cx + dx, 0, cells - 1);
            int yy = clamp_i(cy + dy, 0, cells - 1);
            int zz = clamp_i(cz + dz, 0, cells - 1);
            int linear = (xx * GRID + yy) * GRID + zz;
            score += counts[level][linear] * (level + 1);
          }
        }
      }
    }
    scores[q] = score;
  }
}

int main() {
  float points[POINTS][3];
  int counts[LEVELS][CELLS];
  int scores[QUERIES];
  int total = 0;

  for (int level = 0; level < LEVELS; ++level) {
    for (int i = 0; i < CELLS; ++i) {
      counts[level][i] = 0;
    }
  }
  for (int i = 0; i < POINTS; ++i) {
    points[i][0] = (float)((i * 13 + 7) % 127) / 127.0f;
    points[i][1] = (float)((i * 17 + 11) % 131) / 131.0f;
    points[i][2] = (float)((i * 19 + 5) % 137) / 137.0f;
  }
  build_level_counts(points, counts);
  query_neighbors(points, counts, scores);
  for (int i = 0; i < QUERIES; ++i) {
    total += scores[i] * ((i & 7) + 1);
  }
  return total;
}
