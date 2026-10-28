static const int WIDTH = 16;
static const int HEIGHT = 16;
static const int RADIUS = 2;

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

static void compute_iss_response(
    float points[HEIGHT][WIDTH][3],
    float response[HEIGHT][WIDTH]) {
  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      float sxx = 0.0f;
      float syy = 0.0f;
      float szz = 0.0f;
      float sxy = 0.0f;
      float sxz = 0.0f;
      float syz = 0.0f;
      float cx = points[y][x][0];
      float cy = points[y][x][1];
      float cz = points[y][x][2];
      for (int ky_idx = 0; ky_idx < 2 * RADIUS + 1; ++ky_idx) {
        int ky = ky_idx - RADIUS;
        for (int kx_idx = 0; kx_idx < 2 * RADIUS + 1; ++kx_idx) {
          int kx = kx_idx - RADIUS;
          int yy = clamp_i(y + ky, 0, HEIGHT - 1);
          int xx = clamp_i(x + kx, 0, WIDTH - 1);
          float dx = points[yy][xx][0] - cx;
          float dy = points[yy][xx][1] - cy;
          float dz = points[yy][xx][2] - cz;
          sxx += dx * dx;
          syy += dy * dy;
          szz += dz * dz;
          sxy += dx * dy;
          sxz += dx * dz;
          syz += dy * dz;
        }
      }
      response[y][x] = (sxx + syy + szz) - 0.5f * (abs_f(sxy) + abs_f(sxz) + abs_f(syz));
    }
  }
}

static void nonmax_suppress(
    float response[HEIGHT][WIDTH],
    int keypoints[HEIGHT][WIDTH]) {
  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      float center = response[y][x];
      int keep = center > 0.02f;
      for (int ky_idx = 0; ky_idx < 3; ++ky_idx) {
        int ky = ky_idx - 1;
        for (int kx_idx = 0; kx_idx < 3; ++kx_idx) {
          int kx = kx_idx - 1;
          int yy = clamp_i(y + ky, 0, HEIGHT - 1);
          int xx = clamp_i(x + kx, 0, WIDTH - 1);
          if ((ky != 0 || kx != 0) && response[yy][xx] > center) {
            keep = 0;
          }
        }
      }
      keypoints[y][x] = keep;
    }
  }
}

static int summarize(
    float response[HEIGHT][WIDTH],
    int keypoints[HEIGHT][WIDTH]) {
  int total = 0;
  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      total += keypoints[y][x] ? ((int)(response[y][x] * 32.0f) + x + y) : 0;
    }
  }
  return total;
}

int main() {
  float points[HEIGHT][WIDTH][3];
  float response[HEIGHT][WIDTH];
  int keypoints[HEIGHT][WIDTH];

  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      points[y][x][0] = (float)x * 0.03f;
      points[y][x][1] = (float)y * 0.03f;
      points[y][x][2] = 0.02f * (float)((x * 5 + y * 9 + (x ^ y)) % 19);
    }
  }
  compute_iss_response(points, response);
  nonmax_suppress(response, keypoints);
  return summarize(response, keypoints);
}
