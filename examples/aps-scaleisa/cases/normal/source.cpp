static const int WIDTH = 12;
static const int HEIGHT = 12;

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

static void compute_normals(
    float points[HEIGHT][WIDTH][3],
    float normals[HEIGHT][WIDTH][3],
    float curvature[HEIGHT][WIDTH]) {
  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      int xl = clamp_i(x - 1, 0, WIDTH - 1);
      int xr = clamp_i(x + 1, 0, WIDTH - 1);
      int yu = clamp_i(y - 1, 0, HEIGHT - 1);
      int yd = clamp_i(y + 1, 0, HEIGHT - 1);
      float dx0 = points[y][xr][0] - points[y][xl][0];
      float dx1 = points[y][xr][1] - points[y][xl][1];
      float dx2 = points[y][xr][2] - points[y][xl][2];
      float dy0 = points[yd][x][0] - points[yu][x][0];
      float dy1 = points[yd][x][1] - points[yu][x][1];
      float dy2 = points[yd][x][2] - points[yu][x][2];
      float nx = dx1 * dy2 - dx2 * dy1;
      float ny = dx2 * dy0 - dx0 * dy2;
      float nz = dx0 * dy1 - dx1 * dy0;
      float inv_norm = 1.0f / (abs_f(nx) + abs_f(ny) + abs_f(nz) + 1.0e-3f);
      normals[y][x][0] = nx * inv_norm;
      normals[y][x][1] = ny * inv_norm;
      normals[y][x][2] = nz * inv_norm;
      curvature[y][x] = abs_f(points[y][xr][2] - 2.0f * points[y][x][2] + points[y][xl][2]) +
                        abs_f(points[yd][x][2] - 2.0f * points[y][x][2] + points[yu][x][2]);
    }
  }
}

static void smooth_normals(
    float normals[HEIGHT][WIDTH][3],
    float smoothed[HEIGHT][WIDTH][3]) {
  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      float acc0 = 0.0f;
      float acc1 = 0.0f;
      float acc2 = 0.0f;
      for (int ky_idx = 0; ky_idx < 3; ++ky_idx) {
        int ky = ky_idx - 1;
        for (int kx_idx = 0; kx_idx < 3; ++kx_idx) {
          int kx = kx_idx - 1;
          int yy = clamp_i(y + ky, 0, HEIGHT - 1);
          int xx = clamp_i(x + kx, 0, WIDTH - 1);
          acc0 += normals[yy][xx][0];
          acc1 += normals[yy][xx][1];
          acc2 += normals[yy][xx][2];
        }
      }
      smoothed[y][x][0] = acc0 / 9.0f;
      smoothed[y][x][1] = acc1 / 9.0f;
      smoothed[y][x][2] = acc2 / 9.0f;
    }
  }
}

static int accumulate_signature(
    float smoothed[HEIGHT][WIDTH][3],
    float curvature[HEIGHT][WIDTH]) {
  int total = 0;
  for (int y = 1; y < HEIGHT - 1; ++y) {
    for (int x = 1; x < WIDTH - 1; ++x) {
      float score = 17.0f * smoothed[y][x][0] +
                    23.0f * smoothed[y][x][1] +
                    31.0f * smoothed[y][x][2] +
                    11.0f * curvature[y][x];
      total += (int)(score * 64.0f);
    }
  }
  return total;
}

int main() {
  float points[HEIGHT][WIDTH][3];
  float normals[HEIGHT][WIDTH][3];
  float smoothed[HEIGHT][WIDTH][3];
  float curvature[HEIGHT][WIDTH];

  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      float fx = (float)x * 0.05f;
      float fy = (float)y * 0.04f;
      points[y][x][0] = fx;
      points[y][x][1] = fy;
      points[y][x][2] = 0.15f * fx + 0.07f * fy + 0.01f * (float)((x * y) % 9);
    }
  }
  compute_normals(points, normals, curvature);
  smooth_normals(normals, smoothed);
  return accumulate_signature(smoothed, curvature);
}
