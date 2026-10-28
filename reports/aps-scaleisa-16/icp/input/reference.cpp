static const int POINTS = 32;
static const int ITERS = 4;

static float abs_f(float x) {
  return x < 0.0f ? -x : x;
}

static void init_clouds(float src[POINTS][3], float dst[POINTS][3]) {
  for (int i = 0; i < POINTS; ++i) {
    float x = (float)((i * 7) % 41) * 0.03f;
    float y = (float)((i * 11) % 37) * 0.025f;
    float z = (float)((i * 13) % 31) * 0.02f;
    src[i][0] = x;
    src[i][1] = y;
    src[i][2] = z;
    dst[i][0] = x + 0.08f;
    dst[i][1] = y - 0.04f;
    dst[i][2] = z + 0.06f;
  }
}

static float sq_dist(float a[3], float b[3], float tx, float ty, float tz) {
  float dx = a[0] + tx - b[0];
  float dy = a[1] + ty - b[1];
  float dz = a[2] + tz - b[2];
  return dx * dx + dy * dy + dz * dz;
}

static void icp_align(
    float src[POINTS][3],
    float dst[POINTS][3],
    float transform[3],
    float residuals[ITERS]) {
  float tx = 0.0f;
  float ty = 0.0f;
  float tz = 0.0f;

  for (int iter = 0; iter < ITERS; ++iter) {
    float sum_dx = 0.0f;
    float sum_dy = 0.0f;
    float sum_dz = 0.0f;
    float fitness = 0.0f;
    for (int i = 0; i < POINTS; ++i) {
      float best = 1.0e30f;
      int best_j = 0;
      for (int j = 0; j < POINTS; ++j) {
        float d = sq_dist(src[i], dst[j], tx, ty, tz);
        if (d < best) {
          best = d;
          best_j = j;
        }
      }
      sum_dx += dst[best_j][0] - (src[i][0] + tx);
      sum_dy += dst[best_j][1] - (src[i][1] + ty);
      sum_dz += dst[best_j][2] - (src[i][2] + tz);
      fitness += best;
    }
    tx += 0.5f * sum_dx / (float)POINTS;
    ty += 0.5f * sum_dy / (float)POINTS;
    tz += 0.5f * sum_dz / (float)POINTS;
    residuals[iter] = fitness / (float)POINTS;
  }

  transform[0] = tx;
  transform[1] = ty;
  transform[2] = tz;
}

static int summarize(float transform[3], float residuals[ITERS]) {
  int total = 0;
  total += (int)(transform[0] * 1000.0f);
  total += (int)(transform[1] * 1200.0f);
  total += (int)(transform[2] * 1400.0f);
  for (int i = 0; i < ITERS; ++i) {
    total += (int)(residuals[i] * 500.0f);
  }
  return abs_f((float)total) > 0.0f ? total : 1;
}

int main() {
  float src[POINTS][3];
  float dst[POINTS][3];
  float transform[3];
  float residuals[ITERS];

  init_clouds(src, dst);
  icp_align(src, dst, transform, residuals);
  return summarize(transform, residuals);
}
