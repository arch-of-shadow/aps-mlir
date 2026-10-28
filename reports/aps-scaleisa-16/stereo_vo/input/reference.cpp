static const int LANDMARKS = 32;
static const int ITERS = 10;
static const float FX = 180.0f;
static const float FY = 178.0f;
static const float BASELINE = 0.20f;
static const float DEPTH_BIAS = 4.2f;

static float abs_f(float x) {
  return x < 0.0f ? -x : x;
}

static void stereo_project(
    float tx,
    float ty,
    float tz,
    float point[3],
    float obs[3]) {
  float x = point[0] - tx;
  float y = point[1] - ty;
  float z = point[2] - tz + DEPTH_BIAS;
  float inv_z = 1.0f / z;
  obs[0] = FX * x * inv_z;
  obs[1] = FY * y * inv_z;
  obs[2] = FX * BASELINE * inv_z;
}

static void init_stereo_problem(
    float translation[3],
    float points[LANDMARKS][3],
    float meas0[LANDMARKS][3],
    float meas1[LANDMARKS][3]) {
  float gt_translation[3];
  gt_translation[0] = 0.35f;
  gt_translation[1] = -0.04f;
  gt_translation[2] = 0.08f;
  translation[0] = 0.12f;
  translation[1] = 0.03f;
  translation[2] = -0.02f;

  for (int i = 0; i < LANDMARKS; ++i) {
    points[i][0] = -1.0f + 0.42f * (float)(i % 8);
    points[i][1] = -0.6f + 0.31f * (float)(i / 8);
    points[i][2] = 1.6f + 0.11f * (float)(i % 5);
    stereo_project(0.0f, 0.0f, 0.0f, points[i], meas0[i]);
    stereo_project(
        gt_translation[0], gt_translation[1], gt_translation[2], points[i], meas1[i]);
    meas0[i][0] += 0.01f * (float)((i % 3) - 1);
    meas0[i][1] -= 0.01f * (float)((i % 4) - 2);
    meas1[i][0] += 0.015f * (float)((i % 5) - 2);
    meas1[i][1] -= 0.012f * (float)((i % 6) - 3);
    meas1[i][2] += 0.008f * (float)((i % 4) - 1);
    points[i][0] += 0.06f * (float)(i % 2);
    points[i][1] -= 0.05f * (float)(i % 3);
    points[i][2] += 0.04f * (float)((i % 4) - 1);
  }
}

static void optimize_stereo(
    float translation[3],
    float points[LANDMARKS][3],
    float meas0[LANDMARKS][3],
    float meas1[LANDMARKS][3]) {
  float trans_delta[3];
  float trans_diag[3];
  float point_delta[LANDMARKS][3];
  float point_diag[LANDMARKS][3];

  for (int iter = 0; iter < ITERS; ++iter) {
    trans_delta[0] = 0.0f;
    trans_delta[1] = 0.0f;
    trans_delta[2] = 0.0f;
    trans_diag[0] = 0.001f;
    trans_diag[1] = 0.001f;
    trans_diag[2] = 0.001f;
    for (int i = 0; i < LANDMARKS; ++i) {
      point_delta[i][0] = 0.0f;
      point_delta[i][1] = 0.0f;
      point_delta[i][2] = 0.0f;
      point_diag[i][0] = 0.001f;
      point_diag[i][1] = 0.001f;
      point_diag[i][2] = 0.001f;
    }

    for (int i = 0; i < LANDMARKS; ++i) {
      float pred0[3];
      float pred[3];
      stereo_project(0.0f, 0.0f, 0.0f, points[i], pred0);
      stereo_project(translation[0], translation[1], translation[2], points[i], pred);
      float z0 = points[i][2] + DEPTH_BIAS;
      float inv_z0 = 1.0f / z0;
      float du0 = pred0[0] - meas0[i][0];
      float dv0 = pred0[1] - meas0[i][1];
      float dd0 = pred0[2] - meas0[i][2];
      float jx0 = FX * inv_z0;
      float jy0 = FY * inv_z0;
      float jz0_u = -FX * points[i][0] * inv_z0 * inv_z0;
      float jz0_v = -FY * points[i][1] * inv_z0 * inv_z0;
      float jz0_d = -FX * BASELINE * inv_z0 * inv_z0;
      float x = points[i][0] - translation[0];
      float y = points[i][1] - translation[1];
      float z = points[i][2] - translation[2] + DEPTH_BIAS;
      float inv_z = 1.0f / z;
      float du = pred[0] - meas1[i][0];
      float dv = pred[1] - meas1[i][1];
      float dd = pred[2] - meas1[i][2];
      float jx = FX * inv_z;
      float jy = FY * inv_z;
      float jz_u = -FX * x * inv_z * inv_z;
      float jz_v = -FY * y * inv_z * inv_z;
      float jz_d = -FX * BASELINE * inv_z * inv_z;

      trans_delta[0] += -du * jx;
      trans_delta[1] += -dv * jy;
      trans_delta[2] += -(du * jz_u + dv * jz_v + dd * jz_d);
      trans_diag[0] += jx * jx;
      trans_diag[1] += jy * jy;
      trans_diag[2] += jz_u * jz_u + jz_v * jz_v + jz_d * jz_d;

      point_delta[i][0] += du0 * jx0;
      point_delta[i][1] += dv0 * jy0;
      point_delta[i][2] += du0 * jz0_u + dv0 * jz0_v + dd0 * jz0_d;
      point_diag[i][0] += jx0 * jx0;
      point_diag[i][1] += jy0 * jy0;
      point_diag[i][2] += jz0_u * jz0_u + jz0_v * jz0_v + jz0_d * jz0_d;

      point_delta[i][0] += du * jx;
      point_delta[i][1] += dv * jy;
      point_delta[i][2] += du * jz_u + dv * jz_v + dd * jz_d;
      point_diag[i][0] += jx * jx;
      point_diag[i][1] += jy * jy;
      point_diag[i][2] += jz_u * jz_u + jz_v * jz_v + jz_d * jz_d;
    }

    translation[0] -= 0.25f * trans_delta[0] / trans_diag[0];
    translation[1] -= 0.25f * trans_delta[1] / trans_diag[1];
    translation[2] -= 0.18f * trans_delta[2] / trans_diag[2];
    for (int i = 0; i < LANDMARKS; ++i) {
      points[i][0] -= 0.16f * point_delta[i][0] / point_diag[i][0];
      points[i][1] -= 0.16f * point_delta[i][1] / point_diag[i][1];
      points[i][2] -= 0.12f * point_delta[i][2] / point_diag[i][2];
    }
  }
}

static int summarize(float translation[3], float points[LANDMARKS][3]) {
  int total = (int)(translation[0] * 211.0f) + (int)(translation[1] * 257.0f) +
              (int)(translation[2] * 307.0f);
  for (int i = 0; i < LANDMARKS; ++i) {
    total += (int)(points[i][0] * 79.0f);
    total += (int)(points[i][1] * 103.0f);
    total += (int)(points[i][2] * 127.0f);
  }
  return abs_f((float)total) > 0.0f ? total : 1;
}

int reference_main() {
  float translation[3];
  float points[LANDMARKS][3];
  float meas0[LANDMARKS][3];
  float meas1[LANDMARKS][3];

  init_stereo_problem(translation, points, meas0, meas1);
  optimize_stereo(translation, points, meas0, meas1);
  __builtin_memcpy(expected_0, translation, sizeof(expected_0));
  __builtin_memcpy(expected_1, points, sizeof(expected_1));
  __builtin_memcpy(expected_2, meas0, sizeof(expected_2));
  __builtin_memcpy(expected_3, meas1, sizeof(expected_3));
  return summarize(translation, points);
}
