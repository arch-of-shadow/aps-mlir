static const int CAMERAS = 8;
static const int POINTS = 24;
static const int OBS = CAMERAS * POINTS;
static const int ITERS = 6;
static const float FX = 220.0f;
static const float FY = 216.0f;
static const float DEPTH_BIAS = 6.0f;

static float abs_f(float x) {
  return x < 0.0f ? -x : x;
}

static void project_point(
    float camera[3],
    float point[3],
    float *u,
    float *v,
    float *inv_z) {
  float x = point[0] - camera[0];
  float y = point[1] - camera[1];
  float z = point[2] - camera[2] + DEPTH_BIAS;
  *inv_z = 1.0f / z;
  *u = FX * x * (*inv_z);
  *v = FY * y * (*inv_z);
}

static void init_sfm_problem(
    float cameras[CAMERAS][3],
    float points[POINTS][3],
    int obs_camera[OBS],
    int obs_point[OBS],
    float meas[OBS][2]) {
  for (int c = 0; c < CAMERAS; ++c) {
    cameras[c][0] = -1.8f + 0.55f * (float)c;
    cameras[c][1] = 0.25f * (float)((c % 3) - 1);
    cameras[c][2] = 0.12f * (float)(c % 2);
  }
  for (int p = 0; p < POINTS; ++p) {
    points[p][0] = -1.2f + 0.45f * (float)(p % 6);
    points[p][1] = -0.9f + 0.6f * (float)(p / 6);
    points[p][2] = 2.8f + 0.18f * (float)(p % 5);
  }

  int obs = 0;
  for (int c = 0; c < CAMERAS; ++c) {
    for (int p = 0; p < POINTS; ++p) {
      float u;
      float v;
      float inv_z;
      obs_camera[obs] = c;
      obs_point[obs] = p;
      project_point(cameras[c], points[p], &u, &v, &inv_z);
      meas[obs][0] = u + 0.03f * (float)((obs % 5) - 2);
      meas[obs][1] = v - 0.02f * (float)((obs % 7) - 3);
      obs += 1;
    }
  }

  for (int c = 0; c < CAMERAS; ++c) {
    cameras[c][0] += 0.12f * (float)((c % 4) - 1);
    cameras[c][1] -= 0.08f * (float)(c % 2);
    cameras[c][2] += 0.04f * (float)((c % 3) - 1);
  }
  for (int p = 0; p < POINTS; ++p) {
    points[p][0] -= 0.10f * (float)(p % 3);
    points[p][1] += 0.07f * (float)(p % 2);
    points[p][2] += 0.05f * (float)((p % 4) - 1);
  }
}

static void optimize_sfm(
    float cameras[CAMERAS][3],
    float points[POINTS][3],
    int obs_camera[OBS],
    int obs_point[OBS],
    float meas[OBS][2]) {
  float camera_delta[CAMERAS][3];
  float camera_diag[CAMERAS][3];
  float point_delta[POINTS][3];
  float point_diag[POINTS][3];

  for (int iter = 0; iter < ITERS; ++iter) {
    for (int c = 0; c < CAMERAS; ++c) {
      for (int j = 0; j < 3; ++j) {
        camera_delta[c][j] = 0.0f;
        camera_diag[c][j] = 0.001f;
      }
    }
    for (int p = 0; p < POINTS; ++p) {
      for (int j = 0; j < 3; ++j) {
        point_delta[p][j] = 0.0f;
        point_diag[p][j] = 0.001f;
      }
    }

    for (int i = 0; i < OBS; ++i) {
      int c = obs_camera[i];
      int p = obs_point[i];
      float u;
      float v;
      float inv_z;
      project_point(cameras[c], points[p], &u, &v, &inv_z);
      float x = points[p][0] - cameras[c][0];
      float y = points[p][1] - cameras[c][1];
      float du = u - meas[i][0];
      float dv = v - meas[i][1];
      float jx = FX * inv_z;
      float jy = FY * inv_z;
      float jz_u = -FX * x * inv_z * inv_z;
      float jz_v = -FY * y * inv_z * inv_z;

      camera_delta[c][0] += -du * jx;
      camera_delta[c][1] += -dv * jy;
      camera_delta[c][2] += -(du * jz_u + dv * jz_v);
      camera_diag[c][0] += jx * jx;
      camera_diag[c][1] += jy * jy;
      camera_diag[c][2] += jz_u * jz_u + jz_v * jz_v;

      point_delta[p][0] += du * jx;
      point_delta[p][1] += dv * jy;
      point_delta[p][2] += du * jz_u + dv * jz_v;
      point_diag[p][0] += jx * jx;
      point_diag[p][1] += jy * jy;
      point_diag[p][2] += jz_u * jz_u + jz_v * jz_v;
    }

    for (int c = 0; c < CAMERAS; ++c) {
      for (int j = 0; j < 3; ++j) {
        cameras[c][j] -= 0.18f * camera_delta[c][j] / camera_diag[c][j];
      }
    }
    for (int p = 0; p < POINTS; ++p) {
      for (int j = 0; j < 3; ++j) {
        points[p][j] -= 0.20f * point_delta[p][j] / point_diag[p][j];
      }
    }
  }
}

static int summarize(float cameras[CAMERAS][3], float points[POINTS][3]) {
  int total = 0;
  for (int c = 0; c < CAMERAS; ++c) {
    total += (int)(cameras[c][0] * 97.0f);
    total += (int)(cameras[c][1] * 131.0f);
    total += (int)(cameras[c][2] * 149.0f);
  }
  for (int p = 0; p < POINTS; ++p) {
    total += (int)(points[p][0] * 83.0f);
    total += (int)(points[p][1] * 109.0f);
    total += (int)(points[p][2] * 127.0f);
  }
  return abs_f((float)total) > 0.0f ? total : 1;
}

int reference_main() {
  float cameras[CAMERAS][3];
  float points[POINTS][3];
  int obs_camera[OBS];
  int obs_point[OBS];
  float meas[OBS][2];

  init_sfm_problem(cameras, points, obs_camera, obs_point, meas);
  optimize_sfm(cameras, points, obs_camera, obs_point, meas);
  __builtin_memcpy(expected_0, cameras, sizeof(expected_0));
  __builtin_memcpy(expected_1, points, sizeof(expected_1));
  __builtin_memcpy(expected_2, obs_camera, sizeof(expected_2));
  __builtin_memcpy(expected_3, obs_point, sizeof(expected_3));
  __builtin_memcpy(expected_4, meas, sizeof(expected_4));
  return summarize(cameras, points);
}
