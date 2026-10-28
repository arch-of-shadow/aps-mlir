static const int POSES = 18;
static const int EDGES = 30;
static const int ITERS = 7;

static float abs_f(float x) {
  return x < 0.0f ? -x : x;
}

static void init_pose3_problem(
    float pose[POSES][6],
    int src[EDGES],
    int dst[EDGES],
    float meas[EDGES][6],
    float weight[EDGES][6]) {
  for (int i = 0; i < POSES; ++i) {
    pose[i][0] = 0.55f * (float)i + 0.04f * (float)(i % 3);
    pose[i][1] = 0.15f * (float)(i % 4);
    pose[i][2] = 0.08f * (float)(i % 5);
    pose[i][3] = 0.01f * (float)((i % 5) - 2);
    pose[i][4] = 0.008f * (float)((i % 4) - 1);
    pose[i][5] = 0.012f * (float)((i % 3) - 1);
  }

  int edge = 0;
  for (int i = 0; i < POSES - 1; ++i) {
    src[edge] = i;
    dst[edge] = i + 1;
    meas[edge][0] = 0.55f;
    meas[edge][1] = 0.05f * (float)(i % 2);
    meas[edge][2] = 0.03f * (float)(i % 3);
    meas[edge][3] = 0.005f * (float)((i % 3) - 1);
    meas[edge][4] = 0.004f * (float)((i % 4) - 2);
    meas[edge][5] = 0.006f * (float)((i % 5) - 2);
    for (int j = 0; j < 6; ++j) {
      weight[edge][j] = j < 3 ? 1.3f : 1.8f;
    }
    edge += 1;
  }

  for (int i = 0; edge < EDGES; ++i, ++edge) {
    int base = (i * 4 + 2) % (POSES - 5);
    src[edge] = base;
    dst[edge] = base + 3 + (i % 2);
    meas[edge][0] = 1.75f + 0.05f * (float)(i % 3);
    meas[edge][1] = 0.18f * (float)(i % 2);
    meas[edge][2] = 0.11f * (float)(i % 4);
    meas[edge][3] = 0.01f * (float)((i % 4) - 2);
    meas[edge][4] = 0.009f * (float)((i % 5) - 2);
    meas[edge][5] = 0.012f * (float)((i % 6) - 3);
    for (int j = 0; j < 6; ++j) {
      weight[edge][j] = j < 3 ? 2.0f : 2.6f;
    }
  }

  for (int i = 0; i < POSES; ++i) {
    pose[i][0] += 0.06f * (float)((i % 4) - 1);
    pose[i][1] -= 0.04f * (float)(i % 3);
    pose[i][2] += 0.03f * (float)(i % 2);
    pose[i][3] += 0.003f * (float)(i % 2);
    pose[i][4] -= 0.002f * (float)(i % 3);
    pose[i][5] += 0.004f * (float)(i % 4);
  }
}

static void solve_pose3(
    float pose[POSES][6],
    int src[EDGES],
    int dst[EDGES],
    float meas[EDGES][6],
    float weight[EDGES][6]) {
  float delta[POSES][6];
  float diag[POSES][6];

  for (int iter = 0; iter < ITERS; ++iter) {
    for (int i = 0; i < POSES; ++i) {
      for (int j = 0; j < 6; ++j) {
        delta[i][j] = 0.0f;
        diag[i][j] = 0.001f;
      }
    }

    for (int e = 0; e < EDGES; ++e) {
      int s = src[e];
      int d = dst[e];
      for (int j = 0; j < 6; ++j) {
        float pred = pose[d][j] - pose[s][j];
        float residual = pred - meas[e][j];
        float scaled = weight[e][j] * residual;
        delta[s][j] += scaled;
        delta[d][j] -= scaled;
        diag[s][j] += weight[e][j];
        diag[d][j] += weight[e][j];
      }
    }

    for (int j = 0; j < 6; ++j) {
      pose[0][j] = 0.0f;
    }
    for (int i = 1; i < POSES; ++i) {
      for (int j = 0; j < 6; ++j) {
        float step = j < 3 ? 0.20f : 0.12f;
        pose[i][j] -= step * delta[i][j] / diag[i][j];
      }
    }
  }
}

static int summarize(float pose[POSES][6]) {
  int total = 0;
  for (int i = 0; i < POSES; ++i) {
    for (int j = 0; j < 6; ++j) {
      total += (int)(pose[i][j] * (83.0f + 19.0f * (float)j));
    }
  }
  return abs_f((float)total) > 0.0f ? total : 1;
}

int reference_main() {
  float pose[POSES][6];
  int src[EDGES];
  int dst[EDGES];
  float meas[EDGES][6];
  float weight[EDGES][6];

  init_pose3_problem(pose, src, dst, meas, weight);
  solve_pose3(pose, src, dst, meas, weight);
  __builtin_memcpy(expected_0, pose, sizeof(expected_0));
  __builtin_memcpy(expected_1, src, sizeof(expected_1));
  __builtin_memcpy(expected_2, dst, sizeof(expected_2));
  __builtin_memcpy(expected_3, meas, sizeof(expected_3));
  __builtin_memcpy(expected_4, weight, sizeof(expected_4));
  return summarize(pose);
}
