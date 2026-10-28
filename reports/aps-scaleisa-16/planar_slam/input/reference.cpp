static const int POSES = 16;
static const int LANDMARKS = 20;
static const int OBS = 64;
static const int ITERS = 7;

static float abs_f(float x) {
  return x < 0.0f ? -x : x;
}

static void init_problem(
    float poses[POSES][3],
    float landmarks[LANDMARKS][2],
    int obs_pose[OBS],
    int obs_landmark[OBS],
    float meas[OBS][2]) {
  for (int i = 0; i < POSES; ++i) {
    poses[i][0] = 0.75f * (float)i + 0.05f * (float)(i % 3);
    poses[i][1] = 0.18f * (float)(i % 2);
    poses[i][2] = 0.015f * (float)((i % 5) - 2);
  }
  for (int i = 0; i < LANDMARKS; ++i) {
    landmarks[i][0] = 1.6f + 0.38f * (float)(i % 5);
    landmarks[i][1] = 1.2f + 0.27f * (float)(i / 5);
  }
  for (int i = 0; i < OBS; ++i) {
    int p = i % POSES;
    int l = (i * 3 + i / 7) % LANDMARKS;
    float dx = landmarks[l][0] - poses[p][0];
    float dy = landmarks[l][1] - poses[p][1];
    float theta = poses[p][2];
    obs_pose[i] = p;
    obs_landmark[i] = l;
    meas[i][0] = dx + theta * dy + 0.02f * (float)(i % 3);
    meas[i][1] = dy - theta * dx - 0.015f * (float)(i % 4);
  }

  for (int i = 0; i < POSES; ++i) {
    poses[i][0] += 0.12f * (float)((i % 4) - 1);
    poses[i][1] -= 0.07f * (float)(i % 3);
    poses[i][2] += 0.01f * (float)(i % 2);
  }
  for (int i = 0; i < LANDMARKS; ++i) {
    landmarks[i][0] -= 0.09f * (float)(i % 3);
    landmarks[i][1] += 0.06f * (float)(i % 2);
  }
}

static void optimize_planar(
    float poses[POSES][3],
    float landmarks[LANDMARKS][2],
    int obs_pose[OBS],
    int obs_landmark[OBS],
    float meas[OBS][2]) {
  float pose_delta[POSES][3];
  float pose_diag[POSES][3];
  float landmark_delta[LANDMARKS][2];
  float landmark_diag[LANDMARKS][2];

  for (int iter = 0; iter < ITERS; ++iter) {
    for (int i = 0; i < POSES; ++i) {
      pose_delta[i][0] = 0.0f;
      pose_delta[i][1] = 0.0f;
      pose_delta[i][2] = 0.0f;
      pose_diag[i][0] = 0.001f;
      pose_diag[i][1] = 0.001f;
      pose_diag[i][2] = 0.001f;
    }
    for (int i = 0; i < LANDMARKS; ++i) {
      landmark_delta[i][0] = 0.0f;
      landmark_delta[i][1] = 0.0f;
      landmark_diag[i][0] = 0.001f;
      landmark_diag[i][1] = 0.001f;
    }

    for (int i = 0; i < OBS; ++i) {
      int p = obs_pose[i];
      int l = obs_landmark[i];
      float dx = landmarks[l][0] - poses[p][0];
      float dy = landmarks[l][1] - poses[p][1];
      float theta = poses[p][2];
      float pred0 = dx + theta * dy;
      float pred1 = dy - theta * dx;
      float r0 = pred0 - meas[i][0];
      float r1 = pred1 - meas[i][1];

      pose_delta[p][0] += -0.9f * r0 + 0.2f * theta * r1;
      pose_delta[p][1] += -0.9f * r1 - 0.2f * theta * r0;
      pose_delta[p][2] += dy * r0 - dx * r1;
      pose_diag[p][0] += 1.0f;
      pose_diag[p][1] += 1.0f;
      pose_diag[p][2] += 0.4f * (dx * dx + dy * dy) + 1.0f;

      landmark_delta[l][0] += r0 - 0.1f * theta * r1;
      landmark_delta[l][1] += r1 + 0.1f * theta * r0;
      landmark_diag[l][0] += 1.0f;
      landmark_diag[l][1] += 1.0f;
    }

    poses[0][0] = 0.0f;
    poses[0][1] = 0.0f;
    poses[0][2] = 0.0f;
    for (int i = 1; i < POSES; ++i) {
      poses[i][0] -= 0.22f * pose_delta[i][0] / pose_diag[i][0];
      poses[i][1] -= 0.22f * pose_delta[i][1] / pose_diag[i][1];
      poses[i][2] -= 0.18f * pose_delta[i][2] / pose_diag[i][2];
    }
    for (int i = 0; i < LANDMARKS; ++i) {
      landmarks[i][0] -= 0.25f * landmark_delta[i][0] / landmark_diag[i][0];
      landmarks[i][1] -= 0.25f * landmark_delta[i][1] / landmark_diag[i][1];
    }
  }
}

static int summarize(float poses[POSES][3], float landmarks[LANDMARKS][2]) {
  int total = 0;
  for (int i = 0; i < POSES; ++i) {
    total += (int)(poses[i][0] * 97.0f);
    total += (int)(poses[i][1] * 131.0f);
    total += (int)(poses[i][2] * 197.0f);
  }
  for (int i = 0; i < LANDMARKS; ++i) {
    total += (int)(landmarks[i][0] * 151.0f);
    total += (int)(landmarks[i][1] * 181.0f);
  }
  return abs_f((float)total) > 0.0f ? total : 1;
}

int reference_main() {
  float poses[POSES][3];
  float landmarks[LANDMARKS][2];
  int obs_pose[OBS];
  int obs_landmark[OBS];
  float meas[OBS][2];

  init_problem(poses, landmarks, obs_pose, obs_landmark, meas);
  optimize_planar(poses, landmarks, obs_pose, obs_landmark, meas);
  __builtin_memcpy(expected_0, poses, sizeof(expected_0));
  __builtin_memcpy(expected_1, landmarks, sizeof(expected_1));
  __builtin_memcpy(expected_2, obs_pose, sizeof(expected_2));
  __builtin_memcpy(expected_3, obs_landmark, sizeof(expected_3));
  __builtin_memcpy(expected_4, meas, sizeof(expected_4));
  return summarize(poses, landmarks);
}
