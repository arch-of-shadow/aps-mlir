#ifndef EXAMPLE_CAPTURE
#define EXAMPLE_CAPTURE() ((void)0)
#endif

static const int POSES = 24;
static const int EDGES = 40;
static const int ITERS = 8;

static float abs_f(float x) {
  return x < 0.0f ? -x : x;
}

static void build_pose_graph(
    int src[EDGES],
    int dst[EDGES],
    float meas[EDGES][3],
    float weight[EDGES][3]) {
  int edge = 0;
  for (int i = 0; i < POSES - 1; ++i) {
    src[edge] = i;
    dst[edge] = i + 1;
    meas[edge][0] = 0.9f + 0.02f * (float)(i % 3);
    meas[edge][1] = 0.1f * (float)(i % 2);
    meas[edge][2] = 0.03f * (float)((i % 5) - 2);
    weight[edge][0] = 1.0f;
    weight[edge][1] = 1.1f;
    weight[edge][2] = 1.3f;
    edge += 1;
  }

  for (int i = 0; edge < EDGES; ++i, ++edge) {
    int base = (i * 5 + 3) % (POSES - 6);
    src[edge] = base;
    dst[edge] = base + 4 + (i % 3);
    meas[edge][0] = 4.0f + 0.15f * (float)(i % 4);
    meas[edge][1] = -0.2f + 0.07f * (float)(i % 5);
    meas[edge][2] = 0.05f * (float)((i % 7) - 3);
    weight[edge][0] = 1.7f;
    weight[edge][1] = 1.8f;
    weight[edge][2] = 2.2f;
  }
}

static void init_state(float pose[POSES][3]) {
  for (int i = 0; i < POSES; ++i) {
    pose[i][0] = 0.88f * (float)i + 0.04f * (float)(i % 4);
    pose[i][1] = 0.18f * (float)(i % 3) - 0.01f * (float)i;
    pose[i][2] = 0.02f * (float)((i % 6) - 3);
  }
}

static void solve_pose_graph(
    float pose[POSES][3],
    int src[EDGES],
    int dst[EDGES],
    float meas[EDGES][3],
    float weight[EDGES][3]) {
  float delta[POSES][3];
  float diag[POSES][3];
  for (int iter = 0; iter < ITERS; ++iter) {
    for (int i = 0; i < POSES; ++i) {
      for (int j = 0; j < 3; ++j) {
        delta[i][j] = 0.0f;
        diag[i][j] = 0.001f;
      }
    }

    for (int e = 0; e < EDGES; ++e) {
      int s = src[e];
      int d = dst[e];
      for (int j = 0; j < 3; ++j) {
        float pred = pose[d][j] - pose[s][j];
        float residual = pred - meas[e][j];
        float scaled = weight[e][j] * residual;
        delta[s][j] += scaled;
        delta[d][j] -= scaled;
        diag[s][j] += weight[e][j];
        diag[d][j] += weight[e][j];
      }
    }

    pose[0][0] = 0.0f;
    pose[0][1] = 0.0f;
    pose[0][2] = 0.0f;
    for (int i = 1; i < POSES; ++i) {
      for (int j = 0; j < 3; ++j) {
        pose[i][j] -= 0.28f * delta[i][j] / diag[i][j];
      }
    }
  }
}

static int summarize(float pose[POSES][3]) {
  int total = 0;
  for (int i = 0; i < POSES; ++i) {
    total += (int)(pose[i][0] * 101.0f);
    total += (int)(pose[i][1] * 173.0f);
    total += (int)(pose[i][2] * 257.0f);
  }
  return abs_f((float)total) > 0.0f ? total : 1;
}

#ifndef EXAMPLE_REFERENCE
extern "C" {
  int bench_input_0[EDGES];
  int bench_input_1[EDGES];
  float bench_input_2[EDGES][3];
  float bench_input_3[EDGES][3];
  float bench_input_4[POSES][3];
}
#endif

int main() {
  int src[EDGES];
  int dst[EDGES];
  float meas[EDGES][3];
  float weight[EDGES][3];
  float pose[POSES][3];

#ifdef EXAMPLE_REFERENCE
  build_pose_graph(src, dst, meas, weight);
  init_state(pose);
#else
  for (int stage_0=0; stage_0<EDGES; ++stage_0)
    src[stage_0]=bench_input_0[stage_0];
  for (int stage_0=0; stage_0<EDGES; ++stage_0)
    dst[stage_0]=bench_input_1[stage_0];
  for (int stage_0=0; stage_0<EDGES; ++stage_0)
  for (int stage_1=0; stage_1<3; ++stage_1)
    meas[stage_0][stage_1]=bench_input_2[stage_0][stage_1];
  for (int stage_0=0; stage_0<EDGES; ++stage_0)
  for (int stage_1=0; stage_1<3; ++stage_1)
    weight[stage_0][stage_1]=bench_input_3[stage_0][stage_1];
  for (int stage_0=0; stage_0<POSES; ++stage_0)
  for (int stage_1=0; stage_1<3; ++stage_1)
    pose[stage_0][stage_1]=bench_input_4[stage_0][stage_1];
#endif

  solve_pose_graph(pose, src, dst, meas, weight);
  EXAMPLE_CAPTURE();
  return summarize(pose);
}
