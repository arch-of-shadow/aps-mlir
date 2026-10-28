static const int POINTS = 32;
static const int HYPOTHESES = 4;

static float abs_f(float x) {
  return x < 0.0f ? -x : x;
}

static void init_cloud(float points[POINTS][3]) {
  for (int i = 0; i < POINTS; ++i) {
    if (i < 768) {
      int a = i % 64;
      int b = i / 64;
      points[i][0] = 0.6f + 0.2f * (float)a / 64.0f;
      points[i][1] = 0.4f + 0.2f * (float)b / 12.0f;
      points[i][2] = 0.3f + 0.2f * (float)((a + b) % 31) / 31.0f;
    } else {
      points[i][0] = (float)((i * 13 + 3) % 101) / 101.0f;
      points[i][1] = (float)((i * 17 + 5) % 97) / 97.0f;
      points[i][2] = (float)((i * 19 + 7) % 89) / 89.0f;
    }
  }
}

static void ransac_sphere(
    float points[POINTS][3],
    float best_model[4],
    int inlier_count[HYPOTHESES]) {
  int best_inliers = -1;
  float best_error = 1.0e30f;

  for (int h = 0; h < HYPOTHESES; ++h) {
    int i0 = (h * 7 + 11) % POINTS;
    int i1 = (h * 13 + 19) % POINTS;
    int i2 = (h * 17 + 23) % POINTS;
    int i3 = (h * 29 + 31) % POINTS;
    float cx = 0.25f * (points[i0][0] + points[i1][0] + points[i2][0] + points[i3][0]);
    float cy = 0.25f * (points[i0][1] + points[i1][1] + points[i2][1] + points[i3][1]);
    float cz = 0.25f * (points[i0][2] + points[i1][2] + points[i2][2] + points[i3][2]);
    float radius = 0.0f;
    int inliers = 0;
    float total_error = 0.0f;

    for (int s = 0; s < 4; ++s) {
      int idx = s == 0 ? i0 : (s == 1 ? i1 : (s == 2 ? i2 : i3));
      float dx = points[idx][0] - cx;
      float dy = points[idx][1] - cy;
      float dz = points[idx][2] - cz;
      radius += abs_f(dx) + abs_f(dy) + abs_f(dz);
    }
    radius *= 0.25f;

    for (int i = 0; i < POINTS; ++i) {
      float dx = points[i][0] - cx;
      float dy = points[i][1] - cy;
      float dz = points[i][2] - cz;
      float dist = abs_f(dx) + abs_f(dy) + abs_f(dz);
      float err = abs_f(dist - radius);
      if (err < 0.08f) {
        inliers += 1;
      }
      total_error += err;
    }

    inlier_count[h] = inliers;
    if (inliers > best_inliers || (inliers == best_inliers && total_error < best_error)) {
      best_inliers = inliers;
      best_error = total_error;
      best_model[0] = cx;
      best_model[1] = cy;
      best_model[2] = cz;
      best_model[3] = radius;
    }
  }
}

static int summarize(float best_model[4], int inlier_count[HYPOTHESES]) {
  int total = (int)(best_model[0] * 1000.0f) + (int)(best_model[1] * 1100.0f) +
              (int)(best_model[2] * 1200.0f) + (int)(best_model[3] * 1300.0f);
  for (int i = 0; i < HYPOTHESES; ++i) {
    total += inlier_count[i];
  }
  return total;
}

int reference_main() {
  float points[POINTS][3];
  float best_model[4];
  int inlier_count[HYPOTHESES];

  init_cloud(points);
  ransac_sphere(points, best_model, inlier_count);
  __builtin_memcpy(expected_0, points, sizeof(expected_0));
  __builtin_memcpy(expected_1, best_model, sizeof(expected_1));
  __builtin_memcpy(expected_2, inlier_count, sizeof(expected_2));
  return summarize(best_model, inlier_count);
}
