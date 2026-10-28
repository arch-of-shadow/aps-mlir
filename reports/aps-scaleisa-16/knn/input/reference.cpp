static const int POINTS = 32;
static const int QUERIES = 4;
static const int TOPK = 8;

static float abs_f(float x) {
  return x < 0.0f ? -x : x;
}

static float sq_dist(float a[3], float b[3]) {
  float dx = a[0] - b[0];
  float dy = a[1] - b[1];
  float dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

static void knn_search(
    float points[POINTS][3],
    float queries[QUERIES][3],
    float best_dist[QUERIES][TOPK],
    int best_idx[QUERIES][TOPK]) {
  float next_dist[TOPK];
  int next_idx[TOPK];
  for (int q = 0; q < QUERIES; ++q) {
    for (int k = 0; k < TOPK; ++k) {
      best_dist[q][k] = 1.0e30f;
      best_idx[q][k] = -1;
    }
    for (int i = 0; i < POINTS; ++i) {
      float d = sq_dist(points[i], queries[q]);
      int inserted = 0;
      for (int k = 0; k < TOPK; ++k) {
        if (inserted == 0 && d < best_dist[q][k]) {
          next_dist[k] = d;
          next_idx[k] = i;
          inserted = 1;
        } else {
          int src = k - inserted;
          next_dist[k] = best_dist[q][src];
          next_idx[k] = best_idx[q][src];
        }
      }
      for (int k = 0; k < TOPK; ++k) {
        best_dist[q][k] = next_dist[k];
        best_idx[q][k] = next_idx[k];
      }
    }
  }
}

static int summarize(float best_dist[QUERIES][TOPK], int best_idx[QUERIES][TOPK]) {
  int total = 0;
  for (int q = 0; q < QUERIES; ++q) {
    for (int k = 0; k < TOPK; ++k) {
      total += best_idx[q][k] >= 0 ? best_idx[q][k] + (int)(best_dist[q][k] * 256.0f) : 0;
    }
  }
  return abs_f((float)total) > 0.0f ? total : 1;
}

int main() {
  float points[POINTS][3];
  float queries[QUERIES][3];
  float best_dist[QUERIES][TOPK];
  int best_idx[QUERIES][TOPK];

  for (int i = 0; i < POINTS; ++i) {
    points[i][0] = (float)((i * 11 + 3) % 101) * 0.031f;
    points[i][1] = (float)((i * 17 + 5) % 97) * 0.029f;
    points[i][2] = (float)((i * 19 + 7) % 89) * 0.027f;
  }
  for (int i = 0; i < QUERIES; ++i) {
    queries[i][0] = (float)((i * 23 + 9) % 101) * 0.031f;
    queries[i][1] = (float)((i * 29 + 13) % 97) * 0.029f;
    queries[i][2] = (float)((i * 31 + 17) % 89) * 0.027f;
  }
  knn_search(points, queries, best_dist, best_idx);
  return summarize(best_dist, best_idx);
}
