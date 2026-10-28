static void audit_report();
#define main knn_reference_unused
#include "/newdisk/aps-mlir/build/aps-knn-32/input/reference.cpp"
#undef main
#include <stdint.h>

static float expected_points[POINTS][3], expected_queries[QUERIES][3];
static float expected_dist[QUERIES][TOPK];
static int expected_idx[QUERIES][TOPK];
volatile int test_status, checked_outputs, validation_calls, mismatches;
volatile int first_mismatch = -1;
volatile uint32_t first_actual, first_expected;

extern "C" volatile uint64_t tohost, fromhost;
static uint32_t cycles() {
 uint32_t x;
 asm volatile("fence rw,rw; csrr %0,mcycle" : "=r"(x) :: "memory");
 return x;
}
static void print_cycles(uint32_t cold, uint32_t warm) {
 char text[96]; int n=0;
 const char *label="APS_CYCLES cold=";
 while(*label) text[n++]=*label++;
 uint32_t counts[2]={cold,warm};
 for(int j=0;j<2;++j) {
  char digits[10]; int k=0; uint32_t x=counts[j];
  do {digits[k++]='0'+x%10; x/=10;} while(x);
  while(k) text[n++]=digits[--k];
  if(!j) {label=" warm=";while(*label)text[n++]=*label++;}
 }
 text[n++]='\n';
 alignas(64) volatile uint64_t request[8]={64,1,(uintptr_t)text,(uint32_t)n,0,0,0,0};
 asm volatile("fence rw,rw" ::: "memory");
 tohost=(uintptr_t)request;
 while(!fromhost) {}
 fromhost=0;
 asm volatile("fence rw,rw" ::: "memory");
}

static uint32_t end_cycle;
extern "C" int pcl_selected_main();

static uint32_t bits(float value) {
  uint32_t result;
  __builtin_memcpy(&result, &value, 4);
  return result;
}
static void check(uint32_t actual, uint32_t expected) {
  if (actual != expected) {
    if (first_mismatch < 0) {
      first_mismatch = checked_outputs;
      first_actual = actual;
      first_expected = expected;
    }
    ++mismatches;
  }
  ++checked_outputs;
}
extern "C" void pcl_validate_outputs(float *points, float *queries,
                                      float *dist, int *idx) {
 ((volatile unsigned char *)idx)[0] ^= 1;

  end_cycle=cycles();
 ++validation_calls;
  for (int i = 0; i < POINTS; ++i)
    for (int k = 0; k < 3; ++k)
      check(bits(points[i*3+k]), bits(expected_points[i][k]));
  for (int q = 0; q < QUERIES; ++q) {
    for (int k = 0; k < 3; ++k)
      check(bits(queries[q*3+k]), bits(expected_queries[q][k]));
    for (int k = 0; k < TOPK; ++k) {
      check(bits(dist[q*TOPK+k]), bits(expected_dist[q][k]));
      check(idx[q*TOPK+k], expected_idx[q][k]);
    }
  }

 audit_report();
}
extern "C" int main() {
  for (int i = 0; i < POINTS; ++i) {
    expected_points[i][0] = (float)((i*11+3)%101)*0.031f;
    expected_points[i][1] = (float)((i*17+5)%97)*0.029f;
    expected_points[i][2] = (float)((i*19+7)%89)*0.027f;
  }
  for (int i = 0; i < QUERIES; ++i) {
    expected_queries[i][0] = (float)((i*23+9)%101)*0.031f;
    expected_queries[i][1] = (float)((i*29+13)%97)*0.029f;
    expected_queries[i][2] = (float)((i*31+17)%89)*0.027f;
  }
  knn_search(expected_points, expected_queries, expected_dist, expected_idx);
  int expected = summarize(expected_dist, expected_idx);
   uint32_t start_cycle=cycles();
 int actual=pcl_selected_main();
 uint32_t cold=end_cycle-start_cycle;
 if(actual!=expected || mismatches)return 2;
 start_cycle=cycles();
 actual=pcl_selected_main();
 uint32_t warm=end_cycle-start_cycle;
 print_cycles(cold,warm);
  if(validation_calls!=2 || checked_outputs!=344) return 3;
  if (mismatches) return 1;
  if (actual != expected) return 2;
  return 0;
}

static void audit_report() {
 char text[128]; int n=0;
 const char *label="APS_RESULT_CHECK checked/mismatches/first=";
 while(*label)text[n++]=*label++;
 uint32_t values[3]={(uint32_t)checked_outputs,(uint32_t)mismatches,(uint32_t)first_mismatch};
 const char *hex="0123456789abcdef";
 for(int j=0;j<3;++j) {
  for(int k=7;k>=0;--k)text[n++]=hex[(values[j]>>(k*4))&15];
  text[n++]=j==2?'\n':'/';
 }
 alignas(64) volatile uint64_t request[8]={64,1,(uintptr_t)text,(uint32_t)n,0,0,0,0};
 asm volatile("fence rw,rw" ::: "memory");
 tohost=(uintptr_t)request;
 while(!fromhost){}
 fromhost=0;
 asm volatile("fence rw,rw" ::: "memory");
}
