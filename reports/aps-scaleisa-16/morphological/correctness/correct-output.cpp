static void audit_report();
#define main morphological_reference_unused
#include "/newdisk/aps-mlir/build/aps-morphological-12/input/reference.cpp"
#undef main
#include <stdint.h>

static float expected_base[HEIGHT][WIDTH], expected_tmp0[HEIGHT][WIDTH];
static float expected_tmp1[HEIGHT][WIDTH], expected_out[HEIGHT][WIDTH];
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

static void check(float actual, float expected) {
  if (bits(actual) != bits(expected)) {
    if (first_mismatch < 0) {
      first_mismatch = checked_outputs;
      first_actual = bits(actual);
      first_expected = bits(expected);
    }
    ++mismatches;
  }
  ++checked_outputs;
}

extern "C" void pcl_validate_outputs(float *base, float *tmp0,
                                      float *tmp1, float *out) {
  end_cycle=cycles();
 ++validation_calls;
  for (int y = 0; y < HEIGHT; ++y)
    for (int x = 0; x < WIDTH; ++x) {
      int i = y * WIDTH + x;
      check(base[i], expected_base[y][x]);
      check(tmp0[i], expected_tmp0[y][x]);
      check(tmp1[i], expected_tmp1[y][x]);
      check(out[i], expected_out[y][x]);
    }

 audit_report();
}

extern "C" int main() {
  init_height(expected_base);
  open_close_pipeline(expected_base, expected_tmp0, expected_tmp1, expected_out);
  int expected = summarize(expected_out);
   uint32_t start_cycle=cycles();
 int actual=pcl_selected_main();
 uint32_t cold=end_cycle-start_cycle;
 if(actual!=expected || mismatches)return 2;
 start_cycle=cycles();
 actual=pcl_selected_main();
 uint32_t warm=end_cycle-start_cycle;
 print_cycles(cold,warm);
  if (validation_calls!=2 || checked_outputs!=1152) return 3;
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
