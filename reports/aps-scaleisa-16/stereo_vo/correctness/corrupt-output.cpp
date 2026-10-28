static void audit_report();
#include <stdint.h>
static uint32_t expected_0[3];
static uint32_t expected_1[96];
static uint32_t expected_2[96];
static uint32_t expected_3[96];
extern "C" void *memcpy(void *dest, const void *source, __SIZE_TYPE__ n) {
  volatile unsigned char *d=(volatile unsigned char *)dest;
  const volatile unsigned char *s=(const volatile unsigned char *)source;
  for(__SIZE_TYPE__ i=0;i<n;++i)d[i]=s[i];
  return dest;
}
#include "/newdisk/aps-mlir/build/aps-suite-16/stereo_vo/input/reference.cpp"
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
static void check_words(const void *actual, const uint32_t *expected, int n) {
  for(int i=0;i<n;++i) {
    uint32_t value;
    __builtin_memcpy(&value, (const char *)actual+4*i, 4);
    if(value!=expected[i]) {
      if(first_mismatch<0) { first_mismatch=checked_outputs; first_actual=value; first_expected=expected[i]; }
      ++mismatches;
    }
    ++checked_outputs;
  }
}
extern "C" void pcl_validate_outputs(const void *buffer_0, const void *buffer_1, const void *buffer_2, const void *buffer_3) {
 ((volatile unsigned char *)buffer_0)[0] ^= 1;

  end_cycle=cycles();
 ++validation_calls;
  check_words(buffer_0, expected_0, 3);
  check_words(buffer_1, expected_1, 96);
  check_words(buffer_2, expected_2, 96);
  check_words(buffer_3, expected_3, 96);

 audit_report();
}
extern "C" int main() {
  int expected=reference_main();
   uint32_t start_cycle=cycles();
 int actual=pcl_selected_main();
 uint32_t cold=end_cycle-start_cycle;
 if(actual!=expected || mismatches)return 2;
 start_cycle=cycles();
 actual=pcl_selected_main();
 uint32_t warm=end_cycle-start_cycle;
 print_cycles(cold,warm);
  if(validation_calls!=2 || checked_outputs!=582) return 3;
  if(mismatches) return 1;
  if(actual!=expected) return 2;
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
