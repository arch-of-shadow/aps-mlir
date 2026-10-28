#include <stdint.h>
static uint32_t expected_0[24];
static uint32_t expected_1[72];
static uint32_t expected_2[192];
static uint32_t expected_3[192];
static uint32_t expected_4[384];
extern "C" void *memcpy(void *dest, const void *source, __SIZE_TYPE__ n) {
  volatile unsigned char *d=(volatile unsigned char *)dest;
  const volatile unsigned char *s=(const volatile unsigned char *)source;
  for(__SIZE_TYPE__ i=0;i<n;++i)d[i]=s[i];
  return dest;
}
#include "reference.cpp"


volatile int test_status, checked_outputs, validation_calls, mismatches;
volatile int first_mismatch = -1;
volatile uint32_t first_actual, first_expected;
extern "C" int pcl_selected_main();
extern "C" volatile uint64_t tohost, fromhost;
static void report_mismatch() {
  char text[128]; int n=0;
  const char *label="APS_MISMATCH index actual expected checked (hex): ";
  while(*label)text[n++]=*label++;
  uint32_t fields[4]={(uint32_t)first_mismatch,first_actual,first_expected,(uint32_t)checked_outputs};
  for(int j=0;j<4;++j) {
    for(int shift=28;shift>=0;shift-=4) {
      unsigned digit=(fields[j]>>shift)&15;
      text[n++]=digit<10?'0'+digit:'a'+digit-10;
    }
    text[n++]=j==3?'\n':' ';
  }
  alignas(64) volatile uint64_t request[8]={64,1,(uintptr_t)text,(uint32_t)n,0,0,0,0};
  asm volatile("fence rw,rw" ::: "memory");
  tohost=(uintptr_t)request;
  while(!fromhost){}
  fromhost=0;
}
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
extern "C" void pcl_validate_outputs(const void *buffer_0, const void *buffer_1, const void *buffer_2, const void *buffer_3, const void *buffer_4) {
  ++validation_calls;
  check_words(buffer_0, expected_0, 24);
  check_words(buffer_1, expected_1, 72);
  check_words(buffer_2, expected_2, 192);
  check_words(buffer_3, expected_3, 192);
  check_words(buffer_4, expected_4, 384);
}
extern "C" int main() {
  int expected=reference_main();
  
  int actual=pcl_selected_main();
  if(validation_calls!=1 || checked_outputs!=864) return 3;
  if(mismatches) {report_mismatch(); return 1;}
  if(actual!=expected) return 2;
  return 0;
}
