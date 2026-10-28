#include <stdint.h>
static uint32_t expected_0[64];
static uint32_t expected_1[64];
static uint32_t expected_2[64];
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
extern "C" void pcl_validate_outputs(const void *buffer_0, const void *buffer_1, const void *buffer_2) {
  ++validation_calls;
  check_words(buffer_0, expected_0, 64);
  check_words(buffer_1, expected_1, 64);
  check_words(buffer_2, expected_2, 64);
}
extern "C" int main() {
  int expected=reference_main();
  
  int actual=pcl_selected_main();
  if(validation_calls!=1 || checked_outputs!=192) return 3;
  if(mismatches) return 1;
  if(actual!=expected) return 2;
  return 0;
}
