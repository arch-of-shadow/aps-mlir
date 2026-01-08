#include <stdint.h>
#include <stdio.h>
#include "marchid.h"
#include <riscv-pk/encoding.h>

#pragma megg optimize
uint8_t vgemv3d_vv(int32_t *rs1, int32_t *rs2) {
    int32_t acc = 0;

    uint8_t rd_result = 0;
    int32_t * addr = rs1;
    int32_t * out_addr = rs2;
    // burst_read lowered via register-backed scratchpad
    // burst_read lowered via register-backed scratchpad
    uint32_t i;
    for (i = 0; i < 4; ++i) {
        acc = 0;
        uint32_t j;
        for (j = 0; j < 4; ++j) {
            acc = ((rs1[((i << 2) + j)] * rs1[16 + j]) + acc);
            uint32_t j_ = (j + 1);
        }
        rs2[i] = acc;
        uint32_t i_ = (i + 1);
    }
    // burst_write lowered via register-backed scratchpad
    rd_result = 0;
    return rd_result;
}


// Input data: 4x4 matrix + 4-element vector
volatile int32_t input_data[20] __attribute__((aligned(128))) = {
    // 4x4 matrix (row-major)
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16,

    // 4-element vector
    1, 1, 1, 1
};

// Output: 4-element result vector
volatile int32_t output_data[4] __attribute__((aligned(128))) = {0};

int main(void) {
  printf("VGEMV3D Test - General Matrix-Vector Multiply\n");
  printf("Input data address: 0x%lx\n", (unsigned long)input_data);
  printf("Output data address: 0x%lx\n", (unsigned long)output_data);

  uint64_t marchid = read_csr(marchid);
  const char *march = get_march(marchid);
  printf("Running on: %s\n\n", march);

  // Call custom instruction
  volatile uint32_t result = 0;
  for (int i = 0; i < 10; i++) {
    result = vgemv3d_vv((int32_t *)input_data, (int32_t *)output_data);
  }

  printf("Result: %u\n", result);
  printf("Output vector: [%d, %d, %d, %d]\n",
         output_data[0], output_data[1], output_data[2], output_data[3]);

  return 0;
}
