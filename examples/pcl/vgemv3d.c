// Auto-generated from CADL by cadl-to-c
// DO NOT EDIT - Regenerate from CADL source

#include <stdint.h>

uint8_t vgemv3d_vv(int32_t *acc, int32_t *matrix, int32_t *result, int32_t *vec, uint32_t rd_value, uint32_t rs1_value, uint32_t rs2_value) {
    uint8_t rd_result = 0;
    uint32_t mat_addr = rs1_value;
    uint32_t vec_addr = rs2_value;
    // burst_read eliminated (arrays directly accessible)
    // burst_read eliminated (arrays directly accessible)
    uint32_t i;
    for (i = 0; i < 4; ++i) {
        (*acc) = 0;
        uint32_t j;
        for (j = 0; j < 4; ++j) {
            (*acc) = ((*acc) + (matrix[((i * 4) + j)] * vec[j]));
            uint32_t j_ = (j + 1);
        }
        result[i] = (*acc);
        uint32_t i_ = (i + 1);
    }
    uint32_t out_addr = rd_value;
    // burst_write eliminated (arrays directly accessible)
    rd_result = 0;
    return rd_result;
}
