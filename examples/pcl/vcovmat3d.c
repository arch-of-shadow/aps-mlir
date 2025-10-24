// Auto-generated from CADL by cadl-to-c
// DO NOT EDIT - Regenerate from CADL source

#include <stdint.h>

uint8_t vcovmat3d_vv(int32_t *cov_out, int32_t *points, uint8_t rs2, uint32_t rd_value, uint32_t rs1_value) {
    uint8_t rd_result = 0;
    uint32_t addr = rs1_value;
    // burst_read eliminated (arrays directly accessible)
    int32_t x = points[0];
    int32_t y = points[1];
    int32_t z = points[2];
    int32_t cx = points[3];
    int32_t cy = points[4];
    int32_t cz = points[5];
    int32_t dx = (x - cx);
    int32_t dy = (y - cy);
    int32_t dz = (z - cz);
    cov_out[0] = (dx * dx);
    cov_out[1] = (dx * dy);
    cov_out[2] = (dx * dz);
    cov_out[3] = (dy * dy);
    cov_out[4] = (dy * dz);
    cov_out[5] = (dz * dz);
    uint32_t out_addr = rd_value;
    // burst_write eliminated (arrays directly accessible)
    rd_result = 0;
    return rd_result;
}

// right function for instruction use

uint8_t vcovmat3d_vv(uint32_t *rs1, uint32_t *rd) {
    // rs1: points base
    // rd: output covariance matrix base
    int32_t x = rs1[0];
    int32_t y = rs1[1];
    int32_t z = rs1[2];
    int32_t cx = rs1[3];
    int32_t cy = rs1[4];
    int32_t cz = rs1[5];
    int32_t dx = (x - cx);
    int32_t dy = (y - cy);
    int32_t dz = (z - cz);
    rd[0] = (dx * dx);
    rd[1] = (dx * dy);
    rd[2] = (dx * dz);
    rd[3] = (dy * dy);
    rd[4] = (dy * dz);
    rd[5] = (dz * dz);
    return 0;
}
