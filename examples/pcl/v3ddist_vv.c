// Auto-generated from CADL by cadl-to-c
// DO NOT EDIT - Regenerate from CADL source

#include <stdint.h>

uint8_t v3ddist_vv(uint32_t *dist_out, uint32_t *points1_x, uint32_t *points1_y, uint32_t *points1_z, uint32_t *points2_x, uint32_t *points2_y, uint32_t *points2_z, uint32_t rd_value, uint32_t rs1_value, uint32_t rs2_value) {
    uint8_t rd_result = 0;
    uint32_t addr1 = rs1_value;
    uint32_t addr2 = rs2_value;
    uint32_t vl = 16;
    // burst_read eliminated (arrays directly accessible)
    // burst_read eliminated (arrays directly accessible)
    // burst_read eliminated (arrays directly accessible)
    // burst_read eliminated (arrays directly accessible)
    // burst_read eliminated (arrays directly accessible)
    // burst_read eliminated (arrays directly accessible)
    uint32_t i;
    for (i = 0; i < vl; ++i) {
        uint32_t x1 = points1_x[i];
        uint32_t y1 = points1_y[i];
        uint32_t z1 = points1_z[i];
        uint32_t x2 = points2_x[i];
        uint32_t y2 = points2_y[i];
        uint32_t z2 = points2_z[i];
        uint32_t dx = (x1 - x2);
        uint32_t dy = (y1 - y2);
        uint32_t dz = (z1 - z2);
        uint32_t dist_sq = (((dx * dx) + (dy * dy)) + (dz * dz));
        dist_out[i] = dist_sq;
        uint32_t i_ = (i + 1);
    }
    uint32_t out_addr = rd_value;
    // burst_write eliminated (arrays directly accessible)
    rd_result = 0;
    return rd_result;
}
