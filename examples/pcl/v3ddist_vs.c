// Auto-generated from CADL by cadl-to-c
// DO NOT EDIT - Regenerate from CADL source

#include <stdint.h>
#include <string.h>

uint8_t v3ddist_vs(uint32_t *dist_out, uint32_t *points_x, uint32_t *points_y, uint32_t *points_z, uint32_t *_mem, uint32_t rd_value, uint32_t rs1_value, uint32_t rs2_value) {
    uint8_t rd_result = 0;
    uint32_t addr = rs1_value;
    uint32_t ref_addr = rs2_value;
    uint32_t vl = 16;
    // burst_read eliminated (arrays directly accessible)
    // burst_read eliminated (arrays directly accessible)
    // burst_read eliminated (arrays directly accessible)
    uint32_t ref_x = _mem[ref_addr];
    uint32_t ref_y = _mem[(ref_addr + 4)];
    uint32_t ref_z = _mem[(ref_addr + 8)];
    uint32_t i;
    for (i = 0; i < vl; ++i) {
        uint32_t x = points_x[i];
        uint32_t y = points_y[i];
        uint32_t z = points_z[i];
        uint32_t dx = (x - ref_x);
        uint32_t dy = (y - ref_y);
        uint32_t dz = (z - ref_z);
        uint32_t dist_sq = (((dx * dx) + (dy * dy)) + (dz * dz));
        dist_out[i] = dist_sq;
        uint32_t i_ = (i + 1);
    }
    uint32_t out_addr = rd_value;
    // burst_write eliminated (arrays directly accessible)
    rd_result = 0;
    return rd_result;
}
