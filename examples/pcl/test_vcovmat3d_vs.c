// Test program for VCOVMAT3D_VS instruction
// Tests accumulated covariance matrix computation for 3D point clouds
// Verifies correct outer product accumulation with centroid-based centering

#include "marchid.h"
#include <riscv-pk/encoding.h>
#include <stdint.h>
#include <stdio.h>

// RISC-V custom instruction wrapper for VCOVMAT3D.VS
// Opcode: 0x2B (0101011), Funct7: 0x2B (0101011)
// rs1: address of point cloud + centroid data
// rs2: address of output covariance matrix
// rd: receives return status
__attribute__((always_inline)) uint32_t vcovmat3d_vs(uint32_t points_centroid_addr, uint32_t output_addr) {
    uint32_t result = 0;
    asm volatile(".insn r 0x2B, 0b111, 0x2B, %0, %1, %2"
                 : "=r"(result)
                 : "r"(points_centroid_addr), "r"(output_addr));
    return result;
}

// Point cloud in SOA (Structure of Arrays) format WITH appended centroid
// Memory layout: [X[16] | Y[16] | Z[16] | cx | cy | cz]
// Total: 64 + 64 + 64 + 4 + 4 + 4 = 204 bytes
typedef struct {
    int32_t x[16];     // X coordinates (64 bytes)
    int32_t y[16];     // Y coordinates (64 bytes)
    int32_t z[16];     // Z coordinates (64 bytes)
    int32_t cx;        // Centroid X (4 bytes)
    int32_t cy;        // Centroid Y (4 bytes)
    int32_t cz;        // Centroid Z (4 bytes)
} PointCloudWithCentroid;

// Covariance matrix output (symmetric 3x3, upper triangle + metadata)
typedef struct {
    int32_t c00;      // σ_xx
    int32_t c01;      // σ_xy
    int32_t c02;      // σ_xz
    int32_t c11;      // σ_yy
    int32_t c12;      // σ_yz
    int32_t c22;      // σ_zz
    int32_t count;    // Number of points
    int32_t reserved; // Reserved field
} CovarianceMatrix;

// Software reference implementation
void compute_covariance_reference(const PointCloudWithCentroid *data, CovarianceMatrix *cov) {
    // Use 64-bit accumulator to prevent overflow
    int64_t c00 = 0, c01 = 0, c02 = 0;
    int64_t c11 = 0, c12 = 0, c22 = 0;

    for (int i = 0; i < 16; i++) {
        int64_t dx = data->x[i] - data->cx;
        int64_t dy = data->y[i] - data->cy;
        int64_t dz = data->z[i] - data->cz;

        c00 += dx * dx;
        c01 += dx * dy;
        c02 += dx * dz;
        c11 += dy * dy;
        c12 += dy * dz;
        c22 += dz * dz;
    }

    cov->c00 = (int32_t)c00;
    cov->c01 = (int32_t)c01;
    cov->c02 = (int32_t)c02;
    cov->c11 = (int32_t)c11;
    cov->c12 = (int32_t)c12;
    cov->c22 = (int32_t)c22;
    cov->count = 16;
    cov->reserved = 0;
}

// Helper: compute centroid from points
void compute_centroid(PointCloudWithCentroid *data) {
    int64_t sum_x = 0, sum_y = 0, sum_z = 0;
    for (int i = 0; i < 16; i++) {
        sum_x += data->x[i];
        sum_y += data->y[i];
        sum_z += data->z[i];
    }
    data->cx = sum_x / 16;
    data->cy = sum_y / 16;
    data->cz = sum_z / 16;
}

// Helper: print covariance matrix
void print_covariance(const CovarianceMatrix *cov, const char *label) {
    printf("%s:\n", label);
    printf("  [%10d %10d %10d]\n", cov->c00, cov->c01, cov->c02);
    printf("  [%10d %10d %10d]\n", cov->c01, cov->c11, cov->c12);
    printf("  [%10d %10d %10d]\n", cov->c02, cov->c12, cov->c22);
    printf("  Count: %d\n", cov->count);
}

// Helper: compare and verify results
int verify_covariance(const CovarianceMatrix *hw, const CovarianceMatrix *ref,
                      const char *test_name) {
    int passed = 1;
    const char *fields[] = {"c00", "c01", "c02", "c11", "c12", "c22"};
    int32_t hw_vals[] = {hw->c00, hw->c01, hw->c02, hw->c11, hw->c12, hw->c22};
    int32_t ref_vals[] = {ref->c00, ref->c01, ref->c02, ref->c11, ref->c12, ref->c22};

    for (int i = 0; i < 6; i++) {
        if (hw_vals[i] != ref_vals[i]) {
            printf("  ✗ %s: expected %d, got %d\n", fields[i], ref_vals[i], hw_vals[i]);
            passed = 0;
        }
    }

    if (passed) {
        printf("✓ %s PASSED\n", test_name);
    } else {
        printf("✗ %s FAILED\n", test_name);
    }

    return passed;
}

// Test 1: Points centered at origin
volatile PointCloudWithCentroid test1_data __attribute__((aligned(128))) = {
    .x = {-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1},
    .y = {-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1},
    .z = {-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1},
    .cx = 0, .cy = 0, .cz = 0
};
volatile CovarianceMatrix test1_output __attribute__((aligned(128)));

int test_centered_at_origin() {
    printf("\n=== Test 1: Points centered at origin ===\n");

    CovarianceMatrix ref;
    compute_covariance_reference((const PointCloudWithCentroid *)&test1_data, &ref);

    // Call hardware instruction
    volatile uint32_t result;
    for (int i = 0; i < 10 ; i++) {
        result = vcovmat3d_vs(
        (uint32_t)(unsigned long)&test1_data,
        (uint32_t)(unsigned long)&test1_output);
    }
    printf("Hardware returned: %u\n", result);
    printf("0x%p", (void *)&test1_output);
    print_covariance(&ref, "Reference");
    print_covariance((const CovarianceMatrix *)&test1_output, "Hardware");

    return verify_covariance((const CovarianceMatrix *)&test1_output, &ref, "Test 1");
}

// Test 2: Cube vertices
volatile PointCloudWithCentroid test2_data __attribute__((aligned(128))) = {
    .x = {0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10},
    .y = {0, 0, 10, 10, 0, 0, 10, 10, 0, 0, 10, 10, 0, 0, 10, 10},
    .z = {0, 0, 0, 0, 10, 10, 10, 10, 0, 0, 0, 0, 10, 10, 10, 10},
    .cx = 5, .cy = 5, .cz = 5
};
volatile CovarianceMatrix test2_output __attribute__((aligned(128)));

int test_cube_vertices() {
    printf("\n=== Test 2: Cube vertices (diagonal covariance) ===\n");

    // Compute centroid first
    compute_centroid((PointCloudWithCentroid *)&test2_data);
    printf("Centroid: (%d, %d, %d)\n", test2_data.cx, test2_data.cy, test2_data.cz);

    CovarianceMatrix ref;
    compute_covariance_reference((const PointCloudWithCentroid *)&test2_data, &ref);

    // Call hardware instruction
    uint32_t result = vcovmat3d_vs(
        (uint32_t)(unsigned long)&test2_data,
        (uint32_t)(unsigned long)&test2_output);

    print_covariance(&ref, "Reference");
    print_covariance((const CovarianceMatrix *)&test2_output, "Hardware");

    return verify_covariance((const CovarianceMatrix *)&test2_output, &ref, "Test 2");
}

// Test 3: Linear correlation (y = x)
volatile PointCloudWithCentroid test3_data __attribute__((aligned(128)));
volatile CovarianceMatrix test3_output __attribute__((aligned(128)));

int test_linear_correlation() {
    printf("\n=== Test 3: Linear correlation (y = x) ===\n");

    // Initialize points on line y = x
    for (int i = 0; i < 16; i++) {
        ((int32_t *)test3_data.x)[i] = i * 10;
        ((int32_t *)test3_data.y)[i] = i * 10; // Perfect correlation
        ((int32_t *)test3_data.z)[i] = 5;      // Constant Z
    }

    compute_centroid((PointCloudWithCentroid *)&test3_data);
    printf("Centroid: (%d, %d, %d)\n", test3_data.cx, test3_data.cy, test3_data.cz);

    CovarianceMatrix ref;
    compute_covariance_reference((const PointCloudWithCentroid *)&test3_data, &ref);

    uint32_t result = vcovmat3d_vs(
        (uint32_t)(unsigned long)&test3_data,
        (uint32_t)(unsigned long)&test3_output);

    print_covariance(&ref, "Reference");
    print_covariance((const CovarianceMatrix *)&test3_output, "Hardware");
    printf("Expected: c00 ≈ c11 (equal variance), c01 ≈ c00 (perfect correlation)\n");

    return verify_covariance((const CovarianceMatrix *)&test3_output, &ref, "Test 3");
}

// Test 4: All identical points (zero covariance)
volatile PointCloudWithCentroid test4_data __attribute__((aligned(128)));
volatile CovarianceMatrix test4_output __attribute__((aligned(128)));

int test_identical_points() {
    printf("\n=== Test 4: All identical points (zero covariance) ===\n");

    for (int i = 0; i < 16; i++) {
        ((int32_t *)test4_data.x)[i] = 100;
        ((int32_t *)test4_data.y)[i] = 200;
        ((int32_t *)test4_data.z)[i] = 300;
    }
    ((PointCloudWithCentroid *)&test4_data)->cx = 100;
    ((PointCloudWithCentroid *)&test4_data)->cy = 200;
    ((PointCloudWithCentroid *)&test4_data)->cz = 300;

    CovarianceMatrix ref;
    compute_covariance_reference((const PointCloudWithCentroid *)&test4_data, &ref);

    uint32_t result = vcovmat3d_vs(
        (uint32_t)(unsigned long)&test4_data,
        (uint32_t)(unsigned long)&test4_output);

    print_covariance(&ref, "Reference");
    print_covariance((const CovarianceMatrix *)&test4_output, "Hardware");
    printf("Expected: All zeros (no variance)\n");

    return verify_covariance((const CovarianceMatrix *)&test4_output, &ref, "Test 4");
}

// Test 5: Planar points (all Z = 0)
volatile PointCloudWithCentroid test5_data __attribute__((aligned(128)));
volatile CovarianceMatrix test5_output __attribute__((aligned(128)));

int test_planar_points() {
    printf("\n=== Test 5: Planar points (all on XY plane) ===\n");

    for (int i = 0; i < 16; i++) {
        ((int32_t *)test5_data.x)[i] = (i % 4) * 10;
        ((int32_t *)test5_data.y)[i] = (i / 4) * 10;
        ((int32_t *)test5_data.z)[i] = 0; // All on Z=0 plane
    }

    compute_centroid((PointCloudWithCentroid *)&test5_data);

    CovarianceMatrix ref;
    compute_covariance_reference((const PointCloudWithCentroid *)&test5_data, &ref);

    uint32_t result = vcovmat3d_vs(
        (uint32_t)(unsigned long)&test5_data,
        (uint32_t)(unsigned long)&test5_output);

    print_covariance(&ref, "Reference");
    print_covariance((const CovarianceMatrix *)&test5_output, "Hardware");
    printf("Expected: c02 = c12 = c22 = 0 (no Z variance)\n");

    return verify_covariance((const CovarianceMatrix *)&test5_output, &ref, "Test 5");
}

int main(void) {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║  VCOVMAT3D_VS Custom Instruction Test Suite               ║\n");
    printf("║  Tests accumulated covariance for 16-point batches        ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");

    uint64_t marchid = read_csr(marchid);
    const char *march = get_march(marchid);
    printf("Running on: %s\n", march);

    int total_tests = 5;
    int passed_tests = 0;

    passed_tests += test_centered_at_origin();
    passed_tests += test_cube_vertices();
    passed_tests += test_linear_correlation();
    passed_tests += test_identical_points();
    passed_tests += test_planar_points();

    printf("\n╔════════════════════════════════════════════════════════════╗\n");
    printf("║  Test Summary                                              ║\n");
    printf("╠════════════════════════════════════════════════════════════╣\n");
    printf("║  Total tests:  %2d                                          ║\n", total_tests);
    printf("║  Passed:       %2d                                          ║\n", passed_tests);
    printf("║  Failed:       %2d                                          ║\n", total_tests - passed_tests);
    printf("╚════════════════════════════════════════════════════════════╝\n");

    if (passed_tests == total_tests) {
        printf("\n✓ All tests PASSED - VCOVMAT3D_VS accumulation working correctly!\n");
    } else {
        printf("\n✗ Some tests FAILED - check hardware implementation\n");
    }

    return (passed_tests == total_tests) ? 0 : 1;
}
