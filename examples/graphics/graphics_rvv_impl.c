#include <stddef.h>
#include <stdint.h>

/* Define USE_RVV at build time to enable RVV intrinsics. */

#if defined(USE_RVV)
#include <riscv_vector.h>
#define GRAPHICS_RVV_AVAILABLE 1
#else
#define GRAPHICS_RVV_AVAILABLE 0
#endif

#define RGB_PIXELS 16
#define PHONG_PIXELS 16
#define MEAN_VAR_LEN 32

typedef struct {
    int16_t r[RGB_PIXELS];
    int16_t g[RGB_PIXELS];
    int16_t b[RGB_PIXELS];
} Rgb2YuvInput;

typedef struct {
    int16_t y[RGB_PIXELS];
    int16_t u[RGB_PIXELS];
    int16_t v[RGB_PIXELS];
} Rgb2YuvOutput;

typedef struct {
    int16_t normal_x[PHONG_PIXELS];
    int16_t normal_y[PHONG_PIXELS];
    int16_t normal_z[PHONG_PIXELS];
    int16_t view_x[PHONG_PIXELS];
    int16_t view_y[PHONG_PIXELS];
    int16_t view_z[PHONG_PIXELS];
    int16_t light_x;
    int16_t light_y;
    int16_t light_z;
    int16_t light_i;
} PhongInput;

typedef struct {
    int16_t diffuse[PHONG_PIXELS];
    int16_t specular[PHONG_PIXELS];
} PhongOutput;

typedef struct {
    int32_t data[MEAN_VAR_LEN];
} MeanVarInput;

typedef struct {
    int32_t mean_q;
    int32_t var_q;
} MeanVarOutput;

static inline int16_t clamp_u8_i16(int32_t value) {
    if (value < 0) {
        return 0;
    }
    if (value > 255) {
        return 255;
    }
    return (int16_t)value;
}

static void rgb2yuv_fixed_scalar(const Rgb2YuvInput *in_buf, Rgb2YuvOutput *out_buf) {
    size_t idx;
    for (idx = 0; idx < RGB_PIXELS; ++idx) {
        int32_t r = in_buf->r[idx];
        int32_t g = in_buf->g[idx];
        int32_t b = in_buf->b[idx];
        int32_t y_raw = (77 * r + 150 * g + 29 * b) >> 8;
        int32_t u_raw = (((0 - 43) * r) + ((0 - 85) * g) + (128 * b) + (128 << 8)) >> 8;
        int32_t v_raw = ((128 * r) + ((0 - 107) * g) + ((0 - 21) * b) + (128 << 8)) >> 8;
        out_buf->y[idx] = clamp_u8_i16(y_raw);
        out_buf->u[idx] = clamp_u8_i16(u_raw);
        out_buf->v[idx] = clamp_u8_i16(v_raw);
    }
}

static void phong_fixed_scalar(const PhongInput *in_buf, PhongOutput *out_buf) {
    size_t idx;
    for (idx = 0; idx < PHONG_PIXELS; ++idx) {
        int32_t nx = in_buf->normal_x[idx];
        int32_t ny = in_buf->normal_y[idx];
        int32_t nz = in_buf->normal_z[idx];
        int32_t vx = in_buf->view_x[idx];
        int32_t vy = in_buf->view_y[idx];
        int32_t vz = in_buf->view_z[idx];
        int32_t nl = (int32_t)((((int64_t)nx * in_buf->light_x) +
                                ((int64_t)ny * in_buf->light_y) +
                                ((int64_t)nz * in_buf->light_z)) >> 8);
        int32_t nh;
        int32_t nh_pos;
        int32_t spec_sq;

        if (nl < 0) {
            nl = 0;
        }
        out_buf->diffuse[idx] = (int16_t)(((int64_t)nl * in_buf->light_i) >> 8);

        nh = (int32_t)((((int64_t)nx * (vx + in_buf->light_x)) +
                        ((int64_t)ny * (vy + in_buf->light_y)) +
                        ((int64_t)nz * (vz + in_buf->light_z))) >> 8);
        nh_pos = (nh > 0) ? nh : 0;
        spec_sq = (int32_t)(((int64_t)nh_pos * nh_pos) >> 8);
        out_buf->specular[idx] = (int16_t)(((int64_t)spec_sq * in_buf->light_i) >> 8);
    }
}

static void mean_var_fixed_scalar(const MeanVarInput *in_buf, MeanVarOutput *out_buf) {
    int64_t sum_acc = 0;
    int64_t sum_sq_diff_acc = 0;
    size_t idx;

    for (idx = 0; idx < MEAN_VAR_LEN; ++idx) {
        sum_acc += in_buf->data[idx];
    }
    out_buf->mean_q = (int32_t)(sum_acc << 3);

    for (idx = 0; idx < MEAN_VAR_LEN; ++idx) {
        int64_t diff = ((int64_t)in_buf->data[idx] << 8) - out_buf->mean_q;
        sum_sq_diff_acc += (diff * diff) >> 8;
    }
    out_buf->var_q = (int32_t)(sum_sq_diff_acc >> 5);
}

#if GRAPHICS_RVV_AVAILABLE
void rgb2yuv_fixed_rvv(const Rgb2YuvInput *in_buf, Rgb2YuvOutput *out_buf) {
    int32_t y32_buf[RGB_PIXELS];
    int32_t u32_buf[RGB_PIXELS];
    int32_t v32_buf[RGB_PIXELS];
    size_t idx = 0;

    while (idx < RGB_PIXELS) {
        size_t vl = __riscv_vsetvl_e16m1(RGB_PIXELS - idx);
        vint16m1_t vr16 = __riscv_vle16_v_i16m1(&in_buf->r[idx], vl);
        vint16m1_t vg16 = __riscv_vle16_v_i16m1(&in_buf->g[idx], vl);
        vint16m1_t vb16 = __riscv_vle16_v_i16m1(&in_buf->b[idx], vl);
        vint32m2_t vy32 = __riscv_vwmul_vx_i32m2(vr16, 77, vl);
        vint32m2_t vu32 = __riscv_vwmul_vx_i32m2(vr16, -43, vl);
        vint32m2_t vv32 = __riscv_vwmul_vx_i32m2(vr16, 128, vl);

        vy32 = __riscv_vadd_vv_i32m2(vy32, __riscv_vwmul_vx_i32m2(vg16, 150, vl), vl);
        vy32 = __riscv_vadd_vv_i32m2(vy32, __riscv_vwmul_vx_i32m2(vb16, 29, vl), vl);
        vu32 = __riscv_vadd_vv_i32m2(vu32, __riscv_vwmul_vx_i32m2(vg16, -85, vl), vl);
        vu32 = __riscv_vadd_vv_i32m2(vu32, __riscv_vwmul_vx_i32m2(vb16, 128, vl), vl);
        vv32 = __riscv_vadd_vv_i32m2(vv32, __riscv_vwmul_vx_i32m2(vg16, -107, vl), vl);
        vv32 = __riscv_vadd_vv_i32m2(vv32, __riscv_vwmul_vx_i32m2(vb16, -21, vl), vl);

        vu32 = __riscv_vadd_vx_i32m2(vu32, 128 << 8, vl);
        vv32 = __riscv_vadd_vx_i32m2(vv32, 128 << 8, vl);

        vy32 = __riscv_vsra_vx_i32m2(vy32, 8, vl);
        vu32 = __riscv_vsra_vx_i32m2(vu32, 8, vl);
        vv32 = __riscv_vsra_vx_i32m2(vv32, 8, vl);

        vy32 = __riscv_vmin_vx_i32m2(__riscv_vmax_vx_i32m2(vy32, 0, vl), 255, vl);
        vu32 = __riscv_vmin_vx_i32m2(__riscv_vmax_vx_i32m2(vu32, 0, vl), 255, vl);
        vv32 = __riscv_vmin_vx_i32m2(__riscv_vmax_vx_i32m2(vv32, 0, vl), 255, vl);

        __riscv_vse32_v_i32m2(&y32_buf[idx], vy32, vl);
        __riscv_vse32_v_i32m2(&u32_buf[idx], vu32, vl);
        __riscv_vse32_v_i32m2(&v32_buf[idx], vv32, vl);
        for (size_t lane = 0; lane < vl; ++lane) {
            out_buf->y[idx + lane] = (int16_t)y32_buf[idx + lane];
            out_buf->u[idx + lane] = (int16_t)u32_buf[idx + lane];
            out_buf->v[idx + lane] = (int16_t)v32_buf[idx + lane];
        }
        idx += vl;
    }
}

void phong_fixed_rvv(const PhongInput *in_buf, PhongOutput *out_buf) {
    int32_t diff32_buf[PHONG_PIXELS];
    int32_t spec32_buf[PHONG_PIXELS];
    size_t idx = 0;

    while (idx < PHONG_PIXELS) {
        size_t vl = __riscv_vsetvl_e16m1(PHONG_PIXELS - idx);
        vint16m1_t vnx16 = __riscv_vle16_v_i16m1(&in_buf->normal_x[idx], vl);
        vint16m1_t vny16 = __riscv_vle16_v_i16m1(&in_buf->normal_y[idx], vl);
        vint16m1_t vnz16 = __riscv_vle16_v_i16m1(&in_buf->normal_z[idx], vl);
        vint16m1_t vvx16 = __riscv_vle16_v_i16m1(&in_buf->view_x[idx], vl);
        vint16m1_t vvy16 = __riscv_vle16_v_i16m1(&in_buf->view_y[idx], vl);
        vint16m1_t vvz16 = __riscv_vle16_v_i16m1(&in_buf->view_z[idx], vl);
        vint16m1_t vhx16 = __riscv_vadd_vx_i16m1(vvx16, in_buf->light_x, vl);
        vint16m1_t vhy16 = __riscv_vadd_vx_i16m1(vvy16, in_buf->light_y, vl);
        vint16m1_t vhz16 = __riscv_vadd_vx_i16m1(vvz16, in_buf->light_z, vl);
        vint32m2_t vnl32 = __riscv_vwmul_vx_i32m2(vnx16, in_buf->light_x, vl);
        vint32m2_t vnh32 = __riscv_vwmul_vv_i32m2(vnx16, vhx16, vl);
        vint32m2_t vdiff32;
        vint32m2_t vspec32;

        vnl32 = __riscv_vadd_vv_i32m2(vnl32, __riscv_vwmul_vx_i32m2(vny16, in_buf->light_y, vl), vl);
        vnl32 = __riscv_vadd_vv_i32m2(vnl32, __riscv_vwmul_vx_i32m2(vnz16, in_buf->light_z, vl), vl);
        vnh32 = __riscv_vadd_vv_i32m2(vnh32, __riscv_vwmul_vv_i32m2(vny16, vhy16, vl), vl);
        vnh32 = __riscv_vadd_vv_i32m2(vnh32, __riscv_vwmul_vv_i32m2(vnz16, vhz16, vl), vl);

        vnl32 = __riscv_vsra_vx_i32m2(vnl32, 8, vl);
        vnh32 = __riscv_vsra_vx_i32m2(vnh32, 8, vl);
        vnl32 = __riscv_vmax_vx_i32m2(vnl32, 0, vl);
        vnh32 = __riscv_vmax_vx_i32m2(vnh32, 0, vl);

        vdiff32 = __riscv_vsra_vx_i32m2(__riscv_vmul_vx_i32m2(vnl32, in_buf->light_i, vl), 8, vl);
        vspec32 = __riscv_vsra_vx_i32m2(__riscv_vmul_vv_i32m2(vnh32, vnh32, vl), 8, vl);
        vspec32 = __riscv_vsra_vx_i32m2(__riscv_vmul_vx_i32m2(vspec32, in_buf->light_i, vl), 8, vl);
        vdiff32 = __riscv_vmax_vx_i32m2(vdiff32, 0, vl);
        vspec32 = __riscv_vmax_vx_i32m2(vspec32, 0, vl);

        __riscv_vse32_v_i32m2(&diff32_buf[idx], vdiff32, vl);
        __riscv_vse32_v_i32m2(&spec32_buf[idx], vspec32, vl);
        for (size_t lane = 0; lane < vl; ++lane) {
            out_buf->diffuse[idx + lane] = (int16_t)diff32_buf[idx + lane];
            out_buf->specular[idx + lane] = (int16_t)spec32_buf[idx + lane];
        }
        idx += vl;
    }
}

void mean_var_fixed_rvv(const MeanVarInput *in_buf, MeanVarOutput *out_buf) {
    int32_t scratch[MEAN_VAR_LEN];
    int64_t sum_acc = 0;
    int64_t sum_sq_diff_acc = 0;
    size_t idx = 0;

    while (idx < MEAN_VAR_LEN) {
        size_t vl = __riscv_vsetvl_e32m1(MEAN_VAR_LEN - idx);
        vint32m1_t vsamples = __riscv_vle32_v_i32m1(&in_buf->data[idx], vl);
        size_t lane;
        __riscv_vse32_v_i32m1(scratch, vsamples, vl);
        for (lane = 0; lane < vl; ++lane) {
            sum_acc += scratch[lane];
        }
        idx += vl;
    }

    out_buf->mean_q = (int32_t)(sum_acc << 3);
    idx = 0;

    while (idx < MEAN_VAR_LEN) {
        size_t vl = __riscv_vsetvl_e32m1(MEAN_VAR_LEN - idx);
        vint32m1_t vsamples = __riscv_vle32_v_i32m1(&in_buf->data[idx], vl);
        vint32m1_t vdiff = __riscv_vsub_vx_i32m1(__riscv_vmul_vx_i32m1(vsamples, 256, vl),
                                                 out_buf->mean_q,
                                                 vl);
        vint32m1_t vdiff_sq = __riscv_vsra_vx_i32m1(__riscv_vmul_vv_i32m1(vdiff, vdiff, vl), 8, vl);
        size_t lane;
        __riscv_vse32_v_i32m1(scratch, vdiff_sq, vl);
        for (lane = 0; lane < vl; ++lane) {
            sum_sq_diff_acc += scratch[lane];
        }
        idx += vl;
    }

    out_buf->var_q = (int32_t)(sum_sq_diff_acc >> 5);
}
#else
void rgb2yuv_fixed_rvv(const Rgb2YuvInput *in_buf, Rgb2YuvOutput *out_buf) {
    rgb2yuv_fixed_scalar(in_buf, out_buf);
}

void phong_fixed_rvv(const PhongInput *in_buf, PhongOutput *out_buf) {
    phong_fixed_scalar(in_buf, out_buf);
}

void mean_var_fixed_rvv(const MeanVarInput *in_buf, MeanVarOutput *out_buf) {
    mean_var_fixed_scalar(in_buf, out_buf);
}
#endif
