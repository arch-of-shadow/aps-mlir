#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Uncomment when building for the custom-instruction target. */
/* #define HARDWARE */

#define CPU_FREQ_HZ 80000000ULL
#define PROFILE_ITERS 200ULL
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

typedef uint32_t (*KernelFn)(const void *in_buf, void *out_buf);

static volatile Rgb2YuvInput rgb_input __attribute__((aligned(128)));
static volatile Rgb2YuvOutput rgb_output __attribute__((aligned(128)));
static volatile Rgb2YuvOutput rgb_ref_output __attribute__((aligned(128)));

static volatile PhongInput phong_input __attribute__((aligned(128)));
static volatile PhongOutput phong_output __attribute__((aligned(128)));
static volatile PhongOutput phong_ref_output __attribute__((aligned(128)));

static volatile MeanVarInput mean_var_input __attribute__((aligned(128)));
static volatile MeanVarOutput mean_var_output __attribute__((aligned(128)));
static volatile MeanVarOutput mean_var_ref_output __attribute__((aligned(128)));

static inline int16_t clamp_u8_i16(int32_t value) {
    if (value < 0) {
        return 0;
    }
    if (value > 255) {
        return 255;
    }
    return value;
}

static inline uint64_t read_counter(void) {
#if defined(__riscv_xlen) && (__riscv_xlen == 32)
    uint32_t lo;
    uint32_t hi;
    uint32_t hi2;
    do {
        asm volatile("csrr %0, mcycleh" : "=r"(hi));
        asm volatile("csrr %0, mcycle" : "=r"(lo));
        asm volatile("csrr %0, mcycleh" : "=r"(hi2));
    } while (hi != hi2);
    return ((uint64_t)hi << 32) | lo;
#elif defined(__riscv_xlen)
    uint64_t cycle;
    asm volatile("csrr %0, mcycle" : "=r"(cycle));
    return cycle;
#else
    return 0;
#endif
}

static inline uint64_t counter_to_us(uint64_t delta) {
    return delta / (CPU_FREQ_HZ / 1000000ULL);
}

static inline void hw_fence(void) {
#if defined(__riscv_xlen)
    asm volatile("fence" ::: "memory");
#endif
}

static inline uint32_t narrow_u32(uint64_t value) {
    return (uint32_t)(value & 0xFFFFFFFFu);
}

static void rgb2yuv_fixed_sw_impl(const Rgb2YuvInput *in_buf, Rgb2YuvOutput *out_buf) {
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

static void phong_fixed_sw_impl(const PhongInput *in_buf, PhongOutput *out_buf) {
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
        int32_t diff = 0;
        if (nl > 0) {
            diff = (int32_t)(((int64_t)nl * in_buf->light_i) >> 8);
        }

        {
            int32_t hx = vx + in_buf->light_x;
            int32_t hy = vy + in_buf->light_y;
            int32_t hz = vz + in_buf->light_z;
            int32_t nh = (int32_t)((((int64_t)nx * hx) +
                                    ((int64_t)ny * hy) +
                                    ((int64_t)nz * hz)) >> 8);
            int32_t spec = 0;
            if (nh > 0) {
                int32_t nh_pos = nh;
                int32_t spec_sq = (int32_t)(((int64_t)nh_pos * nh_pos) >> 8);
                spec = (int32_t)(((int64_t)spec_sq * in_buf->light_i) >> 8);
            }
            out_buf->specular[idx] = (int16_t)spec;
        }

        out_buf->diffuse[idx] = (int16_t)diff;
    }
}

static void mean_var_fixed_sw_impl(const MeanVarInput *in_buf, MeanVarOutput *out_buf) {
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

static uint32_t rgb2yuv_fixed(const void *in_buf, void *out_buf) {
#if defined(HARDWARE)
    uintptr_t rd = 0;
    uintptr_t rs1 = (uintptr_t)in_buf;
    uintptr_t rs2 = (uintptr_t)out_buf;
    asm volatile(".insn r 0x2B, 0b111, 0x18, %0, %1, %2"
                 : "=r"(rd)
                 : "r"(rs1), "r"(rs2));
    return (uint32_t)rd;
#else
    rgb2yuv_fixed_sw_impl((const Rgb2YuvInput *)in_buf, (Rgb2YuvOutput *)out_buf);
    return 0;
#endif
}

static uint32_t phong_fixed(const void *in_buf, void *out_buf) {
#if defined(HARDWARE)
    uintptr_t rd = 0;
    uintptr_t rs1 = (uintptr_t)in_buf;
    uintptr_t rs2 = (uintptr_t)out_buf;
    asm volatile(".insn r 0x2B, 0b111, 0x19, %0, %1, %2"
                 : "=r"(rd)
                 : "r"(rs1), "r"(rs2));
    return (uint32_t)rd;
#else
    phong_fixed_sw_impl((const PhongInput *)in_buf, (PhongOutput *)out_buf);
    return 0;
#endif
}

static uint32_t mean_var_fixed(const void *in_buf, void *out_buf) {
#if defined(HARDWARE)
    uintptr_t rd = 0;
    uintptr_t rs1 = (uintptr_t)in_buf;
    uintptr_t rs2 = (uintptr_t)out_buf;
    asm volatile(".insn r 0x2B, 0b111, 0x1B, %0, %1, %2"
                 : "=r"(rd)
                 : "r"(rs1), "r"(rs2));
    return (uint32_t)rd;
#else
    mean_var_fixed_sw_impl((const MeanVarInput *)in_buf, (MeanVarOutput *)out_buf);
    return 0;
#endif
}

static uint64_t profile_kernel(KernelFn kernel, const void *in_buf, void *out_buf) {
    volatile uint32_t sink = 0;
    uint64_t start = read_counter();
    uint64_t iter;
    for (iter = 0; iter < PROFILE_ITERS; ++iter) {
        hw_fence();
        sink ^= kernel(in_buf, out_buf);
        hw_fence();
    }
    if (sink == 0xFFFFFFFFu) {
        printf("unreachable sink value: %u\n", sink);
    }
    return (read_counter() - start) / PROFILE_ITERS;
}

static int compare_i16_arrays(const int16_t *hw_buf,
                              const int16_t *sw_buf,
                              size_t count,
                              uint32_t label_id) {
    size_t idx;
    for (idx = 0; idx < count; ++idx) {
        if (hw_buf[idx] != sw_buf[idx]) {
            printf("mismatch %u %u %d %d\n",
                   label_id,
                   (unsigned)idx,
                   (int)hw_buf[idx],
                   (int)sw_buf[idx]);
            return 0;
        }
    }
    return 1;
}

static void init_rgb2yuv_case(void) {
    size_t row;
    size_t col;
    for (row = 0; row < 4; ++row) {
        for (col = 0; col < 4; ++col) {
            size_t idx = row * 4 + col;
            ((Rgb2YuvInput *)&rgb_input)->r[idx] = (int16_t)(32 + (int32_t)(row * 40 + col * 9));
            ((Rgb2YuvInput *)&rgb_input)->g[idx] = (int16_t)(16 + (int32_t)(row * 24 + col * 17));
            ((Rgb2YuvInput *)&rgb_input)->b[idx] = (int16_t)(64 + (int32_t)(row * 12 + col * 23));
        }
    }
    memset((void *)&rgb_output, 0, sizeof(rgb_output));
    memset((void *)&rgb_ref_output, 0, sizeof(rgb_ref_output));
}

static void init_phong_case(void) {
    size_t row;
    size_t col;
    for (row = 0; row < 4; ++row) {
        for (col = 0; col < 4; ++col) {
            size_t idx = row * 4 + col;
            ((PhongInput *)&phong_input)->normal_x[idx] = (int16_t)(((int32_t)col - 1) * 32);
            ((PhongInput *)&phong_input)->normal_y[idx] = (int16_t)(((int32_t)row - 1) * 16);
            ((PhongInput *)&phong_input)->normal_z[idx] = (int16_t)(192 - (int32_t)(idx * 2));
            ((PhongInput *)&phong_input)->view_x[idx] = (int16_t)(((int32_t)col - 1) * 16);
            ((PhongInput *)&phong_input)->view_y[idx] = (int16_t)(((int32_t)row - 1) * 16);
            ((PhongInput *)&phong_input)->view_z[idx] = 128;
        }
    }
    ((PhongInput *)&phong_input)->light_x = 32;
    ((PhongInput *)&phong_input)->light_y = -16;
    ((PhongInput *)&phong_input)->light_z = 192;
    ((PhongInput *)&phong_input)->light_i = 256;
    memset((void *)&phong_output, 0, sizeof(phong_output));
    memset((void *)&phong_ref_output, 0, sizeof(phong_ref_output));
}

static void init_mean_var_case(void) {
    size_t idx;
    for (idx = 0; idx < MEAN_VAR_LEN; ++idx) {
        ((MeanVarInput *)&mean_var_input)->data[idx] = (int32_t)((idx % 9) - 4 + (idx / 8));
    }
    memset((void *)&mean_var_output, 0, sizeof(mean_var_output));
    memset((void *)&mean_var_ref_output, 0, sizeof(mean_var_ref_output));
}

static void print_profile(uint32_t kernel_id, uint64_t counter) {
    printf("profile %u %u %u\n",
           kernel_id,
           narrow_u32(counter),
           narrow_u32(counter_to_us(counter)));
}

static int run_rgb2yuv_test(void) {
    int pass;
    uint64_t counter;

    init_rgb2yuv_case();
    rgb2yuv_fixed((const void *)&rgb_input, (void *)&rgb_output);
    rgb2yuv_fixed_sw_impl((const Rgb2YuvInput *)&rgb_input, (Rgb2YuvOutput *)&rgb_ref_output);

    pass = compare_i16_arrays(((const Rgb2YuvOutput *)&rgb_output)->y,
                              ((const Rgb2YuvOutput *)&rgb_ref_output)->y,
                              RGB_PIXELS,
                              0) &&
           compare_i16_arrays(((const Rgb2YuvOutput *)&rgb_output)->u,
                              ((const Rgb2YuvOutput *)&rgb_ref_output)->u,
                              RGB_PIXELS,
                              1) &&
           compare_i16_arrays(((const Rgb2YuvOutput *)&rgb_output)->v,
                              ((const Rgb2YuvOutput *)&rgb_ref_output)->v,
                              RGB_PIXELS,
                              2);

    counter = profile_kernel(rgb2yuv_fixed, (const void *)&rgb_input, (void *)&rgb_output);

    if (pass) {
        printf("[rgb2yuv_fixed] PASS\n");
    } else {
        printf("[rgb2yuv_fixed] FAIL\n");
    }
    print_profile(0, counter);
    return pass;
}

static int run_phong_test(void) {
    int pass;
    uint64_t counter;

    init_phong_case();
    phong_fixed((const void *)&phong_input, (void *)&phong_output);
    phong_fixed_sw_impl((const PhongInput *)&phong_input, (PhongOutput *)&phong_ref_output);

    pass = compare_i16_arrays(((const PhongOutput *)&phong_output)->diffuse,
                              ((const PhongOutput *)&phong_ref_output)->diffuse,
                              PHONG_PIXELS,
                              3) &&
           compare_i16_arrays(((const PhongOutput *)&phong_output)->specular,
                              ((const PhongOutput *)&phong_ref_output)->specular,
                              PHONG_PIXELS,
                              4);

    counter = profile_kernel(phong_fixed, (const void *)&phong_input, (void *)&phong_output);

    if (pass) {
        printf("[phong_fixed] PASS\n");
    } else {
        printf("[phong_fixed] FAIL\n");
    }
    print_profile(1, counter);
    return pass;
}

static int run_mean_var_test(void) {
    int pass;
    uint64_t counter;

    init_mean_var_case();
    mean_var_fixed((const void *)&mean_var_input, (void *)&mean_var_output);
    mean_var_fixed_sw_impl((const MeanVarInput *)&mean_var_input, (MeanVarOutput *)&mean_var_ref_output);

    pass = (((const MeanVarOutput *)&mean_var_output)->mean_q ==
            ((const MeanVarOutput *)&mean_var_ref_output)->mean_q) &&
           (((const MeanVarOutput *)&mean_var_output)->var_q ==
            ((const MeanVarOutput *)&mean_var_ref_output)->var_q);

    if (!pass) {
        printf("mismatch %u %d %d %d %d\n",
               5u,
               (int)((const MeanVarOutput *)&mean_var_output)->mean_q,
               (int)((const MeanVarOutput *)&mean_var_output)->var_q,
               (int)((const MeanVarOutput *)&mean_var_ref_output)->mean_q,
               (int)((const MeanVarOutput *)&mean_var_ref_output)->var_q);
    }

    counter = profile_kernel(mean_var_fixed, (const void *)&mean_var_input, (void *)&mean_var_output);

    if (pass) {
        printf("[mean_var_fixed] PASS\n");
    } else {
        printf("[mean_var_fixed] FAIL\n");
    }
    print_profile(2, counter);
    return pass;
}

int main(void) {
    int passed = 0;

#if defined(HARDWARE)
    printf("graphics kernel test harness\n");
    printf("custom instruction mode\n");
#else
    printf("graphics kernel test harness\n");
    printf("software wrapper mode\n");
#endif
    printf("bilinear excluded\n");

    passed += run_rgb2yuv_test();
    passed += run_phong_test();
    passed += run_mean_var_test();

    printf("summary: %d/3 kernels passed\n", passed);
    return (passed == 3) ? 0 : 1;
}
