#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CPU_FREQ_HZ 80000000ULL
#define PROFILE_ITERS 200ULL
#define RUNS 10
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

void rgb2yuv_fixed_rvv(const Rgb2YuvInput *in_buf, Rgb2YuvOutput *out_buf);
void phong_fixed_rvv(const PhongInput *in_buf, PhongOutput *out_buf);
void mean_var_fixed_rvv(const MeanVarInput *in_buf, MeanVarOutput *out_buf);

typedef void (*RvvKernelFn)(const void *in_buf, void *out_buf);

static volatile Rgb2YuvInput rgb_input __attribute__((aligned(128)));
static volatile Rgb2YuvOutput rgb_output __attribute__((aligned(128)));

static volatile PhongInput phong_input __attribute__((aligned(128)));
static volatile PhongOutput phong_output __attribute__((aligned(128)));

static volatile MeanVarInput mean_var_input __attribute__((aligned(128)));
static volatile MeanVarOutput mean_var_output __attribute__((aligned(128)));

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

static uint64_t profile_kernel(RvvKernelFn kernel, const void *in_buf, void *out_buf) {
    uint64_t start = read_counter();
    uint64_t iter;
    for (iter = 0; iter < PROFILE_ITERS; ++iter) {
        hw_fence();
        kernel(in_buf, out_buf);
        hw_fence();
    }
    return (read_counter() - start) / PROFILE_ITERS;
}

static void print_profile(uint32_t kernel_id, uint64_t counter) {
    printf("profile %u %u %u\n",
           kernel_id,
           narrow_u32(counter),
           narrow_u32(counter_to_us(counter)));
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
}

static void init_mean_var_case(void) {
    size_t idx;
    for (idx = 0; idx < MEAN_VAR_LEN; ++idx) {
        ((MeanVarInput *)&mean_var_input)->data[idx] = (int32_t)((idx % 9) - 4 + (idx / 8));
    }
    memset((void *)&mean_var_output, 0, sizeof(mean_var_output));
}

static void run_all_once(void) {
    uint64_t counter;

    init_rgb2yuv_case();
    counter = profile_kernel((RvvKernelFn)rgb2yuv_fixed_rvv, (const void *)&rgb_input, (void *)&rgb_output);
    print_profile(0, counter);

    init_phong_case();
    counter = profile_kernel((RvvKernelFn)phong_fixed_rvv, (const void *)&phong_input, (void *)&phong_output);
    print_profile(1, counter);

    init_mean_var_case();
    counter = profile_kernel((RvvKernelFn)mean_var_fixed_rvv, (const void *)&mean_var_input, (void *)&mean_var_output);
    print_profile(2, counter);
}

int main(void) {
    uint32_t run_id;

    printf("graphics rvv profiling\n");
#if defined(USE_RVV)
    printf("rvv mode\n");
#else
    printf("scalar fallback mode\n");
#endif
    printf("bilinear excluded\n");

    for (run_id = 0; run_id < RUNS; ++run_id) {
        printf("run %u\n", run_id);
        run_all_once();
    }

    return 0;
}
