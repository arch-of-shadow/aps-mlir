This directory contains CADL translations of the kernel sketches in `graphics_tests.c`.

Conventions used here:
- Structs are flattened into planar or packed memory layouts because the existing CADL examples do not use struct types.
- `mean_var_fixed` keeps 32-bit words, while the `rgb2yuv_fixed` and `phong_fixed` paths use 16-bit planar buffers plus 16-bit scalar parameters.
- `bilinear_sampling_fixed` is expressed in Q8 fixed-point form because the source sketch used floating point and omitted several declarations.
- `bilinear_sampling_fixed` is not included in the C test harness because the current hardware flow does not support its memory access pattern.
- `csr_partition_loop` is a minimal smoke example for `#[csr_address(...)] register` plus partitioned memory. It is intentionally not included in `run_all.sh` until CSR top-port lowering is wired through the backend.

Related C sources in this folder:
- `test_graphics_kernels.c` contains custom-instruction wrappers, software references, exact output checks, and per-kernel profiling for `rgb2yuv_fixed`, `phong_fixed`, and `mean_var_fixed`.
- `graphics_rvv_impl.c` contains a separate RVV-oriented implementation of the same three kernels, with scalar fallbacks when RVV intrinsics are unavailable.
- `test_graphics_rvv.c` contains a profiling-only RVV harness with a `main()` that runs the three kernels repeatedly and prints cycle counts.
- `test_graphics_kernels.c` uses an explicit `HARDWARE` macro to enable the custom-instruction path.
- `graphics_rvv_impl.c` uses an explicit `USE_RVV` macro to enable RVV intrinsics.

Validation command pattern:

```bash
pixi run mlir examples/graphics/<file>.cadl /tmp/<file>.mlir
pixi run opt /tmp/<file>.mlir /tmp/<file>.opt.mlir
```
