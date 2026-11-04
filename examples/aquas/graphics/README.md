# Graphics rendering

The Aquas paper uses three fixed-point graphics kernels:

| CADL | Paper case |
|---|---|
| [vmvar.cadl](vmvar.cadl) | `vmvar`: vector mean and variance |
| [mphong.cadl](mphong.cadl) | `mphong`: Phong lighting |
| [vrgb2yuv.cadl](vrgb2yuv.cadl) | `vrgb2yuv`: RGB-to-YUV conversion |

Generate hardware from the repository root:

```sh
make -C examples/aquas CADL=graphics/vmvar.cadl rtl
```

`run_all.sh` generates hardware for all three kernels under
`build/examples/aquas/graphics/`.
