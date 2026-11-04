# Aquas paper examples

The CADL kernels below follow the names in the Aquas ICCAD 2026 paper.
Dots in instruction names become underscores in CADL identifiers and filenames.

| Domain | Paper cases | CADL files |
|---|---|---|
| Cryptography | `vdecomp`, `mgf2mm` | [vdecomp](cryptography/vdecomp.cadl), [mgf2mm](cryptography/mgf2mm.cadl) |
| Point-cloud processing | `vdist3.vv`, `mcov.vs`, `vfsmax`, `vmadot` | [vdist3_vv](pointcloud/vdist3_vv.cadl), [mcov_vs](pointcloud/mcov_vs.cadl), [vfsmax](pointcloud/vfsmax.cadl), [vmadot](pointcloud/vmadot.cadl) |
| Graphics rendering | `vmvar`, `mphong`, `vrgb2yuv` | [vmvar](graphics/vmvar.cadl), [mphong](graphics/mphong.cadl), [vrgb2yuv](graphics/vrgb2yuv.cadl) |

From the repository root:

```sh
make -C examples/aquas CADL=graphics/vmvar.cadl rtl
```

The shared Makefile generates MLIR and RTL under `build/examples/aquas/`.
