# Megg instruction examples

Each case pairs a CADL instruction description with a C caller/reference:
`avg_r`, `horner3`, `lerp`, `q15_mulr`, `v3ddist_vs`, `v3ddist_vv`, `vcovmat3d_vs`,
`vfpsmax`, and `vgemv3d`. `csr_scale` demonstrates CSR-configured operands.

Use the group Makefile from the repository root:

```sh
make -C examples/megg-compiler compile
make -C examples/megg-compiler rtl
```

The per-case `compile.sh` scripts invoke the same `pixi run compile` entry point.
They write executable and compiler artifacts under `build/examples/megg/`.
Compile the matching hardware separately with the `rtl` target or `aps-e2e`.

The [compiler guide](../../docs/COMPILER/README.md) covers prerequisites;
[Megg internals](../../docs/COMPILER/pattern-matching.md) describes matching,
rewrites, extraction, and CSR operand binding.
