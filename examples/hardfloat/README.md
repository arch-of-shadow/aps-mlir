# HardFloat ISAX examples

BF16 and FP32 arithmetic executes in the accelerator's own hardware. See the
[arithmetic contract and acceptance guide](../../docs/SYNTHESIS/hardfloat-bf16-fp32-acceptance.md)
for operations, rounding, flags, native pipeline timing and synthesis validation.
Add/sub/mul/FMA use three-cycle arithmetic pipelines, II=1 and five outstanding
credits. Backpressure buffers completed results without padding their latency.

## Build and test

Build APS and CIRCT before running the examples:

```sh
pixi run cmake --build circt/build --target circt-opt firtool CIRCTECMT2 -j 3
pixi run cmake --build build --target aps-opt aps-e2e -j 3
pixi run build-hardfloat-ip
pixi run bash examples/hardfloat/hardfloat-isax-smoke.sh
pixi run python -m pytest tests/hardfloat -q
# Full numerical, mapped-netlist and STA acceptance:
pixi run hardfloat-acceptance
```

The smoke tests require Verilator, Yosys and Berkeley SoftFloat. Set
`SOFTFLOAT_LIB` if the library is not installed under the Pixi RISC-V toolchain.
`PROTOCOL_ONLY=1` selects the IP handshake tests without compiler/ISAX tests.
The complete acceptance additionally requires the IHP SG13G2 PDK and OpenSTA;
see the linked guide for timing criteria and prerequisites.

The `add` and `add_mul` examples exercise CADL → APS → CMT2 → SV, numerical
results, flags, reset, ordering, response backpressure, CMT2 analyses and
negative goldens. The protocol tests cover native L=3 with capacities 1, 5 and 7.
Retained MLIR examples are compiler regression fixtures.

```sh
pixi run mlir examples/hardfloat/add.cadl build/hardfloat/input.mlir
pixi run build/tools/aps-opt/aps-e2e \
  -i build/hardfloat/input.mlir --resource build/hardfloat/resource.json \
  --export sv -o build/hardfloat/isax.sv
```

Use the generated resource overlay: the integer resource database does not
specify floating-point units. The library builder specializes the HardFloat
Verilog, imports it with Yosys `write_firrtl`, and uses `firtool --ir-fir` to
produce regular FIRRTL modules. CMT2 binds those implementations through its
existing module interface; no Verilog blackbox source annotations are required.
Set `YOSYS` or `FIRTOOL` to override the conversion tools. Exported SV contains
the imported implementation; compile that file alone. Timing metadata is a scheduling
contract, not a target-process frequency guarantee.

## Full Rocket/RoCC GEMM

[BF16 GEMM](bf16_gemm/README.md) preserves partitioning, unrolling and tiled DMA,
with one continuous pipeline across 32 output pairs. The C test is compiled to
a RISC-V executable and run on the complete Rocket/RoCC RTL.

```sh
pixi run bf16-gemm-rocket
BENCHMARK=1 pixi run bf16-gemm-rocket
PROFILE=1 BENCHMARK=1 pixi run bf16-gemm-rocket
```

The software baseline uses native FP32 FMA with no unroll pragma. Its precision
and rounding differ from the separately rounded BF16 multiply/add accelerator;
each is checked against its own independent golden results.

## Constraints

FP64 and IEEE FP16 are unsupported. Ordinary multiply/add is not contracted or
reassociated. Floating values are stored as IEEE bits; bitcasts require equal
widths. Arrays requiring reset initialization must be initialized by hardware
stores. Elastic loop bodies currently require a single basic block without
scalar loop-carried results; unsupported cases produce diagnostics. Functional
units are dedicated per operation, with no resource sharing.
