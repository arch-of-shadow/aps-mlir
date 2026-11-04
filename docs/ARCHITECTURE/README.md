# Processor integration

The native configuration exporter supports **RoCC, RV32, one core**. Start with
[examples/interface.json](../../examples/interface.json), then generate the RTL
and Chipyard YAML from the same inputs. This keeps interface capacities and CSR
bindings consistent with synthesis.

## Generate a matched package

```sh
mkdir -p build/integration
pixi run mlir examples/aspdac26-tutorial/cadl/hello.cadl build/integration/hello.mlir
pixi run build/tools/aps-opt/aps-e2e \
  -i build/integration/hello.mlir --export sv -o build/integration/hello.sv \
  --interface-config examples/interface.json \
  --chipyard-config build/integration/hello.chipyard.yaml
```

The JSON is authoritative for interface model fields. Existing IR declarations
must agree; unknown keys, conflicting models, and invalid CSR overrides fail.
The sidecar is emitted after successful lowering/export. Its `vsrc` uses the
absolute generated RTL path, so regenerate the package when moving a checkout.

[InterfaceConfig.h](../../tools/aps-opt/InterfaceConfig.h) validates version 1,
native interface names, beat constraints, platform fields, and CSRs. It derives
`maxBurstBytes` and `nXacts` for the wrapper. Platform configuration requires
SV export to a named file.

## Rocket simulation

Initialize the repository's Chipyard toolchain:

```sh
pixi run setup-chipyard
```

After compiling a matching RISC-V executable, pass the generated YAML explicitly:

```sh
pixi shell
cd thirdparty/chipyard/sims/verilator
make CONFIG=APSRocketConfig APS_CONFIG="$APS/build/integration/hello.chipyard.yaml" \
  run-binary-debug BINARY="$APS/build/program.riscv" LOADMEM=1
```

`build/program.riscv` must be a linked program for that instruction/platform.
[Compiler workflows](../COMPILER/README.md) describe binary generation.
Use an absolute binary path because simulation changes working directory.
Chipyard uses the configured legacy firtool for processor elaboration, while
APS uses its CIRCT fork for accelerator generation; the Pixi setup keeps these
paths distinct.

[generators/aps](../../thirdparty/chipyard/generators/aps) contains the Scala
configuration, blackbox interface, RoCC wrapper, and DMA implementation. The
[port contract](aps-port-convention.md) explains CSR lanes and completion for
instructions without register results. [Memory interfaces](aps-itfc.md) explains
the CPU/DMA timing model and ordering.

## Verification and deployment scope

Use [Spike + Verilator](../SYNTHESIS/aps-spike-verilator-cosim.md) for accelerator
functional checks. Use full Rocket RTL with a defined timed region for processor
cycle measurements. [Benchmark reproduction](../COMPILER/aps-scaleisa-opt.md#benchmark-reproduction-and-evidence)
links current runners and artifact checks.

FPGA and ASIC flows require board/process-specific configuration under
[Chipyard nextvlsi](../../thirdparty/chipyard/nextvlsi) and, for FPGA host access,
[aps-prog](../../aps-prog/README.md). Follow those component instructions for the
actual target. This guide does not assign blanket support status to boards or
CPU/interface combinations that the native exporter does not validate.

Configuration regression coverage is in
[test_interface_config.py](../../tests/synthesis/test_interface_config.py).
