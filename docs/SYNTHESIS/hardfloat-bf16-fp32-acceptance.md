# BF16/FP32 hardware and acceptance

APS floating arithmetic executes in dedicated accelerator units. Operation timing
comes from [ip_catalog.yaml](../../hardware/hardfloat/ip_catalog.yaml), and
packaging lives in [hardware/hardfloat/](../../hardware/hardfloat/).

## Arithmetic contract

BF16 uses 8 exponent bits and 8 significand bits; FP32 uses 8 and 24. Storage and
RoCC transport carry their IEEE bit patterns. Supported operations include
add/sub/mul/div/sqrt/FMA, comparisons, i32/u32 conversions, widening/narrowing,
neg/abs/copysign/min/max, and classification.

CADL exposes `bf16_` and `fp32_` intrinsics; `_flags` selects the five-bit exception
result. Rounding literals are 0=RNE, 1=RTZ, 2=RDN, 3=RUP, 4=RMM. Integer conversion
defaults to RTZ. Exception bits `[4:0]` are NV, DZ, OF, UF, NX and do not implicitly
update CPU `fcsr`. Explicit FMA variants preserve fusion; separate multiply/add
operations remain separately rounded. Unsupported fast-math is rejected.

Decimal literals are rounded directly to their semantic format. Equal-width
bitcasts preserve exact exceptional values. Hardware supports BF16/FP32;
FP64/IEEE FP16 and reset-initialized floating arrays with multiple elements are
rejected. Use explicit hardware stores for those arrays. Scalar floating reset
initialization is supported.

## Timing and backpressure

| Operation | Native latency | II / capacity |
|---|---|---|
| add/sub/mul/FMA | 3 cycles | 1 / 5 |
| integer to FP, narrow, min/max | 2 | 1 / 4 |
| FP to integer, compare/classify, widen, sign operations | 1 | 1 / 3 |
| divide | BF16 ≤11, FP32 ≤27 | iterative / 1 |
| square root | BF16 ≤10, FP32 ≤26 | iterative / 1 |

Latency is accepted issue to first available collect without backpressure.
Iterative exceptional cases can complete early. Reserved response credits retain
data/flags under stalls; reset invalidates queued and in-flight results. Timing
metadata constrains scheduling and does not establish physical frequency.

## Build and use

```sh
pixi run build-hardfloat-ip
pixi run mlir examples/hardfloat/add.cadl build/hardfloat/input.mlir
pixi run build/tools/aps-opt/aps-e2e \
  -i build/hardfloat/input.mlir --resource build/hardfloat/resource.json \
  --export sv -o build/hardfloat/design.sv
```

The builder emits resource overlays, timing metadata, and imported FIRRTL modules.
Use the overlay: integer-only resources lack floating-unit contracts. The backend
checks stale latency resources. The import specializes pinned HardFloat Verilog,
uses Yosys `write_firrtl` and firtool `--ir-fir`, and binds ordinary FIRRTL modules
through CMT2. Exported SV contains the implementation.

## Validation

```sh
pixi run python -m pytest tests/hardfloat -q
pixi run python tests/hardfloat/acceptance/generate_cases.py
pixi run python tests/hardfloat/acceptance/run.py --rtl-only --jobs 3
```

The acceptance runner uses paired CADL/C cases, an independent numerical oracle,
and complete accelerator integer ports. It checks results/flags, destination
registers, backpressure, reset/retry, ordering, and timeouts. This direct-RTL
campaign is separate from executing a C binary on full Rocket RTL.

For mapped-netlist simulation and STA, omit `--rtl-only`. That requires the
IHP SG13G2 PDK and OpenSTA at the paths used by
[acceptance/run.py](../../tests/hardfloat/acceptance/run.py) and
[physical_flow.py](../../hardware/hardfloat/synthesis/physical_flow.py).
`--units` selects standalone unit campaigns; `--only` filters cases;
`--skip-passed` reuses signature-matching results; `--require-timing` makes
negative slack fatal. Functional success alone does not imply timing closure.

Artifacts under `build/hardfloat/acceptance/` include native/mapped results,
provenance, gate logs, pipeline metadata, and setup/hold reports. Read the generated
summary for results from the actual checkout rather than relying on historical
case counts or performance numbers.

See the [HardFloat examples](../../examples/hardfloat/README.md),
[full Rocket BF16 GEMM](../../examples/hardfloat/bf16_gemm/README.md), and
[library tests](../../tests/hardfloat/test_library.py). Packaging provenance is
in [thirdparty/HardFloat/README.aquas.md](../../thirdparty/HardFloat/README.aquas.md).
