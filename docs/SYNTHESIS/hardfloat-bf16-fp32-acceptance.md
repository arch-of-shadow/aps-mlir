# Standalone BF16 / FP32 ISAX arithmetic and acceptance

Arithmetic executes inside the accelerator, **not** in Rocket's FPU or the test host.

## Semantics and API

- BF16: exponent 8, significand 8; **not IEEE FP16**. FP32: exponent 8,
  significand 24. Physical storage is UInt<16>/UInt<32> containing IEEE bits.
- Add, subtract, multiply, divide, sqrt, fused FMA; all 16 comparison predicates;
  signed/unsigned 32-bit integer conversion; BF16 widening and FP32 narrowing;
  negation, absolute value, copysign, min/max and RISC-V classification.
- `bf16_` / `fp32_` named intrinsics expose these operations. `_flags` selects
  the separately observable five-bit exception result. Explicit `fmsub`,
  `fnmadd`, `fnmsub` preserve one fused FMA after sign operations.
- Rounding: literal 0=RNE, 1=RTZ, 2=RDN, 3=RUP, 4=RMM. Integer conversion
  defaults to RTZ. Flags: `[4:0] = NV DZ OF UF NX`; no implicit CPU `fcsr` update.
- Ordinary multiplication plus addition is **not** contracted. Floating
  reassociation / unsupported fast-math is rejected, not silently ignored.
- Named operations lower through typed `aps.fp` with data and flags results.
  Standard arith/math operations use the same issue/collect backend.
- Floating memories store bits without integer numeric conversion. Paired tests
  cover dynamic-address explicit stores/reads and scalar 1.25 reset initialization.
  Initialized floating arrays with more than one element are explicitly rejected:
  use hardware stores. Unsupported formats and integer conversion widths are
  diagnostic; neither FP64 nor native portable-C BF16 arithmetic is claimed.

### Example

```cadl
#[opcode(7'b0001011)]
#[funct7(7'b0000000)]
rtype bf_add(rs1:u5, rs2:u5, rd:u5) {
    let ab:u16 = _irf[rs1];
    let bb:u16 = _irf[rs2];
    let a:bf16 = bitcast<bf16>(ab);
    let b:bf16 = bitcast<bf16>(bb);
    let result:u32 = bitcast<u16>(bf16_add(a,b,0));
    _irf[rd] = result;
}
```

Decimal constants default to f32; explicit `_bf16`, `_f32`, `_f64` suffixes
select their semantic type. Bitcasts require equal widths. For exact exceptional
values, use integer IEEE encodings and bitcasts.

## Native arithmetic pipelines (catalog version 2)

`hardware/hardfloat/ip_catalog.yaml` is the source of operation/precision timing.
The packager generates the SV timing macros, C++ lookup tables, metadata and
precision-specific HLS resources. Fixed latency is measured from accepted issue
to first available collect, without backpressure; the result buffer adds no
cycle when empty. The compiler rejects schedules from stale latency resources.

| Operation | BF16 | FP32 | II / capacity |
|---|---:|---:|---|
| add/sub/mul/FMA | 3 | 3 | 1 / 5 |
| signed/unsigned integer to FP | 2 | 2 | 1 / 4 |
| FP to integer, compare, classify | 1 | 1 | 1 / 3 |
| widen, neg, abs, copysign | 1 | 1 | 1 / 3 |
| narrow, min, max | 2 | 2 | 1 / 4 |
| div | at most 11 | at most 27 | bound / 1 |
| sqrt | at most 10 | at most 26 | bound / 1 |

Widen is BF16→FP32 and narrow is FP32→BF16. Iterative bounds assume collection
when ready; early results are not padded. Both iterative operations complete
special cases in 2 cycles. Backpressure can extend occupancy without limit.

`HFNativePipeline` follows Rocket's three FMA register boundaries: recoded
operands; multiply result and aligned control; post-multiply raw result and
rounding control. Rounding and IEEE encoding follow the final boundary.
Add/sub/mul specialize the FMA inputs at elaboration, including signed-zero
handling. Separate CADL multiply/add operations remain separately rounded.
Other operations use their own one/two-cycle native paths. Div/sqrt use the
HardFloat iterative core on the primary clock, with one input capture stage,
combinational result rounding and flow-through response delivery.

The credit-reserved response FIFO holds data/flags during arbitrary stalls.
Reset invalidates all in-flight and queued responses. Iterative units support
completion and next admission on the same edge, with one outstanding operation.
No CPU FPU or FP instruction implements accelerator arithmetic.

`hardware/hardfloat/synthesis/physical_flow.py` maps these actual RTL registers without
repartitioning or inserting stages. `pipeline.json` records native boundary
mode and mapped cell/DFF counts; `mapping.log` records Liberty cell area.
Clock constraints remain explicit. There are no generated slow clocks.

Numerical/protocol failures, unconstrained real endpoints, and STA tool errors
remain fatal. Setup/hold slack is reported independently as `timing_met`;
`--require-timing` also makes negative slack fail acceptance. The default 4 ns
IHP130 target is a measurement point, not a reason to add padding or a frequency
claim. Equal cycle counts with Rocket do not prove equal physical critical paths.

## Reproduce acceptance

The CMT2 library build specializes the pinned Verilog and imports its logic
through Yosys `write_firrtl` and `firtool --ir-fir`. It emits ordinary FIRRTL
modules with the existing named ports. Upstream files in
`thirdparty/HardFloat/source` remain hash-checked and unchanged. Asynchronous
reset registers in division/square-root retain asynchronous FIRRTL resets.
Truncated shifts retain Verilog overshift behavior while bounding FIRRTL
intermediate widths, including floating-point-to-integer conversions.
Run `pixi run python -m pytest tests/hardfloat/test_library.py -q` to test
imported implementations, CMT2 bindings, and numerical/protocol RTL behavior.

```sh
pixi run cmake --build build -j 3
pixi run build-hardfloat-ip
pixi run python tests/hardfloat/acceptance/generate_cases.py
# Compiler and RTL acceptance, without cell mapping or OpenSTA:
pixi run python tests/hardfloat/acceptance/run.py --rtl-only --jobs 6
# Also run mapped-gate simulation and timing analysis:
pixi run python tests/hardfloat/acceptance/run.py --jobs 6
pixi run python tests/hardfloat/acceptance/run.py --units --jobs 3
```

- **75 paired CADL/C cases**, 113,000 exact reference requests: arithmetic,
  predicates, flags, conversions, five rounding modes, FMA sign variants,
  independent packed BF16 FMA operands, branch captures, loop-carried values,
  floating memory, scalar initializers and decimal halfway-boundary constants
  materialized by single rounding (without host-double double rounding).
- **66 standalone configurations**: both precisions, all operations and all
  comparison predicates. Data and flags are checked separately for five RMs.
  Reset-invalidated requests are retried, so every reference vector must have a
  numerically checked response.
  Every BF16 encoding is tested for sqrt, classification, sign operations,
  integer conversion and widening. There are 2,706,760 reference requests.
- Native C comparison programs call `isax_execute()`. The C++ driver implements
  that ABI through the **complete accelerator's integer RoCC ports**, with
  command bubbles, prolonged/random response stalls, reset/retry, destination
  checks, stale/duplicate response checks and timeouts. A second phase presents
  up to eight outstanding commands under response stalls, checks every result
  and destination, and reports the actually observed `peak_outstanding`.
  Standalone DIV/SQRT tests additionally enforce the generated latency bound.
  No CPU floating
  instructions implement the arithmetic. This is not a full Rocket+C binary
  integration test; earlier CPU-FPU smoke tests are a separate validation layer.
- Independent integer/Fraction oracle, directly rounded BF16/FMA, cross-checked
  against SoftFloat. Tininess-after-rounding uses unbounded exponent range at
  the format's precision, before subnormal precision loss. A rounded normal
  can legitimately raise UF|NX: `minSubnormal*minSubnormal+maxSubnormal` under
  RUP is a directed regression.
- Real TT SG13G2 Liberty, 1.20 V / 25 C: primary **4 ns**;
  setup/hold uncertainties 0.10/0.05 ns; input max/min 0.20/0.10 ns,
  transition 0.10 ns; output delay 0.20/0.00 ns and load 0.02 pF.
- All actual unconstrained endpoints are fatal. Constant outputs and constant-fed
  DFF data pins are exempted **only after structural constant proof** using the
  Liberty functional logic. Missing reports, unknown clock domains, STA errors and unresolved cells fail
  acceptance. Negative slack is reported and is fatal with `--require-timing`.
- Verilator uses nonblocking functional cell models generated from the same
  unmodified Liberty: original PDK specify/UDP models are not supported faithfully
  in its zero-delay two-state scheduling. Numerical gate simulation is separate
  from STA using real cell delays. No SDF simulation, extracted routing,
  placed clock tree, multi-corner or post-layout signoff is claimed.

Generated CADL/C sources, relative-path manifest and vectors live under
`build/hardfloat/acceptance/generated/`; regeneration does not modify source files.

Artifacts: `build/hardfloat/acceptance/<case>/{native,mapped}/`, including
`gates.v`, `pipeline.json`, `provenance.json`, gate logs,
constant proofs, setup/hold reports and `result.json`. `summary.json` and
`unit-summary.json` aggregate the selected campaigns. `--skip-passed` accepts
cached passes only when their source/compiler/Liberty signature matches, and
keeps cached reports in the aggregate. `--only` is a regex.

## Native-pipeline validation

- 35 Python/compiler regressions pass, including stale-resource rejection and
  precision-specific DIV/SQRT loop II checks.
- 75/75 CADL/C cases pass native RTL and mapped-gate numerical checks.
- 66/66 IP configurations pass RTL assertions, RTL/gate numerical and protocol
  checks: 2,706,760 reference requests per implementation. This includes all five
  rounding modes and exhaustive BF16 cases in the existing campaign.
- The direct FP32 wrapper also passes SoftFloat comparisons at capacity 1, 5 and
  7, and deliberate wrong-result tests verify failure reporting. CMT2 call,
  conflict, scheduler, inlining and call-sequence checks pass in the smoke flow.
- Full Rocket/RoCC correctness passes 32 tiles / 2048 outputs, including C
  accumulation, in-place writeback, high SPM offsets and DMA padding guards.
  The corrupted-golden run fails with `tohost = 1`, as required.
- DIV ready latency is observed at 2–11 / 2–27 cycles for BF16/FP32; SQRT at
  2–10 / 2–26. The 2-cycle exceptional path is asserted, as are the upper bounds.
- STA ran for all cases. At the existing 4 ns TT constraint, **11/75 CADL cases
  and 25/66 IP configurations meet setup/hold**. All other cases retain their
  negative-slack reports; functional acceptance does not claim timing closure.

Selected IP measurements (Liberty mapped cell area, before placement/routing;
DFF count before explicit drive/hold buffers):

| IP | Data/control DFFs | Cell area (µm²) | Setup slack at 4 ns | Hold slack |
|---|---:|---:|---:|---:|
| BF16 add | 234 | 30009.31 | −1.3291 ns | +0.1122 ns |
| BF16 mul | 226 | 32210.02 | −1.4777 ns | +0.1122 ns |
| BF16 FMA | 251 | 36595.31 | −3.0551 ns | +0.1122 ns |
| FP32 add | 411 | 63822.88 | −4.9650 ns | +0.1122 ns |
| FP32 mul | 387 | 92445.15 | −6.0017 ns | +0.1122 ns |
| FP32 FMA | 444 | 103872.36 | −7.8096 ns | +0.1122 ns |

`build/hardfloat/acceptance/native-pipeline-summary.json` records all IP cell
areas, register counts and timing results. `summary.json` and `unit-summary.json`
retain per-case results and source signatures. Current full Rocket GEMM methodology and measurements are documented in
[the GEMM example](../../examples/hardfloat/bf16_gemm/README.md): the continuous
pipeline achieves 2.174× end-to-end speedup over native FP32 C at equal assumed
clock. Both engines use independent numerical goldens.

Original HardFloat Release 1 sources, license and hashes remain unchanged.
Packaging fixes BF16 power-of-two significand alignment (`clog2(p+3)`) and a
redundant wire declaration; these generated-source normalizations are documented
in `thirdparty/HardFloat/README.aquas.md`.
