# APS ScaleISA driver

`aps-scaleisa-opt` is the C++ driver for ScaleISA discovery and selection,
selected-kernel adaptation, APS RTL generation, RV32 software compilation,
and optional Spike + Verilator cosimulation. The adapter and CI compiler pass
executables live under `tools/aps-scaleisa-opt/` beside the driver. CI lowering
remains in `lib/scaleisa-compiler`; tool invocation and linking belong to the driver.

```bash
cmake --build build --target aps-scaleisa-opt scaleisa-adapter-opt scaleisa-compiler-opt
build/tools/aps-scaleisa-opt/aps-scaleisa-opt input.mlir \
  --area-budget=500 --output-dir=build/scaleisa-run
```

The default emits `design.sv` and `program.o`. Discovery runs all default
ScaleISA stages with greedy selection, using the requested area budget.
`--discovery-stages=scf` restricts discovery to scalar patterns for scalar
kernel experiments; the default is `linalg,affine,scf`.
`--discovery-dir` reuses a completed ScaleISA run, whose budget and input must
match. Selected implementations are collected from its implementation artifacts;
the final selected program supplies the AU call sites for software compilation.
Selected loop pragmas are preserved by the adapter. Writable memory arguments
receive noalias annotations only when all observed call sites prove distinct
allocation origins; unresolved alias contracts are checked by the adapter.
Defined globals with distinct symbols also prove disjoint storage; casts retain
their underlying storage identity. Passing one global more than once does not
provide a disjointness proof.
The report records ScaleISA's kernel area and latency estimates and the measured
wall time of each stage.

Use `--emit=elf --entry=_start`, repeated `--link-object` or `--link-source`,
and `--link-arg=-Tplatform.ld` for a linked executable. Runtime/startup objects
and their target ABI are explicit inputs. `--cosim` builds the matching RTL
simulator and executes that ELF with CSR bindings derived from the hardware
export. `APS_COSIM_PROGRESS=1` is enabled for cosimulation. Stage output is logged
under the output directory, and progress is streamed to the terminal.

Validation can rename `main` with `--rename-main`, and insert a callback using
`--observe-function` plus repeated `--observe-type='memref<...>'` arguments.
Each requested type must identify exactly one entry-block allocation. For repeated
types, use repeated `--observe-allocation=N` instead, selecting zero-based
entry-block allocation indices in the selected main. Arguments follow the order
of the requested indices. The two selection forms are mutually exclusive;
out-of-range indices fail before compilation. A supplied validation object
implements the callback and checks the arrays against an independent reference.
Instrumentation runs after discovery and selection, preserving the selected AU
calls and keeping validation outside the performance model.

`--compile-only --hardware=hardware.mlir -o program.o` compiles an already
selected program without rerunning discovery or synthesis. `--allow-no-ci`
permits an explicit software baseline. Compilation intermediates and counts are
saved in `<output>.scaleisa/compile-report.json`; full runs save `report.json`
with stage commands, durations, selection estimates, and artifact hashes.
Failed stages stop the flow and record the diagnostic; a binary is published
only after compilation and linking succeed. `--timeout` bounds each subprocess.

The full octree regression can be run directly with the driver. Its C++ harness
checks all point coordinates, occupancy counts, query scores, and signature
against the source reference.

`python tests/aps-scaleisa/run_octree_case.py --area-budget 5000` runs this
full-sized case and a zero-CI software baseline with shared preprocessing. It
writes correctness, selected area, modeled program speedup, and measured RTL CI
cycles to the output directory's `summary.json`. `--reuse-pipeline` validates
an existing successful run's input, budget, and target profile before reusing it.

An earlier run used IHP130 delays together with ASAP7 area estimates. With a
4 ns target, the
2048-point, four-level, 128-query snapshot passes all 22656 observed values and
the signature 17565906. At budget 5000 it selects no CIs. The program estimate
changes from 2346341 to 2162021 cycles (1.085×) through software rewrites.
Spike kernel instruction counts change from 1167637 to 1051660, excluding the
reference calculation and observer validation. These instruction counts are
not cycle measurements. The hardware-feasibility conclusions from that mixed
model are superseded. Its original records remain in
`build/aps-octree-speedup-a5000/summary.json`.

An earlier corrected ASAP7 run at 8 ns selected two instruction implementations (ten
static sites), with an area proxy of 1499.29. It passes RTL cosimulation and the
independent software baseline checks for all 22656 values. The same functional
AU library was separately rescheduled and simulated at 4 ns: both helpers remain
feasible and their measured CI latencies are unchanged (34834 cycles for the
bulk-memory instruction and three cycles per scalar multiplication). This is a
fixed-library clock comparison, not a separate 4 ns budgeted selection.
The records are in `build/aps-octree-asap7-8ns-a5000/summary.json` and
`build/aps-octree-asap7-4ns-fixed-selection/comparison.json`. The bulk-memory
instruction exceeds its 18461-cycle prediction; long serial DMA loop overhead
still needs modeling. The updated selection and full-processor speedup are
recorded below.

The equivalent direct driver invocation is:

```bash
build/tools/aps-scaleisa-opt/aps-scaleisa-opt \
  scaleisa/benchmarks/pcl/octree_search/octree_search.mlir \
  --area-budget=500 --output-dir=build/aps-scaleisa-octree \
  --emit=elf --entry=_start --abi=ilp32f \
  --mattr=+m,+a,+f,+zicsr,+zifencei --cosim \
  --rename-main=pcl_selected_main --observe-function=pcl_validate_outputs \
  '--observe-type=memref<2048x3xf32>' \
  '--observe-type=memref<4x4096xi32>' '--observe-type=memref<128xi32>' \
  --link-source=tests/aps-scaleisa/fixtures/octree_main.cpp \
  --link-source=tests/scaleisa-compiler/fixtures/start.S \
  --link-arg=-Ttests/scaleisa-compiler/fixtures/ram.ld
```

This uses the project's Pixi tools when available and the existing CIRCT LLVM
build. ScaleISA's release engine must be built under
`scaleisa/tools/scaleisa-engine/target/release/`. Run regression checks with
`cmake --build build --target check-aps-scaleisa check-scaleisa-adapter check-scaleisa-compiler`.

For fast scalar experiments, run:

```bash
.pixi/envs/default/bin/python tests/aps-scaleisa/run_scalar_cases.py
```

The default cases are ORB descriptor `bitcount` and PCL k-d tree `sq_dist`,
each with 64 deterministic inputs and area budget 500. These are extracted
kernel microbenchmarks, not complete application runs. Original kernel bodies
are preserved; standard affine lowering prepares the inputs for SCF discovery.
Public result buffers keep the computations observable during optimization.
The production driver performs AU discovery, insertion, RTL generation,
compilation, and Spike + Verilator execution. Both accelerated and zero-CI
software executions are checked against an independent C++ reference, with
bit-exact floating-point comparisons. Input provenance is saved under
`scaleisa/tmp/scalar-inputs-aps-scaleisa-scalars/`.
The software baseline uses the same ScaleISA preprocessing, including private
counted-loop promotion, so its correctness check also covers the normalized
software path. Use `--area-budget 5000` to admit a fully unrolled bitcount CI;
the default budget admits the smaller sequential implementation.

`build/aps-scaleisa-scalars/summary.json` records correctness, selected area,
discovery counts, simulation time, and the estimated cycle speedup:
ScaleISA software-only latency divided by selected latency. This is a model
estimate, not a measured processor speedup; simulator wall time is separate.
The optional `--cases clamp` currently exposes a ScaleISA parsing failure for
the synthesized Boolean attribute `true : i1`, so it is excluded from defaults.

An earlier mixed-model experiment with private counted-loop promotion enabled
selected one
whole-loop CI per input. At area budget 5000, the area proxy is 4125.29 and the
program model predicts 9109 / 3093 = 2.95×. RTL cosimulation measures 34 cycles
per CI, compared with 32 scheduled AU cycles and 36 interface-inclusive
predicted cycles. Budget 500 selects a sequential implementation with area
326.56 and 101 measured CI cycles; its 69-cycle prediction misses controller
overhead. The larger implementation therefore reduces measured CI latency by
2.97×. Squared distance remains software-only at both budgets. Both programs
and their normalized baselines pass all 64 independent output checks. These
results are kernel microbenchmarks; the 2.95× ratio is not a Chipyard measurement.
The comparison artifact is `build/aps-speedup-summary.json`.

For processor cycle measurements, reuse those scalar artifacts with:

```bash
.pixi/envs/default/bin/python tests/aps-scaleisa/run_chipyard_scalars.py
```

This runs Chipyard's `make CONFIG=APSRocketConfig run-binary-debug LOADMEM=1`
with explicit `BINARY=.../baseline.riscv` and `BINARY=.../accelerated.riscv`.
`APS_CONFIG` selects each generated YAML; build and output directories are
isolated under `build/aps-scaleisa-chipyard/`, leaving Chipyard's default YAML
and simulator untouched. Both binaries run on the same Rocket SoC simulator
for each case, including caches, RoCC, TileLink, memory, and the selected RTL.
The target reads `mcycle` around each 64-input function call, including memory
fences. The first call measures cold execution; the second measures warm
execution. Each call is validated outside its timed region. HTIF prints both
counts after timing; Chipyard verifies the executable's exit status.
The summary records software/accelerated cycle ratios alongside ScaleISA's
model estimates, the Chipyard commit, and simulator/accelerator hashes.
Accelerator RTL is regenerated with the current APS lowering by default;
`--no-resynthesize` explicitly reuses the original scalar-run RTL.
The native driver uses the calibrated Rocket profile. Generic ScaleISA runs
retain their U74 default. Predictions remain separate from measured ratios.

Single-rule instructions forward the command's current destination register
into their response; instructions completing in later rules use the saved
destination. The Spike RTL bridge also rejects mismatched response tags,
matching Rocket's requirement for completing a pending register write.

The earlier SCF-only experiment, before Rocket cost alignment and private-loop
promotion, measured these results at Chipyard `7fc52519` with 64 inputs per call
and area budget 500:

| Case | Software cycles, cold / warm | Accelerated cycles, cold / warm | Chipyard ratio, cold / warm | ScaleISA estimate |
| --- | ---: | ---: | ---: | ---: |
| ORB bitcount | 32,984 / 32,418 | 100,933 / 99,423 | 0.327× / 0.326× | 1.502× |
| PCL squared distance | 3,549 / 2,401 | 5,821 / 4,702 | 0.610× / 0.511× | 1.499× |

Both accelerated binaries and their software baselines pass the reference
checks. Ratios below one indicate a slowdown. These small selected instructions
do not yield a Rocket speedup in this experiment despite the model predictions.
The raw target output, FST waveforms, make invocations, and aggregated results
are under `build/aps-scaleisa-chipyard/`. Bitcount's timed simulation exposed
the destination-tag issue described above; its reported counts use regenerated,
corrected RTL. Squared distance's RTL is byte-identical under the corrected
lowering.

Inspection of the existing full-SoC waveforms explains these slowdowns:

- The scalar runner explicitly used SCF-only discovery. SCF implementation
  enumeration uses direct implementations; affine pipeline/unroll variants
  were excluded. Both candidate inventories contain no loops. The adapter
  preserves implementation pragmas, but these selected helpers have none.
- The CSR allowance was five, and every selected helper takes at most two
  operands and uses zero extra CSRs. No helper was split for CSR capacity.
  This adapter limit applies after selection and did not constrain discovery.
- Bitcount offloads compare, shift, AND, and addition as separate instructions.
  It executes 161 RoCC commands per input, or 10,304 per 64-input call.
  Each command takes four core cycles from accepted command to accepted
  response (two cycles at the accelerator wrapper). The resulting 41,216
  cycles pending responses already exceed the software's entire 32,418-cycle
  warm call. The bit loop grows from nine native instructions to twelve
  instructions including five custom instruction sites, plus their stalls.
- Squared distance offloads three subtractions and two additions per point;
  the three multiplications remain on Rocket. Each RoCC command takes seven
  core cycles (five at the wrapper). It also adds six integer/FP register
  transfers per point, and the loop grows from 28 instructions to 37.
- The model uses raw candidate latency without an explicit Rocket RoCC
  command/response cost. Its FP resource profile assumes one-cycle add/sub;
  APS emits three-cycle HardFloat operations, before interface overhead.
  Bitcount also exposes an execution-count limitation: the model assigns each
  inner-while site count 64, whereas the trace shows 2,048 body executions
  and 2,112 guard evaluations. Trip-count annotation handles `affine.for`
  and `scf.for`, but not this `scf.while`.

The machine-readable evidence is `build/aps-scaleisa-chipyard/investigation.json`;
`rocc-analysis.json` records command counts and measured latency distributions.
Future optimization comparisons should explore fused/loop implementations and
their pragmas, align the hardware latency profile with APS, account for the
RoCC/register-transfer cost, and supply accurate loop execution counts.

The performance-alignment investigation under
`scaleisa/tmp/aps-performance-investigation/` adds a concrete banking probe.
A helper loads elements zero and one from an eight-element array, adds them,
and returns the result. A two-bank cyclic plan assigns the loads to banks zero
and one, but Hector schedules them at cycles one and two, exactly as for the
unpartitioned helper. Both variants report latency four and identical area.
The scheduler currently does not consume the bank annotations: resource
reservations use a common resource name, and memory dependencies and pipeline
reservations use the original memory value. Independent bank ports must become
explicit scheduling resources before a partition pragma can improve throughput.
The APS export probe also shows that argument partition attributes are absent
from the generated scratchpad global. Export must translate the selected plan
through buffer flattening and preserve its bank/address mapping for APS's
partition and DMA passes.

Target alignment requires a shared, explicit profile for the processor,
accelerator timing, port topology, ABI costs, and area basis. Operator latency
and initiation interval must remain independent: APS's three-cycle FP units
accept work every cycle, while Hector currently raises II to at least latency
for characterized resources. APS also supports combinational integer operators,
where Hector's registered-primitive scheduling clamps latency to at least one.
The current ScaleISA clock profile, APS scheduling clock, and Chipyard core
clock are separate settings and should be recorded and checked explicitly.
Profile changes must invalidate cached implementation evaluations.

Implementation exploration should jointly vary partial/full unroll, pipeline
II, and per-array partition plans, charging bank storage, ports, muxing, and
control as well as arithmetic. The selected plan must agree across evaluated
MLIR, APS scratchpads, scheduled TOR, and RTL. After correcting execution counts
and calibrating interface/transfer costs, fresh budget sweeps at 500, 1,000,
2,000, and 5,000 can distinguish extra area from changes in discovery and
parallelism. Each reported improvement still requires matching software and
accelerated Chipyard runs with reference validation. The investigation report
records the component-level findings, implementation order, and acceptance
checks in `scaleisa/tmp/aps-performance-investigation/investigation.json`.

## Target-aware bounded implementation search

The driver realizes `target-profile.json` from `hardware/performance/rocket-rv32.json`
and ScaleISA's characterized ASAP7 delay/area tables. The default period is
read from the hardware profile (currently 10 ns); `--clock` overrides the
scheduling target.
The APS resource JSON supplies execution latencies, initiation intervals and
resource limits. Its IHP130 delay values are excluded. Both Hector and APS use
the same ASAP7 delays, with the APS aliases preserved for RTL lowering.
`--resource` supplies those execution contracts; a complete physical profile
can be supplied with `--target-profile`. CPU calibration comes from dependent-operation
chains in full Chipyard `APSRocketConfig`; hardware uses the requested APS clock,
latencies and initiation intervals. `--target-profile` accepts a realized profile
and supplies its default period. An explicit `--clock` must match that profile.
Profile changes invalidate discovery reuse.
Area remains an ASAP7 primitive proxy, with memory ports charged per bank; it is
not a mapped physical area prediction.

Search admits at most eight implementations per helper. Pipeline II is chosen
automatically, partial factors 2/4/8 are coupled with cyclic innermost banks,
and static expansion is limited to 4,096 operations. Hector evaluations have a
30-second limit, four workers, and a cache keyed by implementation, scheduler
binary, resources and clock. The selected implementation proceeds to RTL.
Partition attributes survive the flat scratchpad ABI; unsupported layouts fail
explicitly. CPU-memory import/export, RoCC command overhead, FP register moves,
and extra CSR writes contribute to instruction-boundary cost.
Long runs of identical DMA bursts are lowered to affine loops once they exceed
16 transactions. Each iteration retains the selected burst size and advances
both the CPU byte address and scratchpad element index. Short plans remain
static so their independent requests can overlap. This bounds generated control
logic for large buffers without changing transfer coverage or tail handling.

If selection chooses software only, the driver emits a zero-CI binary, skips
RTL generation, and uses Spike when simulation was requested. Reports distinguish
software simulation from RTL cosimulation. This is a valid selection result when
instruction or memory-transfer overhead exceeds the saved computation.

The instruction calibration source is
`tests/aps-scaleisa/fixtures/rocket_calibration.cpp`. It times 1,024 dependent
operations per class twice with `mcycle`, printing the warm result after timing.
Recorded measurements and the Chipyard commit live in the target profile.
`rocket_conversion_calibration.cpp` additionally measures 1,024 dependent pairs
of signed i32/f32 conversions and FP register moves. Its conservative per-op
cost is four cycles. Other conversion type pairs require separate calibration.
Memory calibration uses 16 warm read-only invocations per size; its fitted setup
and throughput describe warm transfers rather than cache-miss timing.

RTL cosimulation writes `ci-runtime.json`, grouping dynamic calls by opcode and
funct7 with exact total/minimum/maximum cycles and a mean. `report.json` joins
those measurements to `estimated_au_cycles` (Hector datapath/schedule estimate)
and `estimated_ci_cycles` (including the modeled command/context/DMA boundary).
The comparison excludes CPU CSR writes and FP register moves, which execute
outside the RTL command. Unexecuted selected helpers have no measured sample.
Split CI parts retain their encoding-level measurements; an AU estimate is joined
only when the exported instruction matches that AU implementation directly.

The APS profile rejects pipeline variants with loop-carried SSA results, which
the CMT2 backend currently cannot lower. Reduction helpers can still use direct
or fully unrolled implementations. Preserved pipeline pragmas translate
ScaleISA's automatic-II value into APS's separate enable flag and optional II.
For rapid model tuning, compare these AU/CI cycle reports; use full Chipyard for
processor-level measurements on real benchmark data.

Run the validated octree selection and its normalized software baseline through
full Chipyard Rocket RTL with:

```bash
.pixi/envs/default/bin/python tests/aps-scaleisa/run_octree_case.py \
  --area-budget=5000 --clock=8 --output-dir=build/aps-octree-optimized
.pixi/envs/default/bin/python tests/aps-scaleisa/run_chipyard_pcl.py \
  --source-dir build/aps-octree-optimized \
  --output-dir build/aps-octree-optimized-chipyard
```

The runner builds one `APSRocketConfig` simulator, then runs both binaries with
`APS_CONFIG`, `run-binary-debug`, `LOADMEM=1`, and explicit `BINARY` paths. Each
binary checks all 22,656 values and the checksum on two calls. `mcycle` timing
starts immediately before the benchmark call and ends on entry to its output
observer, excluding reference generation, checking, and HTIF printing. Cold and
warm mean the first and second calls with identical inputs; checking occurs
between them. Reports contain measured cycle ratios and the separate ScaleISA
estimate. These are cycle measurements, not a physical timing signoff.

Waveform recording is disabled by default through Chipyard's
`get_waveform_flag=` override; `--waveform` enables it. Both modes execute the
full Verilator model with protocol assertions. The Chipyard wrapper discovers
generated `instruction_completed_<id>_` ports and uses their toggle to release
void CIs, which have no register response. Register-producing CIs retain
outstanding-response tracking. `fixtures/chipyard_void_dma.cpp` under
`tests/aps-scaleisa` tests two 64-KiB zeroing CIs and a subsequent multiply for
the earlier two-instruction octree hardware configuration. Its multiply opcode
is absent from the current one-instruction selection.

The earlier 8-ns, area-5000 octree selection passed full Chipyard validation.
Baseline cycles were 1,872,789 cold and 1,855,586 warm; selected-program cycles
were 1,969,408 cold and 1,950,739 warm. Warm speedup was 0.9512x, corresponding
to 5.13% longer runtime, versus ScaleISA's 1.1619x estimate. The CPU baseline
estimate was 26.45% above its measured warm runtime. A confirmed costing gap is
constant multiplication: the baseline backend emits shifts where the selected
program invokes generic multiply CIs. Results, commands, and source/binary
hashes are recorded in `build/aps-octree-chipyard/summary.json` and
`commands.json`. The run required the compiler-owned-buffer alignment fix and
the Chipyard void-completion fix; both are included in the validated state.

The current flow costs proven constant multiplies as native shifts/negates and
uses a six-cycle command boundary calibrated with dependent CI chains on Rocket.
It automatically selects one eight-way pipelined, cyclically banked clearing
instruction, removing 64,128 scalar multiply CI calls per application call.
The adapter proves that clearing completely overwrites its buffer and omits
the 64-KiB input DMA. Partial and conditional writes retain their input copies.
The selected area proxy is 1245.93 at the same area budget and scheduling period.

Full Chipyard measurements with the current selection are:

| Run | Software baseline cycles | Selected-program cycles | Speedup |
| --- | ---: | ---: | ---: |
| Cold | 1,872,789 | 1,649,725 | 1.135× |
| Warm | 1,855,586 | 1,624,970 | 1.142× |

Both binaries pass every output comparison and checksum on both calls. The
baseline ELF is byte-identical to the earlier measurement. Warm runtime is
12.43% below the software baseline and 16.70% below the earlier accelerated
program. ScaleISA estimates 1.103×; standalone RTL measures 23,569 cycles for
the bulk CI versus 34,834 previously. Its current 18,464-cycle model still
underestimates DMA control overhead, while conservatively retaining the input
import charge. Full-processor results and hashes are in
`build/aps-octree-optimized-chipyard/summary.json`; the investigation and
regression evidence are in `build/aps-octree-optimization/comparison.json`.

For a smaller application test, run the 16×16 ISS keypoint snapshot:

```bash
.pixi/envs/default/bin/python tests/aps-scaleisa/run_iss_case.py \
  --size=16 --area-budget=10000 --clock=8 --output-dir=build/aps-iss-16-direct
.pixi/envs/default/bin/python tests/aps-scaleisa/run_chipyard_pcl.py \
  --source-dir build/aps-iss-16-direct --output-dir build/aps-iss-16-direct-chipyard
```

Only spatial dimensions, matching loop/type bounds, boundary clamps and symbol
names change from the repository's 40×40 C++/MLIR snapshots. The 5×5 covariance
calculation and 3×3 suppression remain intact. Generated inputs and source
hashes are saved under `input/provenance.json`. The reference checks all 768
point coordinates, 256 responses, 256 keypoint flags and the return signature.
Selection uses the production driver and current Rocket profile. The shared
Chipyard runner obtains the validation harness and output count from the case
report; it times two complete application calls with checking excluded.
Software-only selections require an explicit existing `--soc-dir`; both
binaries contain zero CIs, and the SoC's accelerator remains unused. For CI
selections, reuse requires byte-identical selected RTL. A software-only ratio
measures software rewrites and must not be reported as accelerator speedup.
The Rocket profile includes measured f32 sign and ordered-comparison costs.
`fixtures/rocket_fp_control_calibration.cpp` measures dependent sign chains and
integer/FP move-comparison pairs on full Rocket RTL. The model charges four
cycles for each sign or simple comparison; composite predicates retain their
multi-instruction recipes. The calibration records raw counters and its policy
in `hardware/performance/rocket-rv32.json`.

The 16×16 ISS case automatically selects three loop AUs at area budget 10000:
covariance-row computation, suppression-row computation, and final summary.
The selected primitive area proxy is 9076.74. The model estimates 958452 software
cycles versus 348648 selected-program cycles (2.749×). Native Spike + Verilator
cosimulation checks all 1280 output values and return signature 2013; the
normalized zero-CI software baseline passes the same checks.

Private scalar scratch remains inside extracted loop helpers when an unconditional
per-iteration initialization dominates all reads. Materialization preserves the
identity of each allocation across repeated accesses. External scalar references
use one-element scratchpad transfers, including writeback and DMA alignment.
Call-site alias validation and the configured register/CSR operand limit remain
mandatory. The target profile declares the actual extra-CSR count, and the native
driver verifies that it matches its CLI setting. The Chipyard runner derives CSR
bindings from the generated compiler arguments.

Direct and fully unrolled scalar-state implementations are supported. Elastic
pipeline implementations that write scalar global state remain excluded because
the current APS-to-CMT2 pipeline body cannot lower those writes. This restriction
is checked per implementation; it leaves supported implementations available for
selection. Standalone RTL measures 159413 covariance CI cycles over 16 calls,
48396 suppression CI cycles over 16 calls, and 3215 summary CI cycles for one
call. These measurements include CI execution, while the full Chipyard report
also includes processor, cache, DMA and fence overhead. Model estimates and
measured full-processor speedups are reported separately.

The selected functions, target rejection reasons, per-CI model/RTL comparison and
correctness results are in `build/aps-iss-16-direct/summary.json`. Full processor
results are written to `build/aps-iss-16-direct-chipyard/summary.json`.

Full Chipyard (`APSRocketConfig`, `run-binary-debug`, `LOADMEM=1`) passes both
calls of both binaries, checking every output value and signature. The same
newly generated SoC runs both programs, including all five generated CSR
bindings:

| Run | Software baseline cycles | Selected-program cycles | Speedup |
| --- | ---: | ---: | ---: |
| Cold | 849054 | 227790 | 3.727× |
| Warm | 840039 | 225267 | 3.729× |

These are measured processor-cycle speedups at the selected 8-ns scheduling
constraint; they do not establish a post-layout clock frequency. The comparison
uses a fresh normalized software baseline built with the same compiler and
harness. The model estimates 2.749×. The fix and regression evidence are collected
in `build/aps-iss-16-direct/fix-summary.json`.

Additional small PCL stencil cases use the same production pipeline and Rocket
profile:

```bash
.pixi/envs/default/bin/python tests/aps-scaleisa/run_pcl_cases.py \
  normal_estimation --size=12 --area-budget=10000 --clock=8 \
  --output-dir=build/aps-normal-12
.pixi/envs/default/bin/python tests/aps-scaleisa/run_pcl_cases.py \
  morphological_filter --size=12 --area-budget=10000 --clock=8 \
  --output-dir=build/aps-morphological-12
```

The normal case checks all point coordinates, normals, smoothed normals and
curvatures (1440 floating-point values). Morphology checks all four buffers
after its four-pass open/close pipeline (576 floating-point values). Both also
check the return signature against the C++ reference. Only spatial dimensions,
matching bounds, clamps and symbols are scaled; neighborhood sizes and
computations are preserved. An output observer identifies the snapshot's entry
allocations explicitly because several buffers have identical types. Its call
is present in both the selected program and the normalized zero-CI baseline.
The runner requests allocation indices 3, 2, 1, 0 and verifies their actual
contents against the corresponding reference buffers. Input and reference hashes
are recorded in each `input/provenance.json`.

After correctness passes, measure each case with `run_chipyard_pcl.py` using its
`--source-dir` and a separate `--output-dir`. The runner builds the selected SoC
with its generated CSR bindings and times the complete application twice in
each binary, excluding reference generation and output checking.

Both 12×12 cases pass standalone RTL cosimulation, zero-CI baseline validation,
and full Chipyard checking of every value and signature on both cold and warm
calls of both binaries:

| Case | AUs / static CI sites | Area proxy | Model speedup | Cold Chipyard speedup | Warm Chipyard speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normal estimation | 2 / 2 | 5963.92 | 1.510× | 1.652× | 1.629× |
| Morphological filtering | 3 / 5 | 3107.67 | 5.635× | 3.422× | 3.398× |

Normal estimation's warm baseline/accelerated runtimes are 136025/83511 cycles;
morphology's are 525669/154694 cycles. Normal estimation selects the smoothing
row and final reduction; morphology selects erosion, dilation and final
reduction. Dynamic CI counts are 13 and 49 respectively. The reference signatures
are 97464 and 3571. The comparison and source hashes are recorded in
`build/aps-pcl-stencils-12/summary.json`; each case also has its native and
Chipyard `summary.json` reports.

The morphology run exercises singleton memory across multiple functions.
Singleton promotion runs at module scope so checking for conditional writes
cannot inspect functions undergoing concurrent rewrites. The regression checks
that a conditional writer prevents promotion of all accesses to its shared
one-element global while unrelated unconditional storage still promotes.

Small search cases use the same runner:

```bash
.pixi/envs/default/bin/python tests/aps-scaleisa/run_pcl_cases.py \
  kdtree_search --points=32 --queries=4 --max-extra-csrs=8 \
  --output-dir=build/aps-knn-32
.pixi/envs/default/bin/python tests/aps-scaleisa/run_pcl_cases.py \
  registration_icp --points=32 --iterations=4 --max-extra-csrs=8 \
  --output-dir=build/aps-icp-32
```

These retain top-8 insertion for nearest-neighbor search and the full ICP
nearest-correspondence and translation-update algorithm. The default area
budget is 10000 and scheduling constraint 8 ns. Eight extra CSRs are explicitly
configured for these search runs; this is an interface setting, not an intrinsic
limit on supported arithmetic. Input/reference hashes and dimensions are in
`input/provenance.json`. Native preprocessing normalizes the snapshots' supported
Polygeist row views before discovery; validation instrumentation is added only
after selection.

The native flow emits `design.scheduled-tor.mlir` and records
`backend_pipeline_outcomes` and `backend_pipeline_fallback_count`. A requested
pipeline that cannot be scheduled emits a warning and records its sequential
realization. Successful RTL export alone does not establish that the selected
pipeline was realized. Compare this report with the model and measured CI cycles.

Both search runs pass full Chipyard validation, checking all 172 KNN values and
199 ICP values plus their return signatures on both calls of both binaries:

| Case | AUs / static CI sites | Model speedup | Cold Chipyard speedup | Warm Chipyard speedup |
| --- | ---: | ---: | ---: | ---: |
| KNN, 32 points / 4 queries / top-8 | 4 / 4 | 1.972× | 2.027× | 1.961× |
| ICP, 32 points / 4 iterations | 2 / 2 | 2.824× | 1.633× | 1.618× |

Warm cycles are 52198 → 26624 for KNN and 195374 → 120733 for ICP. Both report
zero backend pipeline fallbacks. ICP's nine-operand AU needs seven extra CSR
bindings; the eight-CSR configuration admits it and the Chipyard runner preserves
that setting during compilation. The default five-CSR configuration would
exclude it. The comparison is in `build/aps-search-cases-32/summary.json`.

The related rejection/fallback audit is in `build/aps-rejection-audit/audit.md`.
It distinguishes backend restrictions from configurable compilation limits,
records a sequential fallback probe, and documents the constant-SCF unroll and
frontend row-normalization fixes.
