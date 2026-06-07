# aps-e2e Synthesis Flow

This document records the synthesis flow that the current APS-MLIR refactor must preserve. The important product boundary is `aps-e2e`: all APS/TOR/CMT2 restructuring should keep this end-to-end path working first.

## Top-Level Flow

The tutorial synthesis script uses three stages:

```bash
pixi run mlir tutorial/cadl/v3ddist_vv.cadl tutorial/outputs/v3ddist_vv.mlir
pixi run opt tutorial/outputs/v3ddist_vv.mlir tutorial/outputs/v3ddist_vv_cmt.mlir
pixi run sv tutorial/outputs/v3ddist_vv_cmt.mlir tutorial/outputs/v3ddist_vv.sv
```

The stages expand to:

```text
CADL
  -> APS MLIR
  -> aps-e2e fixed pipeline
  -> CMT2 MLIR
  -> lower-cmt2-to-firrtl
  -> firtool
  -> SystemVerilog
```

Source entry points:

| Stage | Command | Implementation |
| --- | --- | --- |
| CADL to APS MLIR | `pixi run mlir <cadl> <mlir>` | `pixi.toml` task `mlir`, `aps-frontend mlir` |
| APS/TOR to CMT2 | `pixi run opt <mlir> <cmt2>` | `pixi.toml` task `opt`, `build/tools/aps-opt/aps-e2e` |
| CMT2 to SV | `pixi run sv <cmt2> <sv>` | `circt-opt --lower-cmt2-to-firrtl | firtool` |

## aps-e2e Driver

`aps-e2e` is built from `tools/aps-opt/main.cc`. It is not a general pass driver; it constructs a fixed command line and then calls the shared `aps_opt_driver`.

The driver requires:

- `--input`: input APS MLIR file.
- `--output`: output CMT2 MLIR file.
- `--clock`: schedule clock period, default `6.0`.
- `--resource`: scheduling resource JSON, default `examples/resource_ihp130.json`.
- `--print-ir-after-all`: forwards MLIR pass dumps for debugging.

The actual pass registration is in `tools/aps-opt/aps_opt_driver.cc`. It registers MLIR dialects, TOR, APS, CIRCT comb, and CIRCT CMT2, then registers TOR/APS/CMT2 passes.

## Fixed Pass Pipeline

`aps-e2e` currently runs this sequence:

```text
--place-readrf-at-entry
--aps-memory-map
--normalize-scf-for-indices
--aps-mem-to-memref
--canonicalize
--raise-scf-to-affine
--canonicalize
--affine-raise-from-memref
--raise-memref-to-affine
--canonicalize
--hls-unroll
--cse
--canonicalize
--affine-loop-normalize
--canonicalize
--new-array-partition
--canonicalize
--affine-mem-to-aps
--memref-to-aps
--promote-singleton-memref-to-global
--arith-muldiv-to-shift
--canonicalize
--lower-affine-for
--canonicalize
--expression-balance
--convert-input=clock=<clock> resource=<resource> output-path=<tmp>
--canonicalize
--scf-to-tor
--canonicalize
--schedule-tor
--lower-aps-mem-to-req-collect
--tor-time-graph
--duplicate-memloads
--canonicalize
--aps-to-cmt2
```

For refactoring, treat this sequence as the compatibility contract unless we explicitly decide to change the flow.

## Stage Responsibilities

### 1. CADL Frontend Output

The frontend emits a `module` containing:

- `func.func` operations for CADL instructions.
- `opcode` and `funct7` attributes on instruction functions.
- APS register file operations: `aps.read_irf`, `aps.write_irf`.
- APS scratchpad/bulk memory operations: `aps.read_smem`, `aps.write_smem`, `aps.copy`.
- `memref.global` declarations for static memories, often with partition metadata.

Relevant files:

- `include/APS/APSOps.td`
- `lib/APS/aps.cpp`

### 2. Early APS and Affine Normalization

This part prepares the input for HLS-style transforms:

- `place-readrf-at-entry` moves register-file reads to function entry so scheduling can place them early.
- `aps-memory-map` creates `aps.memorymap` with `aps.mem_entry` records for global memories.
- `normalize-scf-for-indices`, `aps-mem-to-memref`, affine raising, and affine inference normalize loops and memory access patterns.
- `hls-unroll`, `affine-loop-normalize`, and `new-array-partition` expose parallel memory access and transform globals.
- `affine-mem-to-aps` and `memref-to-aps` return memory operations to APS-specific hardware operations.
- `promote-singleton-memref-to-global` converts scalar single-element memories into `aps.globalload` and `aps.globalstore`.
- `arith-muldiv-to-shift` simplifies power-of-two arithmetic before scheduling.

Relevant files:

- `lib/APS/PlaceReadRFAtEntry.cpp`
- `lib/APS/APSMemoryMap.cpp`
- `lib/APS/APSMemToMemRef.cpp`
- `lib/APS/AffineMemToAPS.cpp`
- `lib/APS/MemRefToAPS.cpp`
- `lib/APS/PromoteSingletonMemRefToGlobal.cpp`
- `lib/TOR/HlsUnroll.cpp`
- `lib/TOR/NewArrayPartition.cpp`
- `lib/TOR/RaiseToAffine.cpp`

### 3. TOR Conversion and Scheduling

This stage moves from software-like control flow to scheduled TOR:

- `convert-input` prepares scheduling inputs using the clock period and resource database.
- `scf-to-tor` converts SCF structure into TOR design/function/control operations.
- `schedule-tor` assigns timing attributes such as `ref_starttime`, `ref_endtime`, and `dump`.
- `lower-aps-mem-to-req-collect` splits memory operations into request/collect pairs where needed.
- `tor-time-graph` derives timegraph information from scheduled attributes.
- `duplicate-memloads` duplicates loads when doing so can avoid cross-cycle FIFO storage.

The CMT2 generator relies on these timing attributes. A rule can only be synthesized correctly if every relevant operation has a legal scheduled cycle.

Relevant files:

- `include/TOR/Passes.td`
- `lib/TOR/ConvertInput.cpp`
- `lib/TOR/SCFToTOR.cpp`
- `lib/TOR/TORSchedulePass.cpp`
- `lib/TOR/TORTimeGraphPass.cpp`
- `lib/APS/LowerAPSMemToReqCollect.cpp`
- `lib/APS/MemLoadDuplication.cpp`
- `lib/Schedule/`

### 4. APS/TOR to CMT2 Generation

The final APS pass is `aps-to-cmt2`. It replaces the module body with generated CMT2 circuit/module/interface operations.

Top-level responsibilities:

- Load CIRCT CMT2 module library manifest from `circt/lib/Dialect/Cmt2/ModuleLibrary/manifest.yaml`.
- Create a CMT2 `Circuit`.
- Add burst DMA, RoCC response, and HellaCache command interfaces.
- Generate `ScratchpadMemoryPool` from `aps.memorymap`.
- Generate RoCC and memory adapter modules.
- Generate the `main` CMT2 module.
- For every `tor.func`, generate CMT2 rules from scheduled TOR operations.
- Materialize the CMT2 MLIR and replace the original module body.

Relevant files:

- `include/APS/APSToCMT2.h`
- `lib/APS/APSToCMT2/APSToCMT2.cpp`
- `lib/APS/APSToCMT2/APSMemoryGenerate.cpp`
- `lib/APS/APSToCMT2/APSInterfaceGenerate.cpp`
- `lib/APS/APSToCMT2/APSGlobalRegisterGenerate.cpp`
- `lib/APS/APSToCMT2/APSOpRuleGenerate.cpp`

## CMT2 Rule Generation Structure

The current rule generator is already partly refactored into handlers and operation generators.

```text
APSToCMT2Pass::generateRulesForFunction
  -> BlockHandler::processFunctionAsBlocks
    -> block segmentation and cross-block FIFO/token setup
    -> LoopHandler for tor.for blocks
    -> BBHandler for regular blocks
      -> collect scheduled ops by time slot
      -> build cross-slot FIFOs
      -> generate cmt2.rule per slot
      -> dispatch operation-specific generation
```

Handler roles:

| Component | Role |
| --- | --- |
| `BlockHandler` | Splits function/control regions into blocks, creates cross-block value FIFOs and token FIFOs, delegates block processing. |
| `LoopHandler` | Builds canonical loop entry/body/next coordination and loop state FIFOs/registers. |
| `BBHandler` | Groups operations by schedule slot, builds cross-slot FIFOs, emits slot rules. |
| `ArithmeticOpGenerator` | Emits FIRRTL/CMT2 logic for arith/comb-like operations. |
| `MemoryOpGenerator` | Emits scratchpad memory calls, burst load/store calls, and global memory register interactions. |
| `InterfaceOpGenerator` | Emits interface memory request/collect calls. |
| `RegisterOpGenerator` | Emits RoCC register-file read/write behavior. |

Refactor priority should keep these boundaries visible. If code moves, preserve the same responsibility split or document the replacement.

## Key IR Milestones

### APS Input

Typical frontend output:

```mlir
func.func @flow_v3ddist_vv(%arg0: i5, %arg1: i5, %arg2: i5)
    attributes {funct7 = 40 : i32, opcode = 11 : i32} {
  %0 = aps.read_irf %arg0 : i5 -> i32
  %mem = memref.get_global @points1_x : memref<16xi32>
  aps.copy %0, (%mem)[%c0_i32], %c16_i32
      : i32, (memref<16xi32>), i32, i32
  scf.for ...
  aps.write_irf %arg2, %c0_i32 : i5, i32
  return
}
```

### Scheduled TOR

After `schedule-tor`, TOR/APS operations are expected to carry schedule metadata such as:

```mlir
{ref_starttime = ..., ref_endtime = ..., dump = ...}
```

The exact op syntax depends on TOR lowering, but this metadata is the contract consumed by `BBHandler` and rule generation.

### CMT2 Output

After `aps-to-cmt2`, the module contains:

```mlir
cmt2.circuit {
  cmt2.interface @BurstDMAController { ... }
  cmt2.interface @roccRespItfc { ... }
  cmt2.interface @hellaCmdItfc { ... }
  cmt2.module @ScratchpadMemoryPool(%clk: !firrtl.clock, %rst: !firrtl.uint<1>) { ... }
  cmt2.module @main(%clk: !firrtl.clock, %rst: !firrtl.uint<1>) { ... }
}
```

This CMT2 output must remain compatible with CIRCT CMT2 analyses, inlining, verification, and lowering.

## Verification Commands

Run the full synthesis flow:

```bash
pixi run mlir tutorial/cadl/v3ddist_vv.cadl /tmp/v3ddist_vv.mlir
pixi run opt /tmp/v3ddist_vv.mlir /tmp/v3ddist_vv_cmt.mlir
pixi run sv /tmp/v3ddist_vv_cmt.mlir /tmp/v3ddist_vv.sv
```

Dump all intermediate IR from `aps-e2e`:

```bash
build/tools/aps-opt/aps-e2e \
  -i /tmp/v3ddist_vv.mlir \
  -o /tmp/v3ddist_vv_cmt.mlir \
  --clock 6.0 \
  --resource examples/resource_ihp130.json \
  --print-ir-after-all \
  2> /tmp/aps-e2e-ir.log
```

Inspect CMT2 after generation:

```bash
circt/build/bin/circt-opt /tmp/v3ddist_vv_cmt.mlir -cmt2-print-call-info
circt/build/bin/circt-opt /tmp/v3ddist_vv_cmt.mlir -cmt2-print-conflict-matrix
circt/build/bin/circt-opt /tmp/v3ddist_vv_cmt.mlir -cmt2-print-scheduler
```

Check the downstream lowering path:

```bash
circt/build/bin/circt-opt /tmp/v3ddist_vv_cmt.mlir \
  -cmt2-inline-private-funcs \
  -cmt2-verify-private-funcs-inlined \
  -cmt2-verify-call-sequence \
  --lower-cmt2-to-firrtl | \
  circt/build/bin/firtool --format=mlir --disable-reg-randomization
```

## Refactor Guardrails

- Keep `aps-e2e` green before optimizing individual passes.
- Treat `tools/aps-opt/main.cc` as the source of truth for the production pass order.
- Preserve scheduled TOR metadata needed by CMT2 rule generation.
- Keep `aps.memorymap` available to `aps-to-cmt2`.
- Keep CMT2 call/interface generation compatible with CallInfo, ConflictMatrix, scheduler, and CMT2 inliners.
- Prefer small separations by responsibility: pass driver, memory pool generation, adapter/interface generation, block/loop/basic-block rule generation, and operation-specific emitters.
- When changing pass order, record the old and new IR milestone that justifies it.
