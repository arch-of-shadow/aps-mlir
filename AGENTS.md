# CMT2 Translation Agent Notes

## Mission Context
- Translate scheduled TOR MLIR into transactional CMT2, then rely on the CMT2→FIRRTL→SystemVerilog toolchain for final RTL.
- Input TOR ops carry `ref_starttime`/`ref_endtime`/`dump`, which feed scheduling-driven rule construction.
- Output must integrate with existing CMT2 analyses, inliners, and conversion pipeline to guarantee legal firing semantics and compatibility with downstream passes.

## High-Value References
- `circt/CMT2.md`: End-to-end status of dialect features, analyses, and required passes.
- `docs/TOR_TO_CMT2_IMPLEMENTATION.md`: Phase-by-phase TOR→CMT2 implementation playbook (schedule analysis, dataflow, FIFO strategy, module library expectations, verification flow).
- `circt/docs/Dialects/Cmt2/RationaleCmt2.md`: Dialect design rationale; concise definitions of modules, rules, methods, values, instances, interfaces, call ops, and return semantics.
- `circt/docs/Dialects/Cmt2/ModuleLibrary.md`: FIRRTL module library layout and manifest expectations for reusable FIFOs/regs/BRAMs.
- `circt/docs/Dialects/Cmt2/INTERFACE_HELPERS_SUMMARY.md`: Quick reference for interface declaration/binding helpers (handy when wiring TOR interfaces).
- `circt/docs/Dialects/Cmt2/ecmt2-EDSL.md` & `ecmt2-Class-API.md`: Optional—describe embedded DSL that mirrors dialect structure; useful sanity check for expected op patterns.

## CMT2 Dialect Essentials
- **Modules**: `cmt2.module` (internal) and `cmt2.module.extern.firrtl` (bindings to FIRRTL externs) expose clock/reset, instances, callable entities, and precedence metadata.
- **Callable Entities**: `cmt2.rule`, `cmt2.method`, `cmt2.value`, plus `cmt2.bind.method`/`cmt2.bind.value` for externals. All implement `Cmt2FunctionLike` (two regions: guard + body). Guards compute readiness; bodies perform state updates or value production.
- **Calls**: `cmt2.call @instance @callee(...)` obey CallOpInterface. Interface indirection (InterfaceDecl/InterfaceDef) must be resolved when emitting calls.
- **Instances & Interfaces**: `cmt2.instance` creates submodules; interface ops define abstract method bundles bound to real instances. Analyses expect `@this` to reference self-instance.
- **Conflict Semantics**: Conflicts (`<>`), sequential-before (`<`), and conflict-free (`/`) relationships are declared on externs and inferred on modules. Scheduler uses these plus precedence attributes to order firings.

## Core Analyses/Transforms (must stay compatible)
- **InstanceGraph**: Tracks module instantiation hierarchy; needed for bottom-up analyses and inlining.
- **CallInfo**: Maps rule/method/value calls to callee instances (including interface resolution); required when wiring FIFO interactions and translating TOR dependencies.
- **ConflictMatrix**: Computes pairwise relationships using call sets and callee conflicts; translation must preserve/annotate necessary metadata so existing inference still holds.
- **Module Inliner**: Inlines non-top/non-extern modules; ensure generated modules respect synthesis/extern flags.
- **PrivateFunc Analyzer + Inliner**: Private callable detection; avoid generating unnecessary private indirections unless simplicity demands.
- **Scheduler Analysis**: Groups functions, enforces precedence, and warns about preventing-firing scenarios. Generated modules should include precedence hints if TOR schedule imposes strict ordering beyond FIFO dependencies.

## TOR→CMT2 Strategy (from implementation guide)
1. **Schedule Extraction**: Group TOR ops by identical `ref_starttime`; produce ordered `ScheduledBlock` records with cycle range metadata.
2. **Dataflow Dependencies**: Compute producer→consumer edges, track value sets, and quantify latency (`consumer.start - producer.end`).
3. **FIFO Planning**: Instantiate FIFO extern modules (depth/width from latency and payload) and connect producer/consumer rules via enqueue/dequeue methods.
4. **Rule Synthesis**:
   - Guard region checks data availability (`fifo.notEmpty`, `fifo.notFull`, memory handshake readiness) plus TOR guard predicates.
   - Body region performs actual data movement / computation and enqueues downstream payloads.
   - For parallel TOR blocks (same start cycle but independent), emit multiple conflict-free rules guarded by shared FIFOs.
5. **External Library Usage**: Leverage module library manifest for FIFOs/regs/mems. Bind methods/values based on manifest definitions and attach provided conflict matrices.
6. **Interface Wiring**: Map TOR-level interface ops (e.g., burst loaders) onto CMT2 interfaces or direct instance calls per `INTERFACE_HELPERS_SUMMARY.md` guidance.
7. **Metadata Preservation**: Propagate schedule-derived precedence into module `precedence` attribute when stronger ordering than FIFO constraints exists.
8. **Verification Hooks**: Plan test passes (`-cmt2-print-...`) to validate CallInfo, ConflictMatrix, Scheduler, and inline passes on generated MLIR before lowering.

## Testing & Tooling Cheatsheet
- `circt/build/bin/circt-opt <file> -cmt2-print-call-info` — sanity-check resolved calls.
- `circt/build/bin/circt-opt <file> -cmt2-print-conflict-matrix` — inspect inferred conflicts post-translation.
- `circt/build/bin/circt-opt <file> -cmt2-inline-modules` — ensure inliner handles generated hierarchy.
- `circt/build/bin/circt-opt <file> -cmt2-inline-private-funcs` — inline private helpers prior to scheduling.
- `circt/build/bin/circt-opt <file> -cmt2-print-scheduler` — view scheduling groups/ordering.
- Full lowering pipeline:
 ```bash
  circt/build/bin/circt-opt translated.mlir \
    -cmt2-inline-private-funcs \
    -cmt2-verify-private-funcs-inlined \
    -cmt2-verify-call-sequence \
    --lower-cmt2-to-firrtl | \
    circt/build/bin/firtool --format=mlir --disable-reg-randomization
  ```

## Workflow Hygiene
- Invoke CIRCT binaries via `circt/build/bin/...`; keep APS-related commands scoped outside the `circt` directory.
- APS utilities live under `build/tools`; prefer absolute path `./build/tools/<tool>` from project root and avoid running them from within `circt/`.

## Outstanding Questions / Follow-Ups
- Depth heuristics for FIFOs derived from latency—need final policy (min depth vs. exact latency).
- Strategy for TOR loops with overlapping lifetimes: flatten into multiple rules vs. parameterized rule families.
- Handling memory/burst interfaces: confirm available extern modules and their conflict matrices in manifest.
- Decide whether to emit explicit precedence edges or rely solely on FIFO guards for schedule fidelity.
- Confirm how multi-cycle operations (different `ref_endtime`) map into single rule vs. chained rules.
