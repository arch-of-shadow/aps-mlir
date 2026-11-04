# APS-MLIR — Agent Guide

Hardware/software co-design framework for RISC-V instruction extensions. CADL and
adapted ScaleISA kernels lower through APS, scheduled TOR, CMT2, FIRRTL, and
SystemVerilog. Megg and the ScaleISA software compiler generate matching CPU code.
Dependencies and commands are managed by Pixi; LLVM/MLIR/CIRCT live under `circt/`.

## Working agreement

- Carry action requests through implementation, relevant validation, and fixes
  for failures caused by the change. Do not stop at a plan or first draft unless
  that is the requested deliverable.
- Proceed with authorized local reads, edits, builds, and tests without repeated
  approval. Ask only when missing information materially affects correctness or
  scope; complete independent work while resolving blockers.
- Preserve existing user changes. Local commits for completed, validated work
  within the requested scope are authorized by the Git rule below. Push,
  publish, deploy, or delete work outside the requested scope only with
  authorization. A workflow mentioning a PR is not itself authorization to
  publish it.
- Read the files and contracts relevant to the task. The index below is a router,
  not a requirement to read every document before editing.
- Skills supply task-specific guidance within higher-priority instructions and
  the user's authorized scope. If a skill blocks or redirects requested work,
  cite its path and exact rule; distinguish the rule from your interpretation.
- Completion means the requested result is present, applicable checks have run,
  introduced failures are fixed, and required documentation is current. Report
  any unresolved blocker or skipped check explicitly, without claiming success.
- Keep responses concise. Source comments, documentation, run reports, and
  feature summaries are English. Record significant design decisions with
  rationale for the current approach.

## Architecture and constraints

- `cadl_frontend/`: Lark grammar, `cadl_ast.py`, parser, `to_mlir/` emitters, and
  `to_c/` transpiler. `tools/aps-frontend` owns the frontend CLI. Grammar acceptance
  alone does not establish MLIR or hardware support.
- `include/APS/`, `lib/APS/`: functional/architectural memory operations,
  interface binding, memory scheduling, scratchpad elision, storage planning,
  and CMT2 generation. `include/TOR/`, `lib/TOR/`, and `lib/Schedule/` own TOR,
  loop/array transformations, dependence analysis, and SDC scheduling.
- `tools/aps-opt/main.cc` owns the production `aps-e2e` pass order and export
  stages. `aps-opt` exposes individual passes. Preserve scheduled timing,
  memory dependencies, request/completion ordering, and loop-carried semantics.
- `lib/APS/APSToCMT2/` owns block/slot/loop lowering, operation generators,
  adapters, and memory pools. Preserve CMT2 method readiness, conflict and
  precedence semantics, queue contracts, and instruction completion. Shared
  execution state must remain protected under stalls and overlapping requests.
- `megg/` matches supplied CADL instruction semantics against application code
  using e-graphs. `tools/aps-megg` owns its end-to-end CLI. `scaleisa/` discovers
  and selects instructions; `lib/scaleisa-adapter/` materializes APS hardware
  contracts; `lib/scaleisa-compiler/` lowers matching CI sites to RV32 code.
  `tools/aps-scaleisa-opt/` owns driver orchestration and the dedicated pass tools.
  Hardware encodings, operand/CSR bindings, alias promises, and software ABI must agree.
- `hardware/hardfloat/` owns BF16/FP32 IP packaging and timing contracts.
  `tools/aps-cosim/` provides Spike + Verilator execution; `thirdparty/chipyard/`
  supplies full Rocket integration. Model estimates, accelerator cycles, processor
  cycles, and physical timing are distinct evidence.
- Run APS commands from the repository root using `build/tools/...`; invoke CIRCT
  tools through `circt/build/bin/...`. Use Pixi for dependencies and activation.
  `pixi run build` owns APS setup and the release ScaleISA engine build.
  Build HardFloat IP afterward because its importer requires the built firtool.
- Keep generated artifacts under `build/` or the owning component's documented
  output directory, such as `scaleisa/tmp/`. Keep contributor-local absolute paths
  out of active docs and configuration. Follow scoped instructions in `circt/`
  and `scaleisa/` when modifying those submodules; preserve unrelated submodule work.
- C++ follows LLVM conventions; Python follows PEP 8 with public type annotations;
  Rust uses `cargo fmt`. Missing functionality must be explicitly rejected or
  tracked, never shipped as fake/mock behavior.

## Policies and validation

- Run checks appropriate to the affected behavior. Reuse passing results for an
  unchanged tree; broaden or repeat only after relevant changes, failures, or
  unresolved concerns. Multi-agent workflow orchestration requires explicit
  user opt-in.
- Docs/instruction-only edits: relevant path/link, command, and diff inspection.
  Frontend changes: affected parser/emitter tests. APS/TOR/scheduling changes:
  affected `tests/test_mlir/` regressions and a relevant lowering example.
  Adapter/compiler/driver changes: the corresponding CMake check targets.
  Memory, ABI, handshake, or completion changes: relevant RTL/co-simulation checks
  with independent output validation. Changes inside submodules follow their
  own scoped validation requirements.
- Common commands (run from the repository root):
  - Setup/build: `pixi run build`
  - Incremental tools: `pixi run cmake --build build --target aps-opt aps-e2e aps-scaleisa-opt`
  - Frontend: `pixi run python -m pytest tests/test_parser tests/test_mlir/test_mlir_conversion.py -q`
  - Megg: `pixi run python -m pytest tests/test_megg_csr.py tests/test_megg_csr_robustness.py tests/test_megg_cleanup.py -q`
  - ScaleISA integration: `pixi run cmake --build build --target check-aps-scaleisa check-scaleisa-adapter check-scaleisa-compiler`
  - HardFloat packaging: `pixi run build-hardfloat-ip`
  - HardFloat tests: `pixi run python -m pytest tests/hardfloat -q`
  - Chipyard setup: `pixi run setup-chipyard`
  - Co-simulation: `pixi run cosim-run PATH_TO_ELF PATH_TO_SV`
- Use `--dump-scheduled-tor`, `--dump-cmt2`, and `--print-ir-after-all` on
  `aps-e2e` to isolate lowering failures. Inspect generated CMT2 with
  `circt/build/bin/circt-opt` and `-cmt2-print-call-info`,
  `-cmt2-print-conflict-matrix`, or `-cmt2-print-scheduler` when relevant.
  Complete benchmark, mapped-gate, STA, and Rocket campaigns have separate
  prerequisites; their guides define what each validates.

## Git

Work on `dev` by default. Do not resume `fix/aps-performance-parallelism` or create
another working branch unless the user requests it. Never force-push shared
branches. Commit messages explain why.

After completing and validating a specific feature change, promptly commit the
related changes locally unless the user asks otherwise. Keep each commit focused
on that feature, include its documentation and tests, and exclude unrelated or
pre-existing user changes. Do not commit incomplete work or claim failed validation
as passing; report unresolved validation failures before committing. This rule
does not authorize pushing or publishing changes.

## Docs index

| Task | Reference |
|---|---|
| Overview and navigation | [docs/README.md](docs/README.md) |
| Setup/build and first RTL export | [Synthesis overview](docs/SYNTHESIS/README.md) |
| CADL syntax, control flow, and frontend | [CADL language](docs/SYNTHESIS/cadl-language.md) |
| Production pipeline and exports | [End-to-end flow](docs/SYNTHESIS/aps-e2e-flow.md) |
| Affine, partitioning, scheduling, elision, storage | [HLS passes](docs/SYNTHESIS/hls-passes.md) |
| TOR/CMT2 tokens, values, loops, and completion | [CMT2 lowering](docs/SYNTHESIS/aps-to-cmt2-transfer-model.md) |
| BF16/FP32 packaging and acceptance | [HardFloat guide](docs/SYNTHESIS/hardfloat-bf16-fp32-acceptance.md) |
| Spike + Verilator execution | [Co-simulation](docs/SYNTHESIS/aps-spike-verilator-cosim.md) |
| Megg matching, rewrites, and extraction | [Megg internals](docs/COMPILER/pattern-matching.md) |
| ScaleISA orchestration and benchmarks | [APS ScaleISA driver](docs/COMPILER/aps-scaleisa-opt.md) |
| Selected kernels to hardware ABI | [ScaleISA adapter](docs/COMPILER/scaleisa-adapter.md) |
| Selected CI sites to RV32 binaries | [Software compiler](docs/COMPILER/scaleisa-compiler.md) |
| Processor configuration and Rocket execution | [Integration overview](docs/ARCHITECTURE/README.md) |
| Modeled memory, DMA, and ordering | [Memory interfaces](docs/ARCHITECTURE/aps-itfc.md) |
| Generated ports, CSRs, and void completion | [Port contract](docs/ARCHITECTURE/aps-port-convention.md) |
| Discovery, anti-unification, and selection internals | [ScaleISA docs](scaleisa/docs/README.md) |
| CMT2 dialect semantics and tools | [CMT2 docs](circt/docs/Cmt2/_index.md) |

## Engineering Principles

- **No backward compatibility**: remove obsolete paths instead of adding compatibility layers, fallbacks, or migrations.
- **Simplicity first**: choose the simplest implementation that fully meets current requirements; avoid speculative abstractions, configuration, and indirection.
- **Grow in working layers**: start with the smallest end-to-end version that works, then add capabilities on top of a working product.
- **Modular design**: keep components modular and concerns clearly separated.
- **Prefer established libraries**: use well-maintained libraries when they reduce overall complexity or improve reliability.
- **Use existing dependencies first**: check the documentation and types of dependencies before writing a custom implementation or adding packages.
- **Design for the long term**: make durable architectural decisions rather than accepting stopgaps intended to be replaced later.
- **Correctness before coverage**: reject unsupported cases explicitly rather than producing silent wrong results.
- **Stage discipline**: preserve functional, architectural, scheduled, and transactional contracts at their respective lowering boundaries.

## Documentation

- Write documentation in English.
- Code and documentation must evolve together: update documentation before
  writing code, and for large changes review both for redundant or outdated content.
- Explain how to use or implement the current approach and why it is appropriate.
  When removing functionality, update documentation to describe current behavior.
- Keep documentation compact and indexed in [docs/README.md](docs/README.md).
  Consolidate overlapping guides and link implementation/tests or component docs
  instead of duplicating them. Keep benchmark results and provenance in `reports/`.
