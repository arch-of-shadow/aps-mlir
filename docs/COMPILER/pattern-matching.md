# Megg matching and e-graph compilation

Megg compiles C against supplied custom-instruction semantics. Its active path is
[tools/aps-megg](../../tools/aps-megg) →
[E2ECompiler](../../megg/megg/e2e_compiler.py) →
[Compiler](../../megg/megg/compiler.py) → LLVM/RISC-V emission.
The [compiler overview](README.md) gives build and invocation commands.

## Select application code

Mark functions intended for Megg optimization:

```c
#pragma megg optimize
int sum(int a, int b) {
  return a + b;
}
```

[PragmaParser](../../megg/megg/frontend/pragma_parser.py) selects and separates
these functions. E2ECompiler sends marked code through Polygeist and Megg and
compiles the remaining code through the ordinary LLVM path, then links them.
With no marked functions, it compiles the source directly; a successful executable
alone does not prove that a custom instruction matched.

The CADL wrapper generates pattern C, pattern MLIR, and encoding JSON first.
CSR operands are included in the encoding contract. Match computation, software
encoding, and generated accelerator behavior must agree.

## Terms and roots

[FuncToTerms](../../megg/megg/egraph/func_to_terms.py) translates supported MLIR
functions into typed egglog terms, defined by
[term.py](../../megg/megg/egraph/term.py) and
[datatype.py](../../megg/megg/egraph/datatype.py). Block/control-flow roots preserve
the function structure needed for reconstruction; the graph includes effects
and loop structure as well as arithmetic expressions.

Egglog e-classes contain equivalent expressions. Internal rewrites add equivalent
forms before instruction matching. Type annotations participate in that process;
preserve widths, signed operations, casts, and memory structure when extending
the term vocabulary. Unsupported types/operations must fail explicitly.

## Current optimization sequence

`Compiler.schedule()` performs these phases:

1. Initialize function e-graphs and statistics.
2. Apply internal algebraic, constant-folding, and type-annotation rules.
3. Apply enabled external loop transformations, guided by extracted pattern hints.
4. Simplify again after successful external transformations.
5. Apply custom-instruction rewrite rules.
6. Apply structural skeleton matching for complex instructions.
7. Extract expressions and reconstruct complete functions.

[internal_rewrites.py](../../megg/megg/rewrites/internal_rewrites.py) defines the
actual type-specific laws; do not assume every algebraic identity applies to
every floating or integer type. External transformations are dispatched through
[external_loop_pass.py](../../megg/megg/rewrites/external_loop_pass.py), using the
compiled MLIR utilities. The compiler's planner chooses relevant unroll/tile and
other supported loop transformations from the current loop and pattern shape.

## Pattern and skeleton matching

[build_ruleset_from_module](../../megg/megg/rewrites/match_rewrites.py) normalizes
pattern functions through the term/extraction pipeline, then derives instruction
rewrites and structural skeletons. Leaf computations can match through equivalent
e-graph expressions. Skeletons retain control-flow and statement relationships,
with operand constraints connecting their components.

[SkeletonMatcher](../../megg/megg/egraph/megg_egraph.py) checks those relationships
against the serialized graph and inserts custom-instruction alternatives. A local
component match is not enough to replace an entire effectful loop: structural and
operand constraints must agree. Used-argument tracking records `operand_indices`
so extraction can compact operands without changing their original meaning.

## Extraction and code generation

[MeggEGraph](../../megg/megg/egraph/megg_egraph.py) provides the Python graph used
for structural matching and extraction.
[extract.py](../../megg/megg/egraph/extract.py) implements costs and best-expression
selection. Costs are compiler heuristics, not measured hardware latency.
[ExprTreeToMLIR](../../megg/megg/egraph/terms_to_func.py) rebuilds functions from
top-level block roots and preserves the surrounding module. Missing children,
invalid roots, and reconstruction failures remain errors.

[llvm_backend.py](../../megg/megg/backend/llvm_backend.py) lowers reconstructed
MLIR and emits custom instructions from encoding metadata. Register constraints
and CSR bindings must preserve operand positions, including unused native register
arguments and instructions without results. Extra inputs require explicit CSR
bindings; they cannot be silently assigned to arbitrary registers.

## Debug and validate

Use `pixi run compile ... --verbose` for diagnostics or `--handson` for phase
snapshots. Inspect `compile_logs/` for pattern MLIR, encoding JSON, optimized
artifacts, assembly, and statistics where produced. Confirm the intended `.insn`
or its disassembly and compare program outputs against an independent reference.
Rewrite counters and snapshot names describe compiler activity; they are not
proof of correct instruction execution.

```sh
pixi run python -m pytest tests/megg/test_csr.py \
  tests/megg/test_csr_robustness.py tests/megg/test_cleanup.py -q
```

These regressions cover matching/reconstruction, CSR operand transport, module
preservation, and failure propagation. The
[tutorial compiler](../../examples/aspdac26-tutorial/compiler) provides the interactive workflow.
