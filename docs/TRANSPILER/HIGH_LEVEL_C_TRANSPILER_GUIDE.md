# High-Level CADL → C Transpiler Guide

This guide explains how to run the high-level CADL-to-C transpiler, what to
expect from the generated output, and how to validate changes locally.

## 1. Overview

`cadl_frontend/transpile_to_c.py` converts CADL flows into pure C functions
without any hardware plumbing. The generated C now keeps track of register
based addressing information so that instruction matching can reason about
operand relationships: register reads that feed memory traffic become typed
pointer parameters, scratchpad bursts fold into direct pointer arithmetic, and
loops are still rewritten into idiomatic C constructs when possible.

Use this tool whenever you need a Polygeist-friendly C view of CADL code, e.g.
for instruction matching or behavioural comparisons with handwritten
implementations.

## 2. Prerequisites

1. Activate the project environment (if you use `pixi`):
   ```bash
   pixi shell
   ```
2. Ensure the CADL source parses with the existing frontend parser. Syntax
   errors are surfaced before any code generation happens.

## 3. Quick Start

Run the module directly and pick an output path. By default the tool emits to
stdout if `-o/--output` is omitted.

```bash
python -m cadl_frontend.transpile_to_c examples/test_array_add.cadl -o array_add.c
```

This produces a translation unit with:

- An auto-generated header comment and the minimal set of `#include`s (only
  `stdint`, `stdbool`, `string`, or `math` depending on actual usage).
- One C function per CADL flow, where `_irf` accesses are rewritten into value
  parameters and the `_mem` pointer is retained only if the flow performs
  memory reads. Burst operations are always removed.

## 4. Reading the Output

### Register handling

- `_irf[rsX]` reads that are only used as scalars still become value
  parameters named `rsX_value` with an inferred C integer type.
- When an `_irf[rsX]` value participates in `_mem[...]` or `_burst_*` accesses,
  the transpiler infers the element type and emits a pointer parameter named
  `rsX` (or similar) instead. All subsequent memory accesses are rewritten to
  index into that pointer, preserving the original addressing relationships.
- Multiple register writes continue to materialise as either a scalar return or
  a struct, as before.
- Register index arguments that never escape `_irf[...]` operations are still
  pruned, and shorthand aliases (`r1`, `r2`, …) reuse the inferred parameter.

### Arrays and memory

- Scratchpad statics that are populated via bursts are no longer emitted as
  function parameters. Instead they are rewritten on the fly to index the
  register-derived base pointer with the appropriate offset.
- Static scratchpad arrays or scalars that never participate in bursts stay
  local inside the generated function—stack allocations with zero
  initialisation replace the previous parameter lifting.
- `_burst_read` / `_burst_write` statements disappear from the final C, leaving
  a short comment to signal that the access was folded into pointer arithmetic.
- `_mem[...]` loads and stores that are anchored on register addresses become
  pointer accesses through the corresponding register parameter, removing the
  need for a global `_mem` pointer whenever the access pattern can be resolved
  statically.

### Control flow

- `with … do … while` loops are analysed for canonical induction patterns. When
  the loop has an affine step and guard, it is emitted as a single C `for`
  loop; otherwise the tool falls back to a guarded `while (1)` form that
  updates bindings at the end of each iteration.

### Type normalisation

- Bit-precise CADL integer types map to the nearest byte-aligned C type
  (`u3 → uint8_t`, `u12 → uint16_t`, `i17 → int32_t`, etc.).
- Arithmetic combines operand widths and upgrades the result type when
  necessary (e.g. `uint16 + uint32 → uint32`).

### Expressions

- `sel { … }` constructs lower to chained ternary expressions, preserving the
  order of CADL select arms and default values.
- Tuple literals currently evaluate each element in order and yield the final
  value via a comma expression (`/* tuple */(a, b, c)`), matching the high-level
  semantics used by Polygeist.

## 5. Workflow Tips

1. Keep CADL statics and flows in the same file—the transpiler pulls statics
   from the surrounding `Proc` definition.
2. Inspect the emitted signature to confirm which register inputs were kept or
   dropped. Missing `_irf` reads in the CADL body mean you should expect fewer
   parameters.
3. When comparing against Polygeist MLIR, feed the generated C directly into
   Polygeist after transpilation:
   ```bash
   python -m cadl_frontend.transpile_to_c my_flow.cadl -o my_flow.c
   polygeist-opt my_flow.c --raise-scf-to-affine -o my_flow.mlir
   ```

## 6. Validating Changes

We added unit tests that exercise the most important lowering features. Run
them before sending patches:

```bash
pytest tests/test_transpile_to_c.py
```

Consider adding a new fixture if you touch handling for loops, register writes,
or type inference—the tests are intentionally lightweight so that extending
them is easy.

## 7. Troubleshooting

- **Parser failures:** resolve syntax errors in the CADL source first; the
  transpiler only operates on successfully parsed flows.
- **Unexpected `while (1)` loops:** the loop pattern detector requires a single
  affine induction variable with a constant step. Introduce an explicit `i + 1`
  or `i - 1` update in the binding to unlock the `for` lowering.
- **Missing return values:** ensure CADL writes to `_irf[rd]` remain in the
  flow body. Explicit CADL `return` statements take precedence over inferred
  register writes.

The transpiler intentionally focuses on high-level semantics. If you need the
hardware-aware translation (with `_irf` and bursts intact), use the older flow
or the MLIR passes under `lib/APS/Transforms/` instead.
