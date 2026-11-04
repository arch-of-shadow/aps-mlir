# CADL language and frontend

CADL describes instruction behavior. The authoritative syntax is
[grammar.lark](../../cadl_frontend/grammar.lark); accepted syntax still needs a
supported MLIR emitter and backend operation to reach hardware.

## Instruction shape

```cadl
#[opcode(7'b0101011)]
#[funct7(7'b0000000)]
rtype hello(rs1: u5, rs2: u5, rd: u5) {
  let a: u32 = _irf[rs1];
  let b: u32 = _irf[rs2];
  _irf[rd] = a + b;
}
```

Register selectors are five-bit values; `_irf[rs1]` obtains the operand data
supplied by the CPU. The encoding attributes identify the custom instruction.
Use the checked-in [hello](../../examples/aspdac26-tutorial/cadl/hello.cadl) as a minimal input.

```sh
pixi run tools/aps-frontend mlir examples/aspdac26-tutorial/cadl/hello.cadl -o build/hello.mlir
pixi run tools/aps-frontend encoding examples/aspdac26-tutorial/cadl/hello.cadl
pixi run tools/aps-frontend cadl2c examples/aspdac26-tutorial/cadl/hello.cadl -o build/hello.c
```

Create `build/` first. The MLIR command verifies its result before writing it.
C output supports software matching and reference workflows; it has its own
supported subset, particularly for floating point.

## Values and storage

| Construct | Meaning |
|---|---|
| `uN`, `iN` | Unsigned/signed fixed-width integer semantics |
| `bf16`, `f32` | Supported hardware floating formats |
| `[i32; 16]` | Array type; further semicolon dimensions express higher ranks |
| `let x: u32 = expr;` | Local value binding |
| `x = expr;` | Update an existing local binding or scalar global |
| `static buf: [i32; 16];` | Persistent storage emitted as a module-level global |
| `_mem[addr]` | CPU-memory access; `addr` is a byte address |
| `buf[index]` | Element access to local/global array storage |

Integer literals include `42`, `0x2a`, `8'h2a`, and explicit suffixes such as
`42_u8`. Decimal floating literals default to f32; suffixes select the semantic
format. Parser recognition of f64 does not establish FP64 hardware support.
`bitcast<T>(value)` preserves bits and requires equal widths. Numeric conversions
and bitcasts have different semantics.

Array partition metadata is attached to the declaration:

```cadl
#[partition_dim_array([0])]
#[partition_factor_array([4])]
#[partition_cyclic_array([1])]
static buf: [i32; 16];
```

For complete CPU-memory slice/copy examples use
[examples/aspdac26-tutorial/cadl/](../../examples/aspdac26-tutorial/cadl) and
[MemoryEmitter](../../cadl_frontend/to_mlir/memory.py). Synthesis binds functional
memory operations using the [interface model](../ARCHITECTURE/aps-itfc.md).

## Control flow

Statement `if` emits `scf.if`. Assignments to outer local values are merged
through results; a missing `else` retains the incoming value. Branch-local
bindings do not escape their scope, and a branch-local `let` cannot shadow a
variable being merged. Expression `if condition { a } else { b }` produces a value.

Counted loops use the CADL `with` form:

```cadl
[[unroll(4)]]
with i: u32 = (0, i + 1) do {
  buf[i] = i;
} while (i + 1 < 16);
```

The frontend recognizes suitable counted patterns as `scf.for`; other supported
patterns use `scf.while`. Directives apply to the subsequent loop. Check generated
IR when changing bounds, steps, or carried values. Pipeline requests remain
subject to scheduler and CMT2 legality checks; see [HLS](hls-passes.md).

The grammar includes constructs such as guarded statements and `spawn` that the
current statement dispatcher does not lower. Treat an explicit unsupported
statement diagnostic as a frontend boundary. Do not infer support from the grammar
alone. Floating intrinsics and restrictions are in the
[BF16/FP32 guide](hardfloat-bf16-fp32-acceptance.md).

## Implementation and tests

[converter.py](../../cadl_frontend/to_mlir/converter.py) coordinates conversion;
[state.py](../../cadl_frontend/to_mlir/state.py) owns symbols;
[expr.py](../../cadl_frontend/to_mlir/expr.py),
[stmt.py](../../cadl_frontend/to_mlir/stmt.py),
[loop.py](../../cadl_frontend/to_mlir/loop.py), and
[memory.py](../../cadl_frontend/to_mlir/memory.py) own emission.
Use [parser tests](../../tests/frontend/parser),
[conditional tests](../../tests/frontend/test_select_expr.py), and
[floating frontend tests](../../tests/hardfloat/test_frontend.py) for executable examples.
