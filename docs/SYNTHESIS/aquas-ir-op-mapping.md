# Aquas-IR Operation Mapping Plan

## Pre-CMT2 storage planning

`aps-optimize-storage` runs on scheduled TOR/APS after temporal memory splitting
and canonicalization, immediately before `aps-to-cmt2`. No post-CMT2 register
optimization or RTL rewriting is used.

- Later consumers rematerialize unsigned widening casts in their own time slot,
  so the intervening registers carry the narrower SSA input instead of padded
  words. The pass does not cross control-flow boundaries or move memory effects.
  A now-unused original widening remains as a timing anchor: removing the only
  operation in a slot could otherwise remove a token stage and advance later
  transactions. Its dead result receives no storage during CMT2 generation.
- Source-level scalar-access analysis allows CMT2 generation to omit the
  HellaCache adapter when there are no scalar transactions. External command
  and response ports remain present; commands are idle and unsolicited
  responses can be discarded. Without a plan, direct `aps-to-cmt2` retains its
  original adapter behavior; stale no-access metadata is checked against actual
  accesses before omitting the adapter.
- Native RoCC commands store only `rs1data`, `rs2data`, and `rd` (69 bits) when
  function arguments are used solely by the native `read_irf`/`write_irf` ABI.
  Routing remains based on the original incoming opcode/funct7. Internal
  command fields not consumed by that ABI are reconstructed rather than stored.
  Unknown argument uses retain the original 96-bit payload.
- When every `write_irf` returns a literal 32-bit zero, the response FIFO stores
  only `rd` (5 bits), reconstructing zero response data at its output. Dynamic,
  nonzero, and mixed responses retain the original 37-bit payload.
- Compiler-generated control queues store occupancy only, not unread token
  payloads. One-entry queues retain simultaneous full dequeue/enqueue support;
  two-entry queues retain their no-enqueue-at-full contract. Ordinary i1 SSA
  data queues are not changed. Without the source storage plan, direct lowering
  continues to use the original payload FIFOs.
- For N-bit modular integer squares, an input used exclusively by later squares
  needs only N-1 stored bits: changing its top bit cannot change the N-bit square.
  Square results also omit bit 1, which is always zero, and restore it at later
  consumers. Packing/unpacking stays in existing schedule slots in the same
  block; mixed input uses, same-cycle dependencies, and cross-region lifetimes
  are excluded. Constants follow TOR's virtual-constant placement convention.

Do not run source CSE/canonicalization between storage planning and CMT2
generation: it could merge rematerialized casts or erase timing anchors.
Diagnostics use `--debug-only=aps-optimize-storage` in debug-enabled builds.

For `tutorial/cadl/v3ddist_vv.cadl`, a DC X-2025.06-SP4 `compile_ultra` audit
using the bundled `lsi_10k.db` sample library and a loose 1,000 ns clock measured:

| Register bits/cells reported by DC | Before planning | Initial planning | Token/square packing |
| --- | ---: | ---: | ---: |
| Elaborated | 5,435 | 4,994 | 4,945 |
| After `compile_ultra` | 4,885 | 4,879 | 4,879 |
| Removed or merged by DC | 550 | 115 | 66 |

This avoids 490 emitted register bits, eliminating most compiler-created
redundancy before CMT2. It is not a claim of a comparable silicon-area reduction:
DC already removed those bits. The second round avoids another 49 bits: 25
control-token payload bits and 24 square operand/result bits. Mapped register
count is unchanged in this round; sample-library cell area changed from 85,979
to 85,928, a small mapping difference rather than a comparable area saving.
Remaining 66 removals
include 56 duplicated memory read-address bits and 10 address/loop-control bits.
The memory duplication needs latency/hold-aware resource lowering, not a blind
replacement with asynchronous reads.

The permanent regression compares external RTL outputs against the pipeline
without storage planning for 200,000 cycles with randomized commands, scratchpad
traffic, backpressure, and synchronous resets. The measured run exercised
12,289 DMA transfers and 250 responses. Whole-design formal equivalence was
attempted but did not complete on a large arithmetic cone; it is not claimed.
All ready/enable signals are checked unconditionally; scalar-command payloads
are compared only when enabled, since idle bus tags are not observable.
Both emitted token queue depths are separately proved against occupancy and
ready/full contracts with Yosys temporal induction after reset. Square codecs
are exhaustively checked for 3–8-bit inputs and tested on 32-bit edge patterns;
their identities hold in N-bit modular arithmetic. Transform regressions check
exclusive uses, execution-scope boundaries, and idempotence.

`tests/test_mlir/test_storage_cosim.py` additionally executes fresh optimized
RTL through Spike + Verilator with self-checking RV32 custom instructions:
`hello` checks 64 dynamic responses, and a DMA-output variant of `vgemv3d`
checks 256 matrix/vector outputs over 64 invocations, including high-bit and
overflow patterns. The latter retains the tutorial's scalar loads and loop.
The tests corrected harness edge sampling, ordinary HellaCache write-mask
interpretation, and RV32 register sign extension; no optimizer change was needed.

The original scalar-output `vgemv3d` exposed a pre-existing completion bug:
`StoreWait` was a no-op, allowing RoCC completion while the final scalar store
remained queued under memory backpressure. This was reproduced with storage
planning disabled. It is now fixed: both reads and stores allocate the adapter's
existing completion slots, and `StoreWait` consumes an ordered acknowledgement
through `resp_to_user`. Call readiness stalls later schedule slots, including
RoCC completion, until the store really completes. No artificial drain cycles
are added to the cosimulation runner. The original GEMV regression checks all
256 outputs over 64 invocations and no longer has an expected-failure marker.

Bus tags identify the two occupied slots independently of address, while the
original user tags are preserved in responses. The isolated adapter regression
mixes loads/stores to the same address and user tag, reorders acknowledgements,
and applies bus/consumer backpressure, checking ordered retirement and payloads.

## Scratchpad buffer elision

The end-to-end driver now attempts elision by default **before** burst
partitioning, scratchpad memory-map allocation, unrolling, and interface binding.
SCF-to-affine preparation preserves the original loop annotations for the
eligibility check. Elision runs automatically; `no_elide = true` on a scratchpad
retains that individual staging buffer.
The functional rewrite is implemented in `lib/APS/ScratchpadElision.cpp`; the
driver owns clone-based trial scheduling and the commit/rollback decision.
It is intentionally not exposed as an unconditional rewrite pass.

The initial supported case is a rank-one static i32 buffer with exactly one
dominating copy-in and one unconditional, unit-stride affine read site in a
top-level, constant-bounded pipelined loop. Bounds must prove that every read is
inside the staged range. The native scalar CPU interface must be visible, and
the buffer must explicitly declare `cache_hint = "warm"` (hierarchy level zero).
This is a cache-residency contract, not an inferred claim about runtime cache
state. Contiguous single-stream traversal avoids stride/reuse-induced cache
thrashing without assuming an undocumented cache capacity or associativity.

The screen retains unrolled buffers, local-only temporaries, writeback buffers,
multiple read sites, cold/unknown cache residency, nested or conditional reads,
unknown trip counts, aliases/views/escapes, cross-function uses, volatile/no-elide
buffers, and unresolved memory mutations. Another direct CPU stream in the
same loop also prevents elision. Main-memory writes between staging and the
end of the consuming loop are rejected even if their addresses look different:
unknown aliasing would invalidate the original snapshot semantics. Writes
strictly before staging or after loop completion are allowed. Pipelined
iter_args/results remain excluded because the native CMT2 loop lowering does
not support them. Unsupported cases retain the original IR.

For each eligible buffer, a private functional clone removes staging and
replaces its read with `aps.load(base + (index - copy_start) * 4)`. The selected
interface plan, relative-window memory schedule, and actual TOR scheduling are
then recomputed on that clone. The original functional and scheduled modules
remain untouched unless the trial is accepted. The scheduler comparison uses
the entire static invocation estimate: straight-line scheduled spans plus each
loop's first-iteration span and `(trip_count - 1) * achieved_II`. TOR has inclusive
upper bounds, unlike SCF. Merely comparing maximum reference end time would
miss repeated-iteration costs and is not used. Every loop II must also remain
non-increasing, and the new scalar stream must fit the interface's finite
in-flight slots at that II. A failed, unsupported, slower, or capacity-infeasible
trial retains the baseline. These are modeled latency checks, not a guarantee
against arbitrary runtime cache misses or external interference.

Candidates are considered deterministically by buffer name, with an accepted
scheduled baseline reused for the next trial. Legality is rechecked after each
commit. This is a conservative greedy search, not an exponential joint subset
search. Removed declarations never enter the scratchpad allocator, so their
SRAM and staging transactions are genuinely absent from the emitted design.

RTL validation also exposed an existing dynamic bank-store hazard: writing
every bank with `select(selected, new_value, old_value)` can replay stale reads
when pipeline stages stall. Dynamic stores now use the existing
`tor.guarded_store` operation (extended to accept builtin memrefs before TOR
scheduling), and CMT2 emits a conditional bank write. Inactive banks are not
read or written. The operation already participates in TOR memory-dependence
and resource analyses; the predicate is a normal SSA operand transported by
the pipeline's live-value FIFOs. Unconditional stores retain their existing
lowering. No scratchpad read/modify/write pattern is used for bank selection.
The native CPU adapter also gates command commitment on the next response
slot being free and clears that slot's received flag when allocating it. Its
two-entry command FIFO is not permission to overwrite two occupied response
slots. This enforces the modeled two-slot limit and preserves response order
under backpressure and out-of-order HellaCache responses.
The shared CMT2 `FIFO2_I` generator also preserves its oldest head when a second
item is enqueued without a dequeue; the new item is appended in the tail
register. The previous transition put the newest item at the head, reversing
stalled command and live-value queues. A standalone Verilator regression tests
full-queue stalls, drains, and simultaneous enqueue/dequeue against FIFO order.

```bash
pixi run syn examples/scratchpad_elision.mlir result.sv examples/interface.json result.chipyard.yaml '--debug-only=scratchpad-elision --dump-scheduled-tor scheduled.mlir'
```

LLVM debug logging (`--debug-only=scratchpad-elision`, in debug-enabled builds)
prints eligibility/rejection reasons, acceptance, estimated baseline/trial cycles,
and achieved IIs. `--dump-scheduled-tor` captures the
selected schedule before temporal splitting; `--export scheduled-tor` can stop
at that boundary for analysis. Later backend support is still required for the
unmodified input when a candidate is rejected.

## Interface configuration and Chipyard packaging

`aps-e2e --interface-config examples/interface.json --export sv -i input.mlir -o output.sv`
reads the versioned JSON model in `examples/interface.json`. It installs
`aps.memitfc` attributes before lowering and, after successful RTL generation,
writes `output.chipyard.yaml`. Use `--chipyard-config path.yaml` to choose another
destination. The Pixi synthesis tasks default to `examples/interface.json` and write
`thirdparty/chipyard/aps_config.yaml` after successful synthesis, replacing its
contents. Optional interface and Chipyard configuration filenames override these
defaults. Additional flags follow those two filenames:

```bash
pixi run syn input.mlir output.sv
pixi run syn input.mlir output.sv custom-interface.json
pixi run syn input.mlir output.sv examples/interface.json custom-chipyard.yaml
pixi run syn-debug input.mlir output.sv synthesis.log custom-interface.json
pixi run syn-debug input.mlir output.sv synthesis.log examples/interface.json custom-chipyard.yaml
```

The file contains `version: 1`, an `interfaces` array using the IR model field
names, a `chipyard` object (`backend`, `arch`, `core_count`, `top_module`, `name`),
and an optional `csrs` array. The supplied model uses 4-byte CPU transactions,
8-byte DMA beats, and a 128-byte DMA maximum. Both interfaces have `inflight=2`:
the CPU translator has two response slots, and the DMA wrapper has two channels.
CPU inflight may be conservatively reduced to one; neither adapter currently
supports a model exceeding its two-slot/channel capacity. Only native
`cpuitfc`/`busitfc` adapters and the single-core RV32 RoCC platform are supported
by this packaging flow. Without the configuration flag, existing IR/default
behavior remains available and no Chipyard YAML is emitted.

Chipyard `maxBurstBytes` is derived from DMA `width * max_beats`; `nXacts` is
derived from **DMA** `inflight`, not CPU `inflight`. The generated `vsrc` contains
the absolute RTL path. Performance parameters such as latency and cache-line
size guide the compiler; Chipyard's current schema has no corresponding fields
and they do not reconfigure the cache itself. Unknown JSON keys, unsupported
hardware models, and conflicts with existing IR declarations are rejected.
Direct `aps-e2e` invocation defaults to a standalone YAML artifact beside the
RTL; the Pixi tasks explicitly select the Chipyard-root configuration instead.

CSR entries are collected in input-IR order from `memref.global` declarations
with `csr_address`; names are the actual RTL port prefixes. Each must contain
one i32 element with a unique 12-bit address. Defaults are mask `0xffffffff`
and init `0`, or the IR's `csr_mask`/`csr_init` attributes if present. Optional
JSON overrides are keyed by the IR name:

```json
"csrs": [{"name": "gain_cfg", "id": "0x801", "mask": "0xff", "init": 0}]
```

`id` is optional and, if supplied, must match the IR address. Overrides cannot
add undeclared CSRs. Mask/init accept non-negative integers or integer strings
(including hexadecimal) and must fit RV32. Reset/mask values are passed to
Chipyard's custom CSR configuration, not used to invent extra accelerator ports.
The generated YAML emits CSR `id`, `mask`, and `init` as quoted hexadecimal
strings (for example `"0x801"`, `"0xffffffff"`, and `"0x0"`), supported by
Chipyard's integer parser.

This document defines the first-step APS IR rename and layering plan for
aligning the current APS dialect with the three-level Aquas-IR structure
described in `Aquas_ICCAD26_-2.pdf`, pages 3-4.

The initial rename is complete, and the architectural binding pass now consumes
the `W/M/I/L/E/C` interface tuple.  The IR shape and implemented flow are:

1. Functional level: express the memory/register operation intent.
2. Architectural level: bind that intent to a concrete interface.
3. Temporal level: express issue/wait timing and transaction ordering.

## 1. Design Scope

The current implementation:

- Keep the existing lowering behavior where possible.
- Preserve old operations during a compatibility period.
- Uses explicit interface symbols and interface references.
- Selects an interface using the analytical latency recurrence plus the cache
  line/hierarchy penalty.
- Greedily decomposes constant-size copies into naturally sized transactions.
- Records the selected cost, latency, size, alignment, group, segment, and
  predecessor segment on the architectural operations.
- Keep CMT2 lowering compatible with current request/collect hardware
  generation.

## 2. Target IR Layers

### 2.1 Functional Level

Functional IR describes what the ISAX wants to do without committing to a
physical memory interface or transaction timing.

Representative target operations:

| Operation | Meaning |
|----------|---------|
| `aps.copy` | Bulk data movement between global/CPU memory and scratchpad memory. |
| `aps.load` | Scalar CPU-memory/global-memory read. |
| `aps.store` | Scalar CPU-memory/global-memory write. |
| `aps.read_smem` | Read from ISAX scratchpad/local memory. |
| `aps.write_smem` | Write to ISAX scratchpad/local memory. |
| `aps.globalload` | APS-specific scalar global state read. Keep unchanged for now. |
| `aps.globalstore` | APS-specific scalar global state write. Keep unchanged for now. |
| `aps.read_irf` | Read CPU integer register file input. |
| `aps.write_irf` | Write CPU integer register file output. |
| `aps.read_csr` | Read an APS custom CSR. |
| `aps.write_csr` | Write an APS custom CSR. |

This layer is the right place for scratchpad buffer elision and other
source-level memory-intent rewrites.

### 2.2 Architectural Level

Architectural IR binds memory operations to visible interfaces and memory
layout. It still does not force explicit issue/wait timing.

Representative target operations:

| Operation | Meaning |
|----------|---------|
| `aps.memitfc @name { ... }` | Module-level memory-interface symbol. |
| `aps.copy_by @itfc, ...` | Interface-bound bulk copy. |
| `aps.load_by @itfc, ...` | Interface-bound memory load. |
| `aps.store_by @itfc, ...` | Interface-bound memory store. |
| `aps.memorymap` | Memory layout and scratchpad allocation metadata. |
| `aps.mem_entry` | Memory allocation entry, optionally associated with an interface. |

`aps.memitfc` is a module-level symbol consumed by architectural lowering. Its
six positive integer fields are validated when the pass runs. Width must be a
power of two, cache-line size must be a multiple of width, and maximum
transaction capacity must fit the byte-count model. `min_beats` defaults to one
and must be a positive power of two no greater than `max_beats`. Minimum length
is `W * min_beats`, not the cache-line size. `cache_hint` is
optional.
`min_write_beats` can raise the write minimum independently; it defaults to
`min_beats` on generic interfaces. The native DMA has a one-beat read minimum
and a two-beat write minimum because its current write prefetch state machine
does not handle a single-beat transaction. Native defaults therefore use
`min_beats = 1` and `min_write_beats = 2` for `busitfc` (8-byte reads, 16-byte
writes). These are adapter limits, not cache-line requirements.

Minimal form:

```mlir
aps.memitfc @cpuitfc
aps.memitfc @busitfc
```

Optional documentation attributes:

```mlir
aps.memitfc @cpuitfc {
  kind = "cpu",
  width = 4 : i32,
  min_beats = 1 : i32,
  max_beats = 1 : i32,
  inflight = 2 : i32,
  read_latency = 2 : i32,
  write_completion = 2 : i32,
  cache_line = 64 : i32,
  cache_hint = "warm"
}
```

For source compatibility, name-only `@cpuitfc` and `@busitfc` declarations are
filled with the defaults above. When no interfaces are declared, both defaults
are materialized. If any custom interface is declared, no unrelated defaults
are added.

### 2.3 Temporal Level

Temporal IR exposes asynchronous transaction issue and completion. This level
is the paper-facing rename of the current `req/collect` split.

Representative target operations:

| Operation | Meaning |
|----------|---------|
| `aps.copy_issue @itfc, ...` | Issue a bulk transfer transaction and return a token. |
| `aps.copy_wait %token` | Wait for a bulk transfer token to complete. |
| `aps.load_issue @itfc, ...` | Issue a memory load transaction and return a token. |
| `aps.load_wait %token` | Wait for a load token and return data. |
| `aps.store_issue @itfc, ...` | Issue a memory store transaction and return a token. |
| `aps.store_wait %token` | Wait for a store token to complete. |
| `aps.read_smem_issue` | Issue a scratchpad/local memory read request. |
| `aps.read_smem_wait` | Wait for a scratchpad/local memory read result. |

Transaction ordering should be represented with an `after` dependency when an
operation must be sequenced after an earlier token:

```mlir
%t0 = aps.copy_issue @busitfc, %s96, %src[96], 8
%t1 = aps.copy_issue @busitfc, %s0, %src[0], 64 {after = %t0}
%t2 = aps.copy_issue @busitfc, %s64, %src[64], 32 {after = %t1}
aps.copy_wait %t2
```

## 3. Current-to-Target Operation Mapping

### 3.1 Functional Operations

| Current operation | Target operation | Layer | Migration plan |
|------------------|------------------|-------|----------------|
| `aps.memdeclare` | `aps.memdeclare` | Functional | Keep unchanged. The current op already means local scratchpad declaration and does not need to participate in the rename. |
| `aps.read_smem` | `aps.read_smem` | Functional | Direct rename. This matches the paper's scratchpad read naming. |
| `aps.write_smem` | `aps.write_smem` | Functional | Direct rename. |
| `aps.memburstload` | `aps.copy` | Functional | Rename bulk load intent to an interface-agnostic copy. Direction is mandatory and represented by the load-like pretty form: `aps.copy %cpu, (%smem)[%start], %len`. |
| `aps.memburststore` | `aps.copy` | Functional | Same target op as burst load, with mandatory `out` direction represented by the store-like pretty form: `aps.copy (%smem)[%start], %cpu, %len`. |
| none or future scalar CPU-memory read frontend form | `aps.load` | Functional | Use only for scalar CPU-memory/global-memory reads. Do not map current `aps.globalload` to this op. |
| none or future scalar CPU-memory write frontend form | `aps.store` | Functional | Use only for scalar CPU-memory/global-memory writes. Do not map current `aps.globalstore` to this op. |
| `aps.globalload` | `aps.globalload` | Functional | Keep unchanged. This is APS-specific scalar global state access and is not the same semantic category as paper `fetch`. |
| `aps.globalstore` | `aps.globalstore` | Functional | Keep unchanged. This is APS-specific scalar global state access and should not be renamed to the scalar transfer write op. |
| `aps.readrf` | `aps.read_irf` | Functional | Direct rename to match the paper table. |
| `aps.writerf` | `aps.write_irf` | Functional | Direct rename. |
| `aps.readcsr` | `aps.read_csr` | Functional or Architectural | Direct style rename. CSR does not need to go through `memitfc` unless a later architecture model requires it. |
| `aps.writecsr` | `aps.write_csr` | Functional or Architectural | Direct style rename. |

Recommended first choice:

- Rename old scratchpad load/store ops to `aps.read_smem` and
  `aps.write_smem`.
- Rename `aps.memburstload` and `aps.memburststore` to one `aps.copy`.
- Keep `aps.globalload` and `aps.globalstore` unchanged.
- Add `aps.load` and `aps.store` only when scalar CPU-memory/global-memory
  traffic is needed.
- Rename `aps.readrf` and `aps.writerf` to `aps.read_irf` and `aps.write_irf`.

### 3.2 Architectural Operations

| Current operation | Target operation | Layer | Migration plan |
|------------------|------------------|-------|----------------|
| none | `aps.memitfc @name` | Architectural | Add a dummy module-level symbol op. It exists so `copy_by/load_by/store_by` can reference an interface, not to implement the interface model yet. |
| `aps.copy` after binding | `aps.copy_by @itfc, ...` | Architectural | Lower functional copy to interface-bound copy. |
| `aps.load` after binding | `aps.load_by @itfc, ...` | Architectural | Lower scalar CPU-memory/global-memory reads to interface-bound loads. |
| `aps.store` after binding | `aps.store_by @itfc, ...` | Architectural | Lower scalar CPU-memory/global-memory writes to interface-bound stores. |
| `aps.globalload` | `aps.globalload` | Functional or APS-specific state | Keep unchanged unless a later pass explicitly chooses to model it as interface traffic. |
| `aps.globalstore` | `aps.globalstore` | Functional or APS-specific state | Keep unchanged unless a later pass explicitly chooses to model it as interface traffic. |
| `aps.memorymap` | `aps.memorymap` | Architectural | Keep the name. It already describes memory layout metadata. |
| `aps.mem_entry` | `aps.mem_entry` with optional `itfc = @name` | Architectural | Keep the name and extend attributes only when an allocation is interface-bound. |
| `aps.mem_finish` | `aps.mem_finish` | Architectural | Keep the terminator. |

Recommended initial `memitfc` policy:

- Add `aps.memitfc` as a dummy symbol op.
- Require only a symbol name.
- Allow partial attributes as documentation only.
- Do not require all memory ops to be bound immediately.
- Let architectural lowering insert explicit interface references only for
  operations that are selected or manually annotated.

### 3.3 Temporal Operations

| Current operation | Target operation | Layer | Migration plan |
|------------------|------------------|-------|----------------|
| `aps.itfc.load_req` | `aps.load_issue` | Temporal | Direct rename. Preserve token result semantics. Add explicit `@itfc` reference. |
| `aps.itfc.load_collect` | `aps.load_wait` | Temporal | Direct rename. Preserve token operand and data result semantics. |
| `aps.itfc.store_req` | `aps.store_issue` | Temporal | Direct rename. Preserve token result semantics. |
| `aps.itfc.store_collect` | `aps.store_wait` | Temporal | Direct rename. Preserve token operand semantics. |
| `aps.itfc.burst_load_req` | `aps.copy_issue` | Temporal | Rename to generic copy issue. Direction is mandatory and represented by the load-like pretty form. |
| `aps.itfc.burst_load_collect` | `aps.copy_wait` | Temporal | Rename to generic copy wait. |
| `aps.itfc.burst_store_req` | `aps.copy_issue` | Temporal | Same target op as burst load issue. Direction is mandatory and represented by the store-like pretty form. |
| `aps.itfc.burst_store_collect` | `aps.copy_wait` | Temporal | Same target op as burst load wait. |
| `aps.read_smem_issue` | `aps.read_smem_issue` | Temporal local-memory | Rename to the paper-style scratchpad timing op. This is scratchpad port timing, not a core-memory-interface transfer. |
| `aps.read_smem_wait` | `aps.read_smem_wait` | Temporal local-memory | Rename to the paper-style scratchpad timing op. |

Recommended first choice:

- Use `issue/wait` for paper alignment.
- Keep `smem` temporal ops separate from `itfc` temporal ops.
- Use a single `aps.copy_issue/copy_wait` temporal pair for both bulk copy
  directions.

## 4. Example Layering

### 4.1 Functional IR

```mlir
aps.copy %dram_addr, (%src)[%c0], %c104
  : i64, (memref<104xi8>), index, i32

aps.copy (%dst)[%c0], %dram_addr, %c104
  : (memref<104xi8>), index, i64, i32

%bias = aps.load %bias_addr : i64 -> i32
%acc = aps.read_smem %buf[%i] : memref<48xi32>, index -> i32
aps.write_smem %acc, %out[%i] : i32, memref<48xi32>, index
%state = aps.globalload @count : i32
```

### 4.2 Architectural IR

```mlir
aps.memitfc @cpuitfc
aps.memitfc @busitfc

aps.copy_by @busitfc, %dram_addr, (%src)[%c0], %c64
  : i64, (memref<104xi8>), index, i32

aps.copy_by @busitfc, (%dst)[%c0], %dram_addr, %c64
  : (memref<104xi8>), index, i64, i32

%bias = aps.load_by @cpuitfc, %bias_addr : i64 -> i32
```

### 4.3 Temporal IR

```mlir
%t0 = aps.copy_issue @busitfc, %dram_addr_0, (%src)[%c0], %c64
  : i64, (memref<104xi8>), index, i32 -> !aps.mem_req
%t1 = aps.copy_issue @busitfc, %dram_addr_64, (%src)[%c64], %c32
  {after = %t0} : i64, (memref<104xi8>), index, i32 -> !aps.mem_req
aps.copy_wait %t1 : !aps.mem_req

%r0 = aps.load_issue @cpuitfc, %bias_addr : i64 -> !aps.mem_req
%bias = aps.load_wait %r0 : !aps.mem_req -> i32
```

These examples are illustrative. The exact assembly format can be adjusted to
match existing APS parser/printer conventions.

## 5. Pass-Level Migration Plan

### 5.1 Compatibility Phase

Add new operation definitions while keeping old operation names valid.

Required work:

- Add target operation definitions in `include/APS/APSOps.td`.
- Keep old operation definitions or add parser aliases.
- Update generated C++ op includes through the existing build flow.
- Add tests that parse both old and new names during the transition.

The compatibility phase should avoid semantic rewrites. It should only prove
that the new names can round-trip and that old input still lowers.

### 5.2 Frontend Emission Phase

Move CADL-to-MLIR emission to the functional names.

Required work:

- Emit `aps.read_irf` and `aps.write_irf` for `_irf`.
- Emit `aps.read_smem` and `aps.write_smem` for local scratchpad access.
- Emit `aps.copy` for `_burst_read` and `_burst_write`.
- Emit `aps.load` and `aps.store` only when CADL needs scalar
  CPU-memory/global-memory traffic.
- Keep `aps.globalload` and `aps.globalstore` for APS-specific scalar global
  state access.
- Keep conversion tests that check operation counts updated to the new names.

### 5.3 Architectural Binding Phase

Introduce a dummy `aps.memitfc` symbol op and interface-bound operations.

Required work:

- Add `aps.memitfc` symbol op with name-only semantics.
- Add `aps.copy_by`, `aps.load_by`, and `aps.store_by`.
- Add an explicit binding pass from functional memory ops to architectural ops.
- Initially support manual/default binding before adding optimization.
- Extend `aps.mem_entry` with an optional `itfc` symbol reference only where
  useful.

This phase should not require automatic interface selection. A simple default
interface policy is enough to make the IR layer visible.

### 5.4 Temporal Lowering Phase

Rename current request/collect operations into issue/wait operations.

Required work:

- Add `aps.load_issue`, `aps.load_wait`, `aps.store_issue`, and
  `aps.store_wait`.
- Add `aps.read_smem_issue` and `aps.read_smem_wait`.
- Rename existing burst temporal ops to `aps.copy_issue/copy_wait`.
- Move existing APSToCMT2 generator handling from old `itfc.*_req/collect` ops
  to the new issue/wait names.
- Preserve the current token-based lowering behavior.
- Add `after` verification only after all users agree on the assembly shape.

### 5.5 Scheduling Resource Policy

`aps.copy` direction still matters to scheduling semantics:

- copy-in form `aps.copy %cpu, (%smem)[%start], %len` maps to `TL_READ_OP`
  because it writes scratchpad/global partitions from CPU/global memory.
- copy-out form `aps.copy (%smem)[%start], %cpu, %len` maps to `TL_WRITE_OP`
  because it reads scratchpad/global partitions into CPU/global memory.

These two operation types are for dependency classification only. They must use
the same `"tl"` scheduler resource because the read and write paths share the
same TileLink/DMA channel pool. The `tl_channel` assignment therefore colors
all TileLink copies together, regardless of direction.

## 6. Naming Decisions to Finalize Before Coding

These are the only naming choices that should be settled before the first code
patch:

| Topic | Preferred choice | Alternative |
|-------|------------------|-------------|
| Scratchpad declaration | keep `aps.memdeclare` | none |
| Functional bulk copy direction | one `aps.copy` with mandatory direction encoded by the two pretty-print forms | `aps.copy_in` / `aps.copy_out` |
| Architectural binding suffix | `aps.copy_by/load_by/store_by` | `aps.copy_via/load_via/store_via` |
| Scratchpad temporal names | `aps.read_smem_issue/wait` | keep `aps.read_smem_issue/collect` |

The lowest-risk path is to rename only high-value externally visible ops first
and keep `aps.memdeclare` and internal scratchpad timing names stable until
tests are migrated.

## 7. Implementation Checklist

- [x] Add new functional operation definitions.
- [x] Add new temporal operation definitions.
- [x] Add dummy `aps.memitfc` symbol op.
- [x] Add architectural `aps.copy_by/load_by/store_by`.
- [ ] Add compatibility parsing or keep old ops temporarily.
- [x] Update CADL MLIR emission to functional names.
- [x] Add a functional-to-architectural binding pass.
- [x] Validate and consume the `W/M/I/L/E/C` interface model.
- [x] Solve region-wide interface assignments with the paper latency/cache objective.
- [x] Canonicalize constant-size copies into legal transfer segments.
- [x] Preserve interface identity on scalar temporal issue operations.
- [x] Record explicit segment ordering metadata for temporal lowering.
- [x] Add memoized transaction ordering with normalized timing windows.
- [x] Consume issue-order dependencies in TOR scheduling before the existing split pass.
- [x] Update APSToCMT2 scalar interface generators to accept new temporal names.
- [x] Migrate focused tests from old op names to new op names.
- [ ] Mark old op names deprecated after downstream tests pass.

## 8. Non-Goals for the First Patch Series

- Do not rewrite CMT2 rule construction unless required by renamed op matching.
- Do not remove old operation names before frontend and lowering tests are
  migrated.
- Do not couple scratchpad local timing ops to external memory-interface ops.

## 9. Interface Assignment Solver

`aps-functional-to-arch` uses dynamic programming over active-interface sets
to solve reads and writes separately in each immediate MLIR region. Every
transition assigns one request to one legal interface (the paper's `X(q,k)`),
charging startup only on that interface's first use. Illegal decompositions
have no transition. The objective implements
the approximation in `4_modeling_and_synthesis.tex`: read activation `L-1`
plus per-segment `max(L/I, m/W)`, or write activation `-1` plus per-segment
`m/W+E`, together with each request's cache-line synchronization penalty.
The cache penalty is exactly `ceil(m/C) * C/W`; hints guide hierarchy ordering
rather than adding an extra, undocumented penalty multiplier.

Splitting is planned separately for every candidate interface **before**
assignment, so legality and the cost reflect the actual transactions. The
chosen plan is emitted during functional-to-architectural conversion. A copy
stays intact only if it is one legal, naturally aligned power-of-two whole-beat
transaction within the interface's minimum and maximum. Larger copies,
non-power-of-two byte counts, and known alignment restrictions require splitting.
Each emitted architectural operation is exactly one transaction. Unknown base
alignment remains an explicit `required_alignment` contract; known constant
addresses constrain each planned segment.

No interface may produce a segment below `W * min_beats` (or the higher
`W * min_write_beats` for writes), and the request must
be exactly representable: the compiler never rounds up or silently discards a
tail. On the native 8-byte DMA, reads may use `104 -> 64 + 32 + 8`,
`200 -> 64 + 64 + 64 + 8`, and `24 -> 16 + 8`. Writes require at least 16 bytes
per transaction: `112 -> 64 + 32 + 16`; 104-byte writes are rejected. `108` is
rejected in both directions because its final 4 bytes are unsupported.
A generic 4-byte interface can represent 108 bytes, but requires a corresponding
platform adapter. Native `cpuitfc` supports 4-byte scalar operations only and
`busitfc` supports DMA copies only; scalarized fully partitioned copies use the
CPU scalar path. An unimplemented CPU bulk-copy fallback is not considered.

For `Q` requests and `K` visible interfaces, the assignment recurrence takes
`O(Q K 2^K)` time. It retains only reachable masks and backpointers for
reconstruction. This avoids general-purpose MIP search; exponential growth is
in interface count, not request count. Masks support up to 63 interfaces,
although practical use assumes only a few interfaces.

Fully partitioned scratchpads use scalar segments in the objective and preserve
the selected interface when scalarized. Constant-size assignments are annotated
with `interface_solver = "active-set-dp"` and the region/direction `solver_objective`;
`estimated_latency` still reports the recurrence estimate for each request.
Unknown transfer lengths are rejected with a diagnostic requiring normalization
before interface binding, so this conversion never emits unsized architectural
copies. These annotations describe
interface assignment, not an optimized temporal issue order.

## 10. Memoized Transaction Ordering

`aps-schedule-memory` runs after control-flow lowering, before `schedule-tor`.
It keeps architectural operations intact. Within each block it collects static
transfers in windows bounded by side effects, control flow, uses of memory
results, and direction changes. Oversized functional copies have already been
disassembled into single-transaction architectural operations by
`aps-functional-to-arch`; there is no separate disassembly pass. Decomposed
transfers remain contiguous groups. Aliasing scratchpad destinations, uncertain write addresses,
and the existing scalar response protocol retain necessary source dependencies.

The search uses the paper's issue/completion recurrences and finite in-flight
slots, with independent interface histories. Warm reads precede cold reads;
cold writes precede warm writes among dependency-ready groups. An explicit
non-negative `cache_level` extends this to more hierarchy levels (smaller is
closer to the CPU). Data dependencies take priority over hints. Equivalent,
independent groups use remaining counts instead of distinct permutation IDs.
Within each original transfer, the search may reorder its segments rather than
fixing decreasing-size order. An active-group lock preserves contiguity, and
counts of equal segment sizes avoid enumerating identical permutations. The
search minimizes completion time; bell-shaped sequences can emerge, but no
bell-shape heuristic is imposed.

Memoization keys contain remaining group/profile counts, the active group and
its remaining segment-size counts, the latest issue and
hierarchy level, the last `I` completions per interface, and completion times
needed by dependent groups. All times are relative to the latest issue; older
times collapse to `-1`, eliminating absolute-time distinctions. This search is
exact within the window's constraints, but can still grow exponentially when
many distinct independent transfer profiles remain.

The chosen schedule adds `transaction_id`, `after` (an array of predecessor
IDs), and `after_gap` (matching minimum start-to-start delays). Dependencies
cover issue order, slot retirement, transfer serialization and source data
dependencies. `memory_issue_cycle`, `memory_completion_cycle`,
`memory_schedule_latency`, and search-state/hit counts expose the modeled
result. `segment` and `transfer_offset` retain the original address slice;
`issue_segment` records the chosen order within the transfer, and
`after_segment` references the previous logical segment in that chosen order.
Sub-beat tails are rejected; timing recurrences use whole beats only. Temporal
ordering checks that its input operations are already canonical. Native CMT2
adapters validate interface identity and transaction size; DMA lowering does
not round a transaction up to a power of two.

TOR scheduling consumes these edges without forcing every issue to wait for
the previous transfer to complete. Existing resource and data constraints may
extend the modeled schedule. The existing `lower-aps-mem-to-req-collect` pass
still performs the issue/wait split after TOR scheduling, preserving ordering
only on issues. Per-transfer waits remain for compatibility with the existing
response/channel adapters; they are not replaced by a single final wait.
