# Aquas-IR Operation Mapping Plan

This document defines the first-step APS IR rename and layering plan for
aligning the current APS dialect with the three-level Aquas-IR structure
described in `Aquas_ICCAD26_-2.pdf`, pages 3-4.

The goal of this step is not to fully implement the interface cost model. In
particular, the `W/M/I/L/E/C` interface tuple can be represented as optional
metadata first. The immediate goal is to make the IR shape match the paper:

1. Functional level: express the memory/register operation intent.
2. Architectural level: bind that intent to a concrete interface.
3. Temporal level: express issue/wait timing and transaction ordering.

## 1. Design Scope

This refactor should be mostly a rename and small metadata extension:

- Keep the existing lowering behavior where possible.
- Preserve old operations during a compatibility period.
- Add explicit interface symbols and interface references.
- Avoid implementing interface selection, cost solving, or complete
  `W/M/I/L/E/C` scheduling semantics in the initial rename.
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

The first implementation can use a dummy interface op. It only needs to be a
module-level symbol that `copy_by/load_by/store_by` can reference. Attributes
are optional documentation fields and should not drive verification or lowering
yet.

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
  max_beats = 1 : i32,
  inflight = 1 : i32,
  read_latency = 2 : i32,
  write_latency = 2 : i32,
  cache_line = 64 : i32,
  cache_hint = "warm"
}
```

Passes should not require these attributes until the interface model is
actually implemented.

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
aps.copy %dram_addr, (%src)[%c0], %c108
  : i64, (memref<108xi8>), index, i32

aps.copy (%dst)[%c0], %dram_addr, %c108
  : (memref<108xi8>), index, i64, i32

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
  : i64, (memref<108xi8>), index, i32

aps.copy_by @busitfc, (%dst)[%c0], %dram_addr, %c64
  : (memref<108xi8>), index, i64, i32

%bias = aps.load_by @cpuitfc, %bias_addr : i64 -> i32
```

### 4.3 Temporal IR

```mlir
%t0 = aps.copy_issue @busitfc, %dram_addr_0, (%src)[%c0], %c64
  : i64, (memref<108xi8>), index, i32 -> !aps.mem_req
%t1 = aps.copy_issue @busitfc, %dram_addr_64, (%src)[%c64], %c32
  {after = %t0} : i64, (memref<108xi8>), index, i32 -> !aps.mem_req
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
- [x] Add a simple functional-to-architectural binding pass.
- [x] Update APSToCMT2 scalar interface generators to accept new temporal names.
- [x] Migrate focused tests from old op names to new op names.
- [ ] Mark old op names deprecated after downstream tests pass.

## 8. Non-Goals for the First Patch Series

- Do not implement full `W/M/I/L/E/C` optimization.
- Do not make `aps.memitfc` attributes required.
- Do not solve automatic interface selection globally.
- Do not rewrite CMT2 rule construction unless required by renamed op matching.
- Do not remove old operation names before frontend and lowering tests are
  migrated.
- Do not couple scratchpad local timing ops to external memory-interface ops.
