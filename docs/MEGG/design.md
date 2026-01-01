# Megg Design Document

This document explains Megg's architecture: how MLIR and e-graphs are unified to solve the phase-ordering problem in compiler optimization.

---

## High-Level Architecture

```
MLIR Module ──> Compiler ──> Optimized MLIR Module
                   │
                   ├─ FuncToTerms: MLIR → e-graph terms
                   │
                   ├─ Phase 1: Internal Rewrites (expand e-graph)
                   │   └─ Algebraic laws + constant folding
                   │
                   ├─ Phase 2: External Rewrites (expand e-graph) [TODO]
                   │   └─ MLIR passes
                   │
                   ├─ Phase 3: Custom Instruction Matching (final pass)
                   │   └─ Pattern matching → custom_instr nodes
                   │
                   └─ Extraction: e-graph → MLIR
                       │
                       ├─ MeggEGraph: e-graph wrapper
                       ├─ Extractor: greedy extraction with cost functions
                       └─ ExprTreeToMLIR: expression trees → MLIR ops
```

### Core Pipeline Stages

1. **CLI Layer** ([megg-opt.py](../megg-opt.py), [cli.py](../python/megg/cli.py))
   - Parse command-line arguments
   - Load MLIR module from file
   - Invoke compiler and backend

2. **Compiler Orchestration** ([compiler.py](../python/megg/compiler.py))
   - Convert MLIR functions to e-graph terms
   - Execute internal and external rewrites
   - Extract optimized MLIR module

3. **E-graph Translation Layer** ([python/megg/egraph/](../python/megg/egraph/))
   - **func_to_terms.py**: MLIR SSA → egglog terms
   - **megg_egraph.py**: E-graph wrapper with Python dataclasses
   - **extract.py**: Greedy extraction with customizable cost functions
   - **terms_to_func.py**: Expression trees → MLIR operations

4. **Rewrite Layer** ([python/megg/rewrites/](../python/megg/rewrites/))
   - **internal_rewrites.py**: Egglog algebraic rules
   - **external_rewrites.py**: MLIR pass pipeline integration [TODO]
   - **match_rewrites.py**: Custom instruction pattern matching

5. **Backend Layer** ([backend/llvm_backend.py](../python/megg/backend/llvm_backend.py))
   - Lower MLIR to LLVM IR
   - Compile to object files or executables

---

## Core Components

### 1. Compiler ([compiler.py](../python/megg/compiler.py))

The `Compiler` class orchestrates the entire optimization pipeline:

**Initialization:**
```python
compiler = Compiler(
    module,                    # Input MLIR module
    target_functions=None,     # Optional: specific functions to optimize
    cost_function=AstSize(),   # Cost function for extraction
    match_ruleset=None         # Optional: custom instruction matching rules
)
```

- Converts each `func.func` to an e-graph using `FuncToTerms`
- Maintains bidirectional mappings: SSA values ↔ e-graph terms
- Stores one `egglog.EGraph` per function in `self.egraphs`
- Skips functions with unsupported operations (e.g., function calls)

**Scheduling (3-Phase Pipeline):**
```python
result = compiler.schedule(
    max_iterations=10,         # Max iterations (not currently used)
    time_limit=60.0,          # Wall-clock time limit
    internal_rewrites=True,    # Phase 1: Expand e-graph with algebraic rules
    external_passes=None,      # Phase 2: Expand e-graph with MLIR passes (TODO)
    custom_rewrites=True,      # Phase 3: Match custom instructions (final pass)
    enable_safeguards=True     # Time limits and error handling
)
```

The schedule executes in order:
1. **Phase 1 - Internal rewrites**: Algebraic laws + constant folding → expand e-graph
2. **Phase 2 - External rewrites**: MLIR passes → further expand e-graph (TODO)
3. **Phase 3 - Custom instruction matching**: Pattern matching → replace with custom_instr (final pass)
4. **Extraction**: Extract best program from e-graph via `_extract_optimized_module()`

**Extraction Process:**
- Wraps each e-graph in `MeggEGraph` (Python dataclass representation)
- Uses `Extractor` with customizable cost function
- Extracts best expression trees from root e-classes
- Reconstructs MLIR via `ExprTreeToMLIR.reconstruct()`
- Preserves `memref.global` definitions from original module
- Falls back to original function on extraction failure

**Debugging:**
```python
compiler.visualize_egraph("output.svg")  # GraphViz visualization
```

### 2. E-graph Translation ([python/megg/egraph/](../python/megg/egraph/))

**FuncToTerms ([func_to_terms.py](../python/megg/egraph/func_to_terms.py)):**
- Converts MLIR `func.func` operations to e-graph `Term` nodes
- Maintains bidirectional mappings:
  - `ssa_to_term`: SSA values → Term nodes (def-use chain)
  - `ssa_to_id`: SSA values → numeric IDs
- Generates type rules for e-graph type consistency
- Tracks function entry block and arguments

**Term System ([term.py](../python/megg/egraph/term.py)):**
- Defines egglog datatype for program representation
- Covers arithmetic ops, control flow, memref access
- `Term`: Regular operations (Add, Mul, Load, Store, etc.)
- `LitTerm`: Literal values (integers, floats)

**MeggEGraph ([megg_egraph.py](../python/megg/egraph/megg_egraph.py)):**
- Python wrapper around egglog's serialized e-graph
- Dataclass-based representation:
  - `MeggENode`: Individual e-nodes with operation and children
  - `MeggEClass`: Equivalence classes containing nodes
  - `ExpressionNode`: Extracted expression trees
- Tracks root e-classes (function outputs)
- Provides statistics and debugging info

**Extractor ([extract.py](../python/megg/egraph/extract.py)):**
- Greedy extraction algorithm (adapted from egg)
- Customizable cost functions:
  - `AstSize`: Minimize total AST size (uniform cost = 1 per node)
  - `AstDepth`: Minimize expression depth
  - `OpWeightedCost`: Custom per-operation weights
  - `ConstantFoldingCost`: Prefer constant expressions
  - `MeggCost`: **Hardware-aware cost model** with operation-specific costs:
    - Literals/arguments: 1.0 (cheapest)
    - Basic arithmetic (add/sub): 2.0
    - Complex arithmetic (mul): 4.0, (div/rem): 8.0
    - Type casts: 2.0-5.0 (based on complexity)
    - Comparisons: 3.0
    - Control flow (if/for): 10.0-20.0 (expensive)
    - Memory operations: 5.0-15.0
    - Custom instructions: 15.0 (hardware-specific, customizable)
  - Supports custom cost overrides via `MeggCost(custom_costs={...})`
- Returns best expression tree per e-class

**ExprTreeToMLIR ([terms_to_func.py](../python/megg/egraph/terms_to_func.py)):**
- Reconstructs MLIR from extracted expression trees
- Preserves types and block structure from original function
- Handles control flow (if/for/while)
- Caches constants to avoid duplication
- **Custom instruction support**: Converts `Term.custom_instr` to `llvm.inline_asm`
  - Assembly string from instruction name
  - Automatic constraint generation (`=r` for output, `r` for inputs)
  - Side-effect tracking (based on return type)
  - Example: `custom_instr("vadd.vv", [a,b])` → `llvm.inline_asm "vadd.vv" "=r,r,r"`

### 3. Rewrite Engine ([compiler.py](../python/megg/compiler.py#L99-L214))

**Phase 1 - Internal Rewrites (Expand E-graph):**
- Method: `apply_internal_rewrites()`
- Applies egglog algebraic rules from `rewrites/internal_rewrites.py`
- Two main rule sets:
  - `basic_math_laws()`: Distributivity, identities, simplification
  - `constant_folding_laws()`: Compile-time constant evaluation
- Saturates e-graph (applies until fixpoint)
- **Goal**: Expand e-graph with all algebraic equivalences
- Reports rule count (not match count)

**Phase 2 - External Rewrites (Expand E-graph) [TODO]:**
- Method: `apply_external_rewrites(passes)`
- Skeleton for MLIR pass integration exists
- Would extract from e-graph → apply MLIR passes → re-insert
- **Goal**: Further expand e-graph with MLIR optimizations
- Currently returns 0 (not implemented)

**Phase 3 - Custom Instruction Matching (Final Pass):**
- Method: `apply_custom_rewrites()`
- Applies pattern matching rules from `match_ruleset`
- Runs **after** e-graph has been fully expanded
- Lightweight: only matches patterns, doesn't explore further
- **Goal**: Replace complex patterns with custom_instr nodes
- Reports pattern count

### 4. Rewrite Rules ([python/megg/rewrites/](../python/megg/rewrites/))

**Internal Rules ([internal_rewrites.py](../python/megg/rewrites/internal_rewrites.py)):**
- Algebraic identities: `x + 0 = x`, `x * 1 = x`, `x * 0 = 0`
- Distributivity: `(a*c) + (b*c) = (a+b)*c`
- Simplification: `-(-x) = x`, `x - x = 0`
- Constant folding: Evaluate operations on literals
- Type-aware rules (separate for i32, f32, f64)

**Match Rewrites ([match_rewrites.py](../python/megg/rewrites/match_rewrites.py)):**
- Custom instruction pattern matching for ASIP targets
- Converts MLIR function definitions → egglog rewrite rules
- Enables domain-specific optimization
- **Applied as Phase 3** (after e-graph expansion)
- **Key Design**: Lightweight final pass on expanded e-graph
- Example: Define `@mac(a,b,c) = a*b+c` in MLIR → auto-generates matching rules

### 5. Backend ([backend/llvm_backend.py](../python/megg/backend/llvm_backend.py))

**MLIRToLLVMBackend:**
- Lowers MLIR to LLVM IR using `mlir-opt` and `mlir-translate`
- Compiles to object files via `llc`
- Links executables via `clang`
- Graceful degradation when tools unavailable
- Supports optimization levels (O0-O3)
- Target triple specification

---

## State Management and Error Handling

### StateManager ([compiler.py](../python/megg/compiler.py#L45-L89))
- Tracks compilation progress
- Enforces wall-clock time limits
- Counts internal/external rewrites
- Generates `CompilationResult` with statistics

### Error Handling Strategy
- `schedule()` catches all exceptions
- Returns `CompilationResult` with `success=False` on failure
- Falls back to original module when extraction fails
- Logs errors with full context
- Subprocess calls use `capture_output=True` for debugging

### Debugging Support
- `--verbose`: Detailed logging
- `--dump-egraph`: GraphViz visualization
- `--log-level`: Configurable log verbosity
- E-graph statistics via `MeggEGraph.get_statistics()`

---

## Supported MLIR Operations

### Arithmetic Dialect
- Integer: `addi`, `subi`, `muli`, `divsi`, `divui`, `remsi`, `remui`
- Float: `addf`, `subf`, `mulf`, `divf`, `negf`
- Comparison: `cmpi`, `cmpf`

### Memory Dialect
- `memref.load`, `memref.store`
- `memref.alloc`, `memref.dealloc`
- `memref.global` (preserved during extraction)

### Control Flow
- `scf.if`, `scf.for`, `scf.while`
- `func.return`

### Types
- Integer: `i1`, `i8`, `i16`, `i32`, `i64`
- Float: `f16`, `f32`, `f64`
- Index: `index`
- Memref: `memref<...>`
- Tensor: `tensor<...>` (partial support)

---

## Current Limitations

### Unsupported Features
- **Function calls**: Functions with `call` operations are skipped
- **Multi-region control flow**: Complex nested regions
- **Dynamic shapes**: Tensor/memref with unknown dimensions
- **LLVM dialect**: Direct LLVM IR operations

### External Rewrites
- MLIR pass integration is stubbed (returns 0)
- Skeleton exists in `RewriteEngine.apply_external_rewrites()`
- Would require: extract → apply passes → union back to e-graph

### Visualization
- Only first e-graph visualized via `visualize_egraph()`
- Multi-function visualization requires refactoring

### CLI Gaps
- `--dump-state` parsed but not implemented
- Would need serialization of full compiler state

---

## Design Decisions

### Why E-graphs?
- **Equivalence saturation**: Explore all equivalent programs simultaneously
- **Phase-ordering solution**: No need to pick optimization order
- **Efficient deduplication**: Share common subexpressions automatically
- **Bidirectional rewrites**: Rules can fire in any direction

### Why 3-Phase Pipeline?
**Core Philosophy**: Expand e-graph first, match custom instructions last

1. **Phase 1 - Internal Rewrites (Expand)**
   - Apply algebraic laws to create many equivalent forms
   - Generate more opportunities for pattern matching
   - Example: `(a+b)*c` ↔ `a*c + b*c` (both forms in e-graph)

2. **Phase 2 - External Rewrites (Expand)**
   - Apply MLIR passes to add more equivalences
   - Leverage existing compiler infrastructure
   - Example: CSE, loop unrolling, etc.

3. **Phase 3 - Custom Instructions (Match)**
   - **Lightweight**: Only pattern matching, no further exploration
   - **Maximum coverage**: Matches against fully expanded e-graph
   - **ASIP focus**: Core goal is to match as many custom instructions as possible
   - Example: `a*c + b*c` matches MAC if pattern exists in e-graph

**Why separate Phase 3?**
- Internal/external rewrites are **exploratory** (expand search space)
- Custom matching is **decisive** (select best patterns)
- Separating them allows maximum e-graph expansion before final selection

### Why Separate Internal/External Rewrites?
- **Internal (egglog)**: Fast algebraic simplification, no round-trip overhead
- **External (MLIR passes)**: Leverage existing optimizations, harder analyses
- **Best of both**: Combine strengths of both systems

### Why Custom Cost Functions?
- **Target-specific**: Different architectures prefer different code shapes
- **Flexible**: Easy to add new optimization criteria
- **Composable**: Combine multiple cost metrics

### Why MeggEGraph Wrapper?
- **Debuggable**: Python dataclasses easier to inspect than raw egglog
- **Extensible**: Add metadata without modifying egglog
- **Type-safe**: Static type checking with dataclasses

---

## Future Work

### High Priority
1. **External rewrite integration**: Complete MLIR pass round-tripping
2. **Function call support**: Handle inter-procedural optimization
3. **Multi-region control flow**: Better CFG representation

### Medium Priority
4. **State serialization**: Implement `dump_state()` for debugging
5. **Multi-function visualization**: Render all e-graphs
6. **Dynamic shapes**: Support unknown tensor dimensions

### Low Priority
7. **Custom cost model library**: Architecture-specific cost functions
8. **Parallel e-graph saturation**: Speed up internal rewrites
9. **Incremental compilation**: Reuse e-graphs across edits

