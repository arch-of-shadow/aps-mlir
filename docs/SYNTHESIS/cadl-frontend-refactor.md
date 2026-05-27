# CADL Frontend MLIR Converter Refactor Notes

This document analyzes the current CADL frontend structure with a focus on `cadl_frontend/to_mlir/converter.py`. The refactor goal is not to redesign the frontend for its own sake. It is to keep the `aps-e2e` synthesis flow stable while making frontend changes easier to reason about and test.

## Current Frontend Path

The synthesis entry path is:

```text
aps-frontend mlir <input.cadl>
  -> cadl_frontend.parser.parse_proc
  -> cadl_frontend.to_mlir.convert_cadl_to_mlir
  -> CADLMLIRConverter.convert_proc
  -> APS-flavored MLIR consumed by aps-e2e
```

The public API used by tests and scripts is:

```python
from cadl_frontend.to_mlir import CADLMLIRConverter, convert_cadl_to_mlir
```

That import surface should remain stable during the first refactor stages.

## Existing Module Layout

| File | Current role |
| --- | --- |
| `cadl_frontend/grammar.lark` | Concrete CADL grammar. |
| `cadl_frontend/parser.py` | Lark parser, error formatting, AST construction. |
| `cadl_frontend/cadl_ast.py` | CADL AST classes, type nodes, literal nodes, helper utilities, and explicit AST exports. |
| `cadl_frontend/to_mlir/converter.py` | Public converter facade and remaining flow/statement/expression lowering. |
| `cadl_frontend/to_mlir/state.py` | Shared conversion state, symbol scopes, global registry. |
| `cadl_frontend/to_mlir/types.py` | CADL type to MLIR type conversion. |
| `cadl_frontend/to_mlir/memory.py` | `_mem` usage scan, static/global declaration, global references, `_irf`, `_mem`, static array, and burst memory lowering. |
| `cadl_frontend/to_mlir/loop.py` | Loop-specific analysis and emission helper for `do-while` to `scf.for` / `scf.while`. |
| `cadl_frontend/to_c/transpiler.py` | CADL-to-C path used by compiler/pattern matching flow. |
| `aps-frontend` | CLI wrapper for parse, MLIR conversion, C transpilation, and encoding extraction. |

## What `to_mlir/converter.py` Currently Does

`to_mlir/converter.py` used to be the large `mlir_converter.py` entry point and contains several separate responsibilities:

### 1. Converter State and Symbol Management

Moved definitions:

- `SymbolBinding`
- `TypedValue` compatibility alias
- `SymbolKind`
- `SymbolScope`
- `GlobalRegistry`
- `ConversionState`

Compatibility properties/methods on `CADLMLIRConverter`:

- `current_scope`
- `scope_stack`
- `current_global_refs`
- `global_ops`
- `constant_vars`
- `pending_directives`
- `get_symbol`
- `set_symbol`
- `get_symbol_type`

Responsibilities:

- Keep SSA values and CADL type metadata together.
- Resolve names across nested scopes.
- Cache `memref.get_global` values.
- Track globals for type inference.
- Track compile-time constants used by loop analysis.

This is foundational state, but it is not MLIR emission logic by itself.

### 2. Dialect and Module Setup

Methods:

- `_load_dialects`
- `convert_proc`
- `convert_cadl_to_mlir`

Responsibilities:

- Create MLIR context/module.
- Register CIRCT dialects.
- Emit module-level declarations.
- Run optional CSE at the end.

This is orchestration logic and should remain close to the public converter API.

### 3. CADL Type to MLIR Type Conversion

Method/proxy:

- `convert_cadl_type`

Responsibilities:

- Map CADL integer/fixed/basic types to signless MLIR integer or float types.
- Map CADL arrays to `memref` types.
- Reject unsupported frontend types such as strings.

The actual mapping now lives in the pure `cast_cadl_type_to_mlir()` helper in `to_mlir/types.py`; the converter method remains as a stable proxy.

### 4. Global and Static Memory Emission

Converter proxy methods:

- `_declare_global_memory`
- `_get_global_reference`
- `_is_scalar_global`
- `_convert_static`
- `_convert_attribute_value`
- `_get_global_element_type`
- `_get_memref_for_symbol`

Responsibilities:

- Emit `memref.global`.
- Preserve `var_name` and user attributes.
- Convert static initializers and attribute expressions.
- Resolve static memory symbols back to memrefs.

Most declaration/reference logic now lives in `GlobalEmitter` in `to_mlir/memory.py`. This area remains tightly coupled to APS memory lowering and `aps-e2e`, because downstream passes expect `memref.global` metadata and later `aps.memorymap`.

### 5. AST Memory-Use Analysis

Converter proxy methods:

- `_function_uses_memory`
- `_flow_uses_memory`
- `_stmt_list_uses_memory`
- `_stmt_uses_memory`
- `_expr_uses_memory`

Responsibilities:

- Decide whether `_cpu_memory` must be declared.
- Scan AST statements and expressions for `_mem[...]`.

This logic is pure AST analysis and does not need MLIR context or builder state.
The implementation now lives as module-level helpers in `to_mlir/memory.py`, next to the memory/global lowering that consumes the result.

### 6. Flow and Statement Emission

Converter proxy methods:

- `_convert_flow`
- `_convert_stmt_list`
- `_convert_stmt`
- `_convert_do_while`
- plus delegation to `LoopTransformer`

Responsibilities:

- Create `func.func @flow_*`.
- Bind flow arguments.
- Emit function body operations.
- Convert assignments, expression statements, returns, directives, and loops.

This is currently one of the largest coupling points because statements call expression conversion, memory conversion, type conversion, and loop conversion.

### 7. Expression Emission

Methods:

- `_convert_expr`
- `_convert_binary_op`
- `_convert_unary_op`
- `_convert_index_expr`
- `_convert_slice_expr`
- `_convert_if_expr`
- `_convert_select_expr`
- `_convert_type_if_needed`
- `_cast_type`
- `_to_signless`
- `_promote_operands`
- `_is_signed_type`
- `_get_expr_signedness`

Responsibilities:

- Emit constants, identifiers, arithmetic, comparisons, shifts, casts, selects, indexing, slicing, and aggregate-like expressions.
- Manage integer width promotion and signedness-sensitive operations.

This is the highest-risk area to split because many CADL features interact here.

### 8. APS Memory and Burst Operation Emission

Methods:

- `_convert_index_assignment`
- `_is_burst_operation`
- `_convert_burst_operation`
- `_convert_burst_load`
- `_convert_burst_store`
- `_convert_range_slice_assignment`
- `_get_memref_element_type`
- `_extract_literal_value`

Responsibilities:

- Lower `_irf[...]` to `aps.readrf` / `aps.writerf`.
- Lower scratchpad/global memory indexing to `aps.memload` / `aps.memstore`.
- Lower range/burst assignments to `aps.memburstload` / `aps.memburststore`.

This is the core frontend contract with `aps-e2e`. It should be refactored carefully with golden MLIR tests.
The implementation now lives in `MemoryEmitter` in `to_mlir/memory.py`, with converter compatibility methods kept during migration.

## Main Problems

### Problem 1: One Class Owns Too Many Axes

`CADLMLIRConverter` currently owns:

- context/module lifecycle,
- symbol table,
- type conversion,
- static/global emission,
- AST analysis,
- statement lowering,
- expression lowering,
- memory/burst lowering,
- post-pass CSE.

This makes every small change feel risky because unrelated concerns share the same class state.

### Problem 2: Pure Analysis Is Mixed With MLIR Emission

The memory-use scan only needs AST nodes, but it lives as converter methods. This makes simple analysis harder to test without creating MLIR context.

### Problem 3: Type Conversion Is Not Isolated

Type conversion is used by expression lowering, static lowering, attribute lowering, and flow argument lowering. Changes to type behavior are difficult to audit because the logic is embedded in the converter class.

### Problem 4: Memory Semantics Are Spread Across Several Methods

Memory-related behavior appears in:

- `_declare_global_memory`
- `_get_global_reference`
- `_convert_static`
- `_convert_index_expr`
- `_convert_index_assignment`
- burst operation helpers
- scalar/global helper methods

For `aps-e2e`, memory handling is the most important frontend output contract, so this should eventually have a clearer boundary.

### Problem 5: Tests Mostly Validate Strings, Not Internal Boundaries

Existing MLIR tests check that generated MLIR contains expected op strings. That is useful for behavior preservation, but before splitting internals we need a few smaller tests for pure helpers such as memory-use analysis and type conversion.

## Refactor Principles

- Keep `convert_cadl_to_mlir` and `CADLMLIRConverter` import-compatible.
- Preserve generated MLIR first; improve structure second.
- Define ownership before moving functions. The first question is what `CADLMLIRConverter` should own.
- Keep `aps-e2e` examples as golden end-to-end checks.
- Avoid changing parser or AST structure unless a converter boundary requires it.
- Move one responsibility at a time and keep compatibility proxy methods where useful.

## What `CADLMLIRConverter` Should Be

`CADLMLIRConverter` should be a conversion session and facade, not the owner of every lowering rule.

It should own:

- MLIR context and module lifecycle.
- Dialect registration.
- Top-level conversion order:
  - analyze proc-level needs,
  - emit globals/statics,
  - emit flows,
  - optionally run cleanup passes such as CSE.
- Construction and wiring of helper components.
- Shared conversion state for one conversion run.
- The stable public API used by tests and CLI code.

It should not directly own:

- the full expression lowering matrix,
- statement-specific lowering rules,
- `_irf`, `_mem`, static memref, and burst operation lowering details,
- pure AST analyses,
- full CADL-to-MLIR type mapping rules,
- loop pattern detection internals.

A target shape is:

```python
class CADLMLIRConverter:
    def __init__(self):
        self.context = ir.Context()
        self.module = None
        self.state = ConversionState(...)
        self.memory = MemoryEmitter(self.state)
        self.expr = ExprEmitter(self.state, self.memory)
        self.stmt = StmtEmitter(self.state, self.expr, self.memory)
        self.loops = LoopEmitter(self.state, self.stmt, self.expr)

    def convert_proc(self, proc: Proc) -> ir.Module:
        ...

    def convert_flow(self, flow: Flow) -> ir.Operation:
        ...
```

During early migration, existing methods such as `_convert_expr`, `_convert_stmt`, and `convert_cadl_type` can remain as compatibility proxies:

```python
def _convert_expr(self, expr: Expr) -> ir.Value:
    return self.expr.emit(expr)
```

## Shared Conversion State

Emitter components need current conversion information, but that information should come from one shared session state rather than from each emitter owning duplicate state.

Proposed state objects:

```python
@dataclass
class ConversionState:
    context: ir.Context
    module: ir.Module | None
    symbols: SymbolTable
    globals: GlobalRegistry
    constants: ConstantRegistry
```

The exact names can change, but the ownership should stay clear:

| Object | Owns |
| --- | --- |
| `ConversionState` | Per-conversion mutable state shared by emitters. |
| `SymbolTable` | Lexical bindings from CADL names to SSA values, globals, or constants. |
| `GlobalRegistry` | `memref.global` operations and cached `memref.get_global` values. |
| `ConstantRegistry` | Compile-time constants used by analysis and loop lowering. |
| `cast_cadl_type_to_mlir()` | Pure CADL type to MLIR type mapping. |

The symbol table should store more than raw `ir.Value`:

```python
class SymbolKind(Enum):
    VALUE = "value"
    GLOBAL = "global"
    CONSTANT = "constant"

@dataclass
class SymbolBinding:
    value: ir.Value | str | int
    cadl_type: DataType | None = None
    kind: SymbolKind = SymbolKind.VALUE
```

This preserves the useful part of today's `TypedValue`, while making symbol meaning explicit.

## How Expression Emission Gets SSA Sources

`ExprEmitter` should not maintain its own SSA environment. It should read from `ConversionState.symbols`, which is updated by flow emission, statement emission, and loop emission.

Information flow:

```text
CADLMLIRConverter
  creates ConversionState and helper emitters
  enters a flow scope
  binds func arguments into state.symbols

StmtEmitter
  emits RHS expressions through ExprEmitter
  binds let/assignment results into state.symbols
  delegates stores to MemoryEmitter

ExprEmitter
  reads IdentExpr from state.symbols
  recursively emits subexpressions
  delegates memory-like IndexExpr to MemoryEmitter

MemoryEmitter
  uses state.globals and state.symbols
  emits aps.readrf / aps.memload / aps.memstore / burst ops
```

Example sketch:

```python
class ExprEmitter:
    def __init__(self, state, memory):
        self.state = state
        self.memory = memory

    def emit_ident(self, expr: IdentExpr) -> ir.Value:
        binding = self.state.symbols.lookup(expr.name)
        if binding is None:
            raise RuntimeError(f"Undefined symbol: {expr.name}")

        if binding.kind == SymbolKind.VALUE:
            return binding.value
        if binding.kind == SymbolKind.GLOBAL:
            return self.state.globals.get_ref(binding.value)
        if binding.kind == SymbolKind.CONSTANT:
            return self.emit_integer_constant(binding.value, binding.cadl_type)

        raise RuntimeError(f"Cannot use symbol {expr.name} as expression")
```

For index expressions, `ExprEmitter` should identify the expression shape but delegate hardware memory semantics:

```python
def emit_index(self, expr: IndexExpr) -> ir.Value:
    if isinstance(expr.expr, IdentExpr):
        base = expr.expr.name
        index = self.emit(expr.index)

        if base == "_irf":
            return self.memory.emit_irf_read(index)
        if base == "_mem":
            return self.memory.emit_cpu_mem_read(index)

        binding = self.state.symbols.lookup(base)
        if binding and binding.kind == SymbolKind.GLOBAL:
            return self.memory.emit_global_load(binding, [index])

    raise NotImplementedError(...)
```

The important rule is: SSA provenance lives in the session symbol table. `ExprEmitter` is a stateless service over the current session state, not a second owner of SSA bindings.

## Proposed Target Structure

Initial target:

```text
cadl_frontend/
  to_mlir/
    __init__.py            # public MLIR conversion API
    converter.py           # public converter facade and orchestration
    state.py               # ConversionState, SymbolTable, GlobalRegistry
    types.py               # cast_cadl_type_to_mlir pure type helper
    memory.py              # _mem scan, globals, _irf/_mem/static/burst lowering
    expr.py                # expression lowering
    stmt.py                # statement lowering
    loop.py                # loop helper, later narrowed
```

This is a direction, not a requirement to create every file immediately.

## Phased Change List

### Phase 0: Baseline and Golden Checks

Purpose: make sure refactoring has a measurable behavior baseline.

Changes:

- Record a small set of frontend examples for comparison:
  - `tutorial/cadl/hello.cadl`
  - `tutorial/cadl/v3ddist_vv.cadl`
  - one example with static arrays and burst load/store
  - one example with `do-while`
- Add or document a command to regenerate MLIR with CSE disabled where needed for stable diffs.
- Keep existing `tests/test_mlir` as behavior checks.

Validation:

```bash
pixi run python -m pytest -q tests/test_parser tests/test_mlir
pixi run mlir tutorial/cadl/v3ddist_vv.cadl /tmp/v3ddist_vv.mlir
pixi run opt /tmp/v3ddist_vv.mlir /tmp/v3ddist_vv_cmt.mlir
```

Current upstream-only baseline, checked with Pixi:

```text
tests/test_parser: 100 passed
tests/test_mlir: 57 passed
pixi run mlir tutorial/cadl/hello.cadl /tmp/hello.mlir: passed
pixi run mlir tutorial/cadl/v3ddist_vv.cadl /tmp/v3ddist_vv.mlir: passed
```

The old `TestSimpleMLIRConversions.test_static_variable` stale expectation was fixed. Scalar statics are checked as `aps.globalload`; array statics remain checked through global memref access.

### Checker Tightening Plan

The current MLIR tests are useful smoke tests, but many checks are loose substring checks. They can miss wrong operands, wrong types, missing attributes, or invalid operation placement.

Tighten checkers in this order:

1. Keep substring smoke checks for broad coverage, but fix stale expectations first.
2. Add small structured helpers that parse generated MLIR with CIRCT and inspect operation counts/names.
3. Add focused checks for important frontend contracts:
   - flow function name and `opcode` / `funct7` attributes,
   - `_irf[...]` read/write producing `aps.readrf` / `aps.writerf`,
   - `_mem[...]` producing `_cpu_memory` plus `aps.memload` / `aps.memstore`,
   - scalar static read producing `aps.globalload`,
   - array static read/write producing `memref.get_global` plus `aps.memload` / `aps.memstore`,
   - burst syntax producing `aps.memburstload` / `aps.memburststore`,
   - partition attributes preserved on `memref.global`.
4. Add pure unit tests for helpers once extracted:
   - symbol lookup/shadowing,
   - `_mem` memory-use analysis,
   - CADL type to MLIR type conversion.
5. Add a small golden-output layer for 2-3 representative CADL files, normalized enough to avoid SSA-number churn where possible.

The first checker fix should update `test_static_variable` to match current scalar static semantics, or split it into scalar-static and array-static tests with explicit expectations.

### Phase 1: Define Conversion State and Symbol Table

Status: implemented.

Purpose: make `CADLMLIRConverter` a session/facade with explicit shared state.

Changes:

- Create `cadl_frontend/to_mlir/state.py`.
- Move or recreate:
  - `SymbolBinding`,
  - `SymbolKind`,
  - `SymbolTable`,
  - `GlobalRegistry` shell,
  - `ConversionState`.
- Keep `CADLMLIRConverter.get_symbol`, `set_symbol`, and `get_symbol_type` as proxies during migration.
- Keep generated MLIR unchanged.

Risk:

- Low to medium. This touches state access broadly, but it should not touch emission behavior.

Validation:

- Parser tests should be unaffected.
- MLIR conversion tests should produce identical output.

### Phase 2: Extract Pure Memory-Use Analysis

Status: implemented.

Purpose: separate AST scanning from MLIR lowering.

Changes:

- Add module-level memory-use analysis helpers to `cadl_frontend/to_mlir/memory.py`.
- Move pure functions:
  - `flow_uses_memory(flow)`
  - `stmt_list_uses_memory(stmts)`
  - `stmt_uses_memory(stmt)`
  - `expr_uses_memory(expr)`
- Add unit tests for `_mem[...]` detection in assignments, expressions, returns, and loops.

Risk:

- Low to medium. Current scanner is incomplete for some expression forms; first extraction should preserve existing behavior before expanding coverage.

Validation:

- Existing MLIR tests.
- New pure unit tests for memory analysis.

### Phase 3: Extract Type Conversion

Status: implemented.

Purpose: isolate CADL type mapping rules.

Changes:

- Create `cadl_frontend/to_mlir/types.py`.
- Move `convert_cadl_type` logic into the pure `cast_cadl_type_to_mlir()` helper.
- Keep `CADLMLIRConverter.convert_cadl_type` as a proxy.
- Add tests for:
  - `uN` / `iN` to signless integer widths,
  - `usize` to index,
  - arrays to `memref`,
  - unsupported string types.

Risk:

- Medium. Type conversion affects almost all emission.

Validation:

- Existing MLIR tests.
- Focused type conversion tests.

### Phase 4: Extract Static/Global Emission

Status: implemented.

Purpose: centralize static memory and global symbol behavior.

Changes:

- Add `GlobalEmitter` to `cadl_frontend/to_mlir/memory.py`.
- Move static declaration helpers:
  - `_convert_static`
  - `_convert_attribute_value`
  - `_declare_global_memory`
  - `_get_global_reference`
  - `_is_scalar_global`
  - `_get_global_element_type`
- Use `ConversionState.globals`, `ConversionState.symbols`, and pure type helpers instead of passing the whole converter where possible.

Risk:

- Medium. Static metadata feeds `aps-memory-map`, so this must preserve attributes exactly.

Validation:

- MLIR tests for statics and partition attributes.
- `pixi run opt` on at least one burst/static-array example.

### Phase 5: Extract Memory Emitter Before Expression Emitter

Status: implemented with converter compatibility proxies still present.

Purpose: make the `aps-e2e` frontend contract explicit before splitting expression dispatch.

Changes:

- Create `cadl_frontend/to_mlir/memory.py`.
- Move or delegate:
  - `_irf` read/write,
  - `_mem` read/write,
  - static memref load/store,
  - burst load/store,
  - range slice assignment.
- Keep compatibility methods in `CADLMLIRConverter` while migrating.
- Make `ExprEmitter.emit_index` call `MemoryEmitter` for memory-like index expressions.
- Make `StmtEmitter` call `MemoryEmitter` for assignment-to-memory forms.

Risk:

- High. This is directly consumed by `aps-e2e`.

Validation:

- MLIR tests for `_irf`, `_mem`, statics, and burst operations.
- `pixi run opt` on at least one tutorial example.

### Phase 6: Separate Expression Emission

Status: implemented with `CADLMLIRConverter._convert_expr` kept as a compatibility proxy.

Purpose: isolate arithmetic/type-promotion logic.

Changes:

- Create `cadl_frontend/to_mlir/expr.py`.
- Move expression-related methods gradually:
  - pure-ish helpers first: signedness, promotion, cast helpers,
  - then `_convert_binary_op`, `_convert_unary_op`,
  - finally `_convert_expr`.
- Use `ConversionState.symbols` to resolve `IdentExpr`.
- Delegate memory-like `IndexExpr` to `MemoryEmitter`.

Risk:

- High. This is where most semantic bugs can enter.

Validation:

- Arithmetic, comparison, shift, cast, if/select, slice tests.
- Golden MLIR diff for representative examples.

### Phase 7: Separate Statement Emission

Status: implemented with `CADLMLIRConverter._convert_stmt_list` kept as a compatibility callback for `LoopTransformer`.

Purpose: isolate statement dispatch after state, type, memory, and expression boundaries are stable.

Changes:

- Create `cadl_frontend/to_mlir/stmt.py` for:
  - assignment dispatch,
  - return/expr/directive statements,
  - loop entry delegation.
- Keep `LoopTransformer` separate, but reduce its dependency on the whole converter where practical.

Risk:

- High. This stage touches the output that `aps-e2e` directly consumes.

Validation:

- Full `tests/test_mlir`.
- `pixi run mlir`, `pixi run opt`, and `pixi run sv` for one tutorial example.

## Recommended First Patch

The first implemented patch series covers Phase 1 through Phase 7:

1. Add the `to_mlir/` package with `state.py`, `types.py`, `memory.py`, `expr.py`, `stmt.py`, `loop.py`, and `converter.py`.
2. Keep `CADLMLIRConverter` and `convert_cadl_to_mlir` as the public API.
3. Keep only narrow converter compatibility callbacks for `LoopTransformer` and memory/expression recursion.
4. Tighten MLIR checkers so they inspect parsed MLIR operation structure, operands, result types, attributes, regions, and local def-use instead of relying on loose substring checks.

Next recommended patch: reduce `LoopTransformer`'s dependency on the full converter facade.

## Things Not To Do First

- Do not split `_convert_expr` first. It has the most hidden coupling.
- Do not create emitters before defining the shared state they will use.
- Do not change generated MLIR naming while restructuring.
- Do not change `convert_cadl_to_mlir` or `CADLMLIRConverter` imports.
- Do not change pass order in `aps-e2e` as part of frontend cleanup.
- Do not broaden memory analysis semantics in the same patch that extracts it; preserve behavior first, improve later.

## Open Questions

- Should the shared state be called `ConversionState`, `EmissionState`, or `MLIRSession`?
- Should `GlobalRegistry` own `memref.get_global` insertion, or should `MemoryEmitter` own that and use the registry only as metadata?
- Should CSE remain in `convert_cadl_to_mlir`, or should the CLI own post-processing passes?
- Should memory-use analysis eventually detect burst/static memory use separately from `_mem` CPU memory use?
- Should `LoopTransformer` become an emitter with a narrower interface, rather than holding the full converter?
- Which examples should become golden MLIR snapshots for frontend refactors?
