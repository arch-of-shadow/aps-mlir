# Custom Instruction Matching in Megg

This document describes the custom instruction matching system in Megg, which allows users to define ASIP (Application-Specific Instruction Processor) instructions as MLIR functions and automatically match and replace corresponding patterns in the input code.

## Overview

The custom instruction matching system supports two distinct pattern types:

1. **Simple Computation Patterns**: Straight-line computation without control flow or side effects
2. **Complex Control Flow Patterns**: Patterns involving control flow structures (loops, conditionals)

Both pattern types are defined as MLIR `func.func` operations, where the function body represents the pattern to match.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input: Pattern Definition                     │
│              (MLIR func.func with pattern body)                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Pattern Classification                         │
│  • Detect if pattern is simple or complex                       │
│  • Simple: No control flow (scf.for/if/while), no side effects  │
│  • Complex: Contains control flow structures                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
┌──────────────────────────┐  ┌──────────────────────────┐
│   Simple Pattern Path    │  │  Complex Pattern Path    │
│                          │  │                          │
│  1. Extract pattern from │  │  1. Build skeleton       │
│     func.return operand  │  │     structure (control   │
│  2. Generate egglog      │  │     flow + leaf patterns)│
│     rewrite rule         │  │  2. Generate component   │
│  3. Apply during e-graph│  │     rewrites for leaves  │
│     saturation           │  │  3. Match skeleton in    │
│                          │  │     MeggEGraph           │
│                          │  │  4. Add custom_instr node│
└──────────────────────────┘  └──────────────────────────┘
                    │                   │
                    └─────────┬─────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Custom Instruction Node                       │
│  • Stored in e-graph with operands and result type             │
│  • Extracted as llvm.inline_asm in optimized MLIR              │
└─────────────────────────────────────────────────────────────────┘
```

## Pattern Classification

### Detection Logic

When processing a pattern definition function, the system checks the following:

```python
# Check for control flow operations
has_control_flow = any(op.name in ['scf.for', 'scf.if', 'scf.while']
                       for op in operations)

# Check for side effects
has_side_effects = any(op.name in ['scf.yield', 'memref.store', 'memref.load']
                       for op in operations)

# Classification
if not has_control_flow and not has_side_effects:
    return SimplePattern
else:
    return ComplexPattern
```

### Simple Pattern Criteria

A pattern is classified as **simple** if:
- ✅ No control flow operations (`scf.for`, `scf.if`, `scf.while`)
- ✅ No side-effect operations (except `func.return`)
- ✅ Straight-line computation only

### Complex Pattern Criteria

A pattern is classified as **complex** if:
- ✅ Contains control flow operations
- ✅ May contain side-effect operations (`scf.yield`, memory operations)

## Simple Pattern Matching

### Phase 1: Pattern Extraction

**Location**: `python/megg/rewrites/match_rewrites.py::_build_skeleton_from_func()`

1. Create generic variables for function arguments (e.g., `a`, `b`, `c`)
2. Build SSA value → Term mapping by converting each operation
3. Extract the pattern from `func.return` operand
4. Extract result type using `mlir_type_to_egraph_ty_string()`

**Example**:
```mlir
func.func @flow_arithmetic(%arg0: i32, %arg1: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %0 = arith.cmpi sgt, %arg0, %c0 : i32
  %1 = arith.select %0, %arg0, %arg1 : i32
  return %1 : i32
}
```

Extracted pattern:
```python
pattern = Term.select(
    Term.cmpi("sgt", a, Term.lit(0, "i32"), "i1"),
    a,
    b,
    "i32"
)
result_type = String("i32")
arg_vars = [a, b]
```

### Phase 2: Rewrite Rule Generation

**Location**: `python/megg/rewrites/match_rewrites.py::build_ruleset_from_module()`

Generate an egglog rewrite rule:
```python
custom_instr = Term.custom_instr(
    String("flow_arithmetic"),
    Vec[Term](a, b),          # Operands
    String("i32")              # Result type
)

rewrite = egglog.rewrite(pattern).to(custom_instr)
```

### Phase 3: Application

The rewrite rule is applied during e-graph saturation (internal rewrite phase). When the pattern matches, the e-graph automatically adds a `custom_instr` node to the matching eclass.

### Phase 4: Extraction

**Location**: `python/megg/egraph/terms_to_func.py::_reconstruct_custom_instr()`

During extraction, `custom_instr` nodes are converted to `llvm.inline_asm`:

**Input (e-graph)**:
```
Custom_instr("flow_arithmetic", Vec[Arg(0), Arg(1)], "i32")
```

**Output (MLIR)**:
```mlir
%0 = llvm.inline_asm "flow_arithmetic", "=r,r,r" %arg0, %arg1 : (i32, i32) -> i32
```

## Complex Pattern Matching

### Phase 1: Skeleton Building

**Location**: `python/megg/rewrites/match_rewrites.py::_build_skeleton_from_func()`

Build a hierarchical skeleton representing the control flow structure:

**Data Structures**:
```python
@dataclass
class SkeletonNode:
    """Control flow node (e.g., scf.for, scf.if)"""
    container_type: str        # "scf.for", "scf.if", "func.body"
    blocks: List[SkeletonBlock]
    result_type: Optional[str]

@dataclass
class SkeletonBlock:
    """A block within a control flow node"""
    name: str                  # "body", "then", "else"
    statements: List[SkeletonStmt]

@dataclass
class SkeletonStmt:
    """A statement in a block"""
    name: str
    pattern_term: Optional[Term]           # Leaf pattern (e.g., yield)
    nested_skeleton: Optional[SkeletonNode] # Nested control flow

@dataclass
class Skeleton:
    """Complete skeleton for an instruction"""
    instr_name: str
    root: SkeletonNode
    leaf_patterns: Dict[str, Term]
    arg_vars: List              # Function argument variables
    result_type: Optional[egglog.String]
```

**Example**: For a loop pattern:
```mlir
func.func @flow_loop(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32
  %c1 = arith.constant 1 : i32
  %0 = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %c0) -> (i32) {
    %1 = arith.muli %arg0, %arg1 : i32
    %2 = arith.addi %acc, %1 : i32
    scf.yield %2 : i32
  }
  return %0 : i32
}
```

Skeleton structure:
```python
Skeleton(
    instr_name="flow_loop",
    root=SkeletonNode(
        container_type="scf.for",
        blocks=[SkeletonBlock(
            name="body",
            statements=[SkeletonStmt(
                name="body_stmt1",
                pattern_term=Term.yield_(
                    Vec[Term.add(acc, Term.mul(a, b, "i32"), "i32")],
                    "void"
                )
            )]
        )]
    ),
    arg_vars=[a, b, c],
    result_type=String("i32")
)
```

### Phase 2: Component Rewrite Generation

For each leaf pattern in the skeleton, generate a component rewrite:

```python
for full_name, pattern in skeleton.leaf_patterns.items():
    comp_instr = Term.component_instr(
        String(full_name),
        Vec[Term](),
        String("void")
    )
    rewrite = egglog.rewrite(pattern).to(comp_instr)
```

These component rewrites mark the leaf patterns during e-graph saturation.

### Phase 3: Skeleton Matching in MeggEGraph (Optimized)

**Location**: `python/megg/egraph/megg_egraph.py::SkeletonMatcher`

After e-graph saturation, the skeleton matcher uses an **indexed search strategy**:

1. **Build control flow index** (one-time O(N) cost during MeggEGraph construction):
   - Index all control flow nodes by operation type
   - `Term.for_with_carry`, `Term.if_`, `Term.while_`, `Term.block`

2. **Fast candidate lookup** (O(1)):
   - Query index for matching control flow type
   - Only process relevant nodes (not all eclasses)

3. **Lightweight component verification** (O(components)):
   - Extract body block statements
   - Verify all expected `component_instr` nodes exist
   - No recursive structure validation needed

4. **Return matches**: List of `(eclass_id, component_bindings)` tuples

**Optimized matching process**:
```python
# Use index for O(1) lookup
op_type = SKELETON_TYPE_TO_OP[skeleton.root.container_type]
candidates = egraph.cf_index.get(op_type, [])

# Only check relevant candidates
for candidate_node in candidates:
    if verify_components(candidate_node, skeleton):
        matches.append((candidate_node.eclass, {}))
```

**Performance improvement**:
- Before: O(N × skeletons) where N = total eclasses
- After: O(candidates × components) where candidates << N
- Typical speedup: **50-100x** for large e-graphs

### Phase 4: Custom Instruction Node Insertion

**Location**: `python/megg/compiler.py::_apply_skeleton_matching()`

For each match, add a `custom_instr` node to the matched eclass:

1. **Find argument eclasses**: Search for `Term.arg(i)` nodes in MeggEGraph
   ```python
   arg_eclasses = []
   for i in range(num_args):
       for enode in megg_egraph.enodes.values():
           if enode.op == "Term.arg" and enode.value == str(i):
               arg_eclasses.append(enode.eclass)
   ```

2. **Extract result type**: Convert `skeleton.result_type` to string
   ```python
   result_type_str = str(skeleton.result_type)  # "String("i32")"
   result_type_str = result_type_str[8:-2]       # "i32"
   ```

3. **Add custom_instr node**:
   ```python
   node_id = megg_egraph.add_custom_instr_node(
       eclass_id=matched_eclass_id,
       instr_name=skeleton.instr_name,
       operands=arg_eclasses,      # List of eclass IDs
       result_type=result_type_str
   )
   ```

### Phase 5: Extraction

**Location**: `python/megg/egraph/terms_to_func.py::_reconstruct_custom_instr()`

During extraction, handle two cases for operand reconstruction:

**Case 1: Simple pattern (children[0] is Vec)**
```python
if len(expr.children) == 1 and expr.children[0].op == 'Vec':
    operands_expr = expr.children[0]
    operand_values = self._collect_values_from_expr(operands_expr, block)
```

**Case 2: Complex pattern (children is direct operand list)**
```python
else:
    for child_expr in expr.children:
        val = self._reconstruct_expr(child_expr, block)
        if val is not None:
            operand_values.append(val)
```

**Output (MLIR)**:
```mlir
%0 = llvm.inline_asm "flow_loop", "=r,r,r,r" %arg0, %arg1, %arg2 : (i32, i32, i32) -> i32
```

## Key Implementation Details

### Operand Handling

**Simple Patterns**:
- Operands are generic variables (e.g., `a`, `b`) embedded in the pattern
- Passed via `Vec[Term](*arg_vars)` in the rewrite rule
- Automatically bound during pattern matching

**Complex Patterns**:
- Operands are function argument eclasses found via search
- Stored directly as `node.children` in the custom_instr node
- Extracted by reconstructing each child expression

### Result Type Handling

**Extraction**:
```python
# From MLIR type
return_value = func_return_op.operands[0]
result_type = mlir_type_to_egraph_ty_string(return_value.type)
# Returns: egglog.String("i32")
```

**Storage**:
- Simple: In `Tuple[Term, egglog.String, List]`
- Complex: In `Skeleton.result_type`

**Usage**:
- Set `eclass.dtype` when creating custom_instr node
- Used to generate MLIR result type in `llvm.inline_asm`

### Constraint String Generation

The constraint string for `llvm.inline_asm` is automatically generated:

```python
# Format: "=r" for output, ",r" for each input
num_inputs = len(operand_values)
if result_type is not None:
    constraints = "=r" + (",r" * num_inputs)  # "=r,r,r" for 2 inputs
else:
    constraints = ",".join(["r"] * num_inputs)  # "r,r" for 2 inputs (no output)
```

## File Organization

```
python/megg/
├── rewrites/
│   └── match_rewrites.py       # Pattern extraction and skeleton building
├── compiler.py                  # Skeleton matching orchestration
├── egraph/
│   ├── megg_egraph.py          # SkeletonMatcher and custom_instr node creation
│   ├── extract.py              # Cost model for custom instructions
│   ├── terms_to_func.py        # Reconstruction to MLIR (llvm.inline_asm)
│   └── func_to_terms.py        # Type conversion utilities
└── utils/
    └── ir_builder.py            # MLIR IR building helpers (inline_asm)
```

## Usage Example

### Input Code
```mlir
func.func @example(%a: i32, %b: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %cmp = arith.cmpi sgt, %a, %c0 : i32
  %result = arith.select %cmp, %a, %b : i32
  return %result : i32
}
```

### Pattern Definition
```mlir
func.func @flow_arithmetic(%arg0: i32, %arg1: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %cmp = arith.cmpi sgt, %arg0, %c0 : i32
  %result = arith.select %cmp, %arg0, %arg1 : i32
  return %result : i32
}
```

### Command
```bash
./megg-opt input.mlir --custom-instructions pattern.mlir -o output.mlir
```

### Output
```mlir
func.func @example(%a: i32, %b: i32) -> i32 {
  %0 = llvm.inline_asm "flow_arithmetic", "=r,r,r" %a, %b : (i32, i32) -> i32
  return %0 : i32
}
```

## Cost Model

Custom instructions have low cost to encourage their selection during extraction:

```python
class MeggCost:
    default_costs = {
        'Term.custom_instr': 0.1,        # Very cheap - prefer custom instructions
        'Term.component_instr': float('inf'),  # Never extract components
        # ... other operations ...
    }
```

## Limitations and Future Work

### Current Limitations

1. **No partial matching**: Patterns must match completely
2. **No commutative matching**: Pattern order must match exactly
3. **Limited type polymorphism**: Types must match exactly
4. **No variadic operands**: Fixed number of operands per instruction

### Future Enhancements

1. **Partial matching**: Allow matching sub-patterns within larger expressions
2. **Commutative matching**: Automatically try operand permutations for commutative operations
3. **Type wildcards**: Allow generic type parameters in patterns
4. **Variadic instructions**: Support instructions with variable operand counts
5. **Cost annotations**: Allow users to specify custom instruction costs
6. **Multi-output instructions**: Support instructions returning multiple values

## Testing

Test cases are located in `tests/match/match_cases/`:

- `03-mux/`: Simple computation pattern (select/cmpi)
- `04-simple-loop/`: Complex control flow pattern (scf.for loop)

Run tests:
```bash
# Simple pattern
./megg-opt tests/match/match_cases/03-mux/03_.mlir \
  --custom-instructions tests/match/match_cases/03-mux/03.mlir \
  -o output.mlir

# Complex pattern
./megg-opt tests/match/match_cases/04-simple-loop/04_.mlir \
  --custom-instructions tests/match/match_cases/04-simple-loop/04.mlir \
  -o output.mlir
```

Verify output:
```bash
mlir-opt output.mlir --verify-diagnostics
```

## References

- [Equality Saturation](https://arxiv.org/abs/2004.03082) - Theoretical foundation
- [MLIR Dialects](https://mlir.llvm.org/docs/Dialects/) - MLIR dialect documentation
- [egglog](https://github.com/egraphs-good/egglog) - E-graph library used for rewriting
