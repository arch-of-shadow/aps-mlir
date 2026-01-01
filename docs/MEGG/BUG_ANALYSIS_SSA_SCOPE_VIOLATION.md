# Bug Analysis: SSA Scope Violation in E-graph Reconstruction

**Date**: 2025-11-08
**Component**: `python/megg/egraph/terms_to_func.py`
**Severity**: High
**Status**: Diagnosed, Fix Ready

---

## Problem Statement

The compilation of `tests/e2e/pcl/vfpsmax/compile.sh` fails with MLIR verification errors. The generated `tmp/megg_e2e_20251108_105634/optimized.mlir` contains invalid SSA references where values defined in inner scopes (nested `scf.if` blocks) are referenced in outer scopes, violating MLIR's SSA dominance rules.

### Error Manifestation

```
tmp/megg_e2e_20251108_105634/optimized.mlir:109:23: error: use of undeclared SSA value name
          "scf.yield"(%204, %212) : (i32, i32) -> ()
                      ^
tmp/megg_e2e_20251108_105634/optimized.mlir:286:33: error: use of undeclared SSA value name
      %38 = "arith.muli"(%arg3, %172) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
                                ^
```

**Root Observation**: Values like `%172`, `%174`, `%204`, `%212`, etc., are defined inside nested `scf.if` blocks (lines 48-89) but are incorrectly referenced outside their lexical scope (e.g., line 286 in a later loop).

---

## Root Cause Analysis

The bug originates from **inconsistent cache key management** and **incorrect block context handling** in the expression tree reconstruction logic.

### Core Issue 1: Cache Key Mismatch

**Location**: `python/megg/egraph/terms_to_func.py`

The code uses a **block-aware caching mechanism** to prevent SSA scope violations:
- Cache key format: `(expr_node_id, block_id)` where `block_id = id(target_block)`

**In `_reconstruct_expr` (lines 193-194)**:
```python
target_block = self.current_block_context if self.current_block_context else block
cache_key = (expr.node_id, id(target_block))
```
✅ Uses tuple `(node_id, block_id)` as cache key.

**In `_pre_materialize_branch_dependencies` (line 603)**:
```python
if not branch_expr or branch_expr.node_id in self.expr_to_ssa:
    return
```
❌ Checks only `node_id` (string), **not** the tuple cache key!

**Result**: Cache checks always fail → expressions are reconstructed multiple times in different blocks → SSA scope violations.

---

### Core Issue 2: Block Context Desynchronization

**Location**: `_reconstruct_if` method (lines 660-714)

**Timeline of Context Changes**:

1. **Pre-materialization phase** (line 662-663):
   ```python
   self._pre_materialize_branch_dependencies(then_expr, block)
   self._pre_materialize_branch_dependencies(else_expr, block)
   ```
   - `current_block_context` = outer block (or possibly a nested block from parent)
   - Operations are materialized with cache key: `(node_id, id(outer_block))`

2. **Branch reconstruction phase** (line 704, 710):
   ```python
   self.current_block_context = then_block
   self._reconstruct_branch(then_expr, then_block)

   self.current_block_context = else_block
   self._reconstruct_branch(else_expr, else_block)
   ```
   - `current_block_context` = inner block (then/else)
   - Attempts to lookup cache with key: `(node_id, id(inner_block))`
   - **Cache miss** → Re-constructs operations
   - **SSA values created in wrong scope**

**Result**: Pure operations that should be shared across blocks are duplicated, and some get placed in incorrect scopes.

---

## Technical Diagnosis

### Why vfpsmax.c Triggers This Bug

The `vfpsmax_v` function contains:
- **Nested control flow**: Multiple `scf.for` loops with deeply nested `scf.if` statements
- **Sentinel value handling**: Complex conditional logic checking for zero values
- **Shared computations**: Index calculations like `i * 2`, `i * 2 + 1` are used across multiple branches

**E-graph behavior**:
1. During optimization, shared subexpressions are deduplicated in the e-graph
2. During extraction, the same expression node may be referenced from multiple scopes
3. During reconstruction:
   - Pre-materialization tries to create shared operations in outer scope
   - But cache key mismatch causes re-materialization in inner scopes
   - Inner-scope SSA values leak into outer scopes

**Specific failure pattern** (line 286):
```mlir
scf.for %arg3 = %1 to %5 step %3 {
  %38 = "arith.muli"(%arg3, %172) ...  // ❌ %172 is from inner scope of previous loop!
```

Value `%172` was defined inside a nested `scf.if` block in the **second loop** but is referenced in the **third loop**.

---

## Solution

### Fix 1: Correct Cache Lookup in Pre-materialization

**File**: `python/megg/egraph/terms_to_func.py`
**Location**: Lines 596-618 (`_pre_materialize_branch_dependencies` method)

**Current Code** (line 603):
```python
if not branch_expr or branch_expr.node_id in self.expr_to_ssa:
    return
```

**Fixed Code**:
```python
if not branch_expr:
    return

# Construct correct cache key (consistent with _reconstruct_expr)
target_block = self.current_block_context if self.current_block_context else block
cache_key = (branch_expr.node_id, id(target_block))
if cache_key in self.expr_to_ssa:
    return
```

**Rationale**: Ensures cache lookup uses the same tuple-based key as `_reconstruct_expr`.

---

### Fix 2: Synchronize Block Context During Pre-materialization

**File**: `python/megg/egraph/terms_to_func.py`
**Location**: Lines 660-665 (`_reconstruct_if` method)

**Current Code**:
```python
# IMPORTANT: Pre-materialize all operations that the branches will use
# This ensures they are created in the current block context, not inside the if/else blocks
self._pre_materialize_branch_dependencies(then_expr, block)
self._pre_materialize_branch_dependencies(else_expr, block)

# Check if branches yield values
has_then_yield_values = self._branch_yields_values(then_expr)
```

**Fixed Code**:
```python
# Save current context before pre-materialization
old_prematerialize_context = self.current_block_context
# Temporarily set context to outer block for consistent cache keys
self.current_block_context = block

# IMPORTANT: Pre-materialize all operations that the branches will use
# This ensures they are created in the current block context, not inside the if/else blocks
self._pre_materialize_branch_dependencies(then_expr, block)
self._pre_materialize_branch_dependencies(else_expr, block)

# Restore context
self.current_block_context = old_prematerialize_context

# Check if branches yield values
has_then_yield_values = self._branch_yields_values(then_expr)
```

**Rationale**: Ensures `current_block_context` is set to the outer block during pre-materialization, making cache keys consistent between pre-materialization and later lookups.

---

### Fix 3: Apply Similar Fixes to For/While Loops (If Needed)

**Locations to Check**:
- `_reconstruct_for` (around line 856)
- `_reconstruct_while` (around line 761)

**Action**: Search for any pre-materialization calls in these methods and apply the same context synchronization pattern.

---

## Implementation Steps

### Step 1: Apply Fix 1
Edit `python/megg/egraph/terms_to_func.py`, line 603:
```python
# Before
if not branch_expr or branch_expr.node_id in self.expr_to_ssa:

# After
if not branch_expr:
    return
target_block = self.current_block_context if self.current_block_context else block
cache_key = (branch_expr.node_id, id(target_block))
if cache_key in self.expr_to_ssa:
```

### Step 2: Apply Fix 2
Edit `python/megg/egraph/terms_to_func.py`, before line 662:
```python
# Add before pre-materialization calls
old_prematerialize_context = self.current_block_context
self.current_block_context = block

# Existing pre-materialization code
self._pre_materialize_branch_dependencies(then_expr, block)
self._pre_materialize_branch_dependencies(else_expr, block)

# Add after pre-materialization
self.current_block_context = old_prematerialize_context
```

### Step 3: Verify and Test
```bash
# Test vfpsmax compilation
cd /home/cloud/megg
tests/e2e/pcl/vfpsmax/compile.sh

# Verify MLIR correctness
3rdparty/llvm-project/install/bin/mlir-opt \
  tmp/megg_e2e_*/optimized.mlir \
  --verify-diagnostics
```

### Step 4: Regression Testing
Run full test suite to ensure no other cases are broken:
```bash
# Test all e2e cases
for dir in tests/e2e/pcl/*/; do
  echo "Testing $dir"
  timeout 120 bash "$dir/compile.sh" || echo "FAILED: $dir"
done
```

---

## Deeper Analysis: Why This Design Exists

### Purpose of Block-Aware Caching

The caching mechanism was introduced to solve the **SSA dominance problem** in e-graph reconstruction:

1. **E-graphs are scope-agnostic**: A single expression node can represent computations used in multiple blocks
2. **MLIR requires SSA dominance**: A value defined in block B cannot be used in block C unless B dominates C
3. **Block-aware cache**: `(node_id, block_id)` ensures the same expression is reconstructed in each block where it's needed

### Why Pre-materialization Was Added

Pre-materialization was added to optimize shared computations:
- **Goal**: Create pure operations (arithmetic, casts) in outer scopes so they can be shared by multiple branches
- **Benefit**: Reduces code duplication and enables better optimization

### Why the Bug Occurred

The pre-materialization feature was added **after** the block-aware caching, but the cache lookup logic was not updated to match the new tuple-based key format. This created a subtle inconsistency that only manifests in complex nested control flow.

---

## Verification Checklist

- [ ] Fix 1 applied: Cache lookup uses tuple key
- [ ] Fix 2 applied: Context synchronized during pre-materialization
- [ ] vfpsmax.c compiles without MLIR errors
- [ ] All existing e2e tests still pass
- [ ] No new SSA scope violations introduced
- [ ] Generated MLIR passes `mlir-opt --verify-diagnostics`

---

## References

- **File**: `python/megg/egraph/terms_to_func.py`
- **Related Issue**: SSA scope violation in nested control flow
- **Test Case**: `tests/e2e/pcl/vfpsmax/vfpsmax.c`
- **Error Log**: `tmp/megg_e2e_20251108_105634/optimized.mlir`

---

## Conclusion

This bug is a **cache coherency issue** caused by inconsistent key formats between cache storage and lookup. The fix is straightforward: ensure all cache operations use the same tuple-based key format and properly synchronize the `current_block_context` during pre-materialization.

**Impact**: Once fixed, complex nested control flow patterns (like those in vfpsmax) will compile correctly without SSA scope violations.
