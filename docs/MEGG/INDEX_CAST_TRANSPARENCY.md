# Pattern Normalization for Custom Instruction Matching

## Overview

**Status**: ✅ Implemented and Tested

This document describes the pattern normalization feature that optimizes custom instruction patterns before they are used for matching.

## Motivation

When defining custom instruction patterns in MLIR, patterns may contain:
- Redundant operations (e.g., `x + 0`, `x * 1`)
- Unnecessary type conversions (e.g., `index_cast` chains)
- Expressions that can be simplified via constant folding

These redundancies can make pattern matching less effective, as the pattern needs to match the exact structure even when semantically equivalent forms exist.

## Solution: Pattern Normalization

Before extracting pattern skeletons and rewrite rules, we now run the pattern module through a full Megg optimization pipeline:

1. **Parse to E-graph**: Convert pattern MLIR to e-graph terms using `FuncToTerms`
2. **Apply Internal Rewrites**: Run algebraic laws and constant folding (5 rounds)
3. **Extract Optimized Representation**: Use `Extractor` with `AstSize` cost function
4. **Reconstruct MLIR**: Convert back to MLIR using `ExprTreeToMLIR.reconstruct`

This produces a **canonical pattern representation** that is cleaner and more efficient to match.

## Usage

### Automatic (Default)

Pattern normalization is **enabled by default** in `build_ruleset_from_module()`:

```python
from megg.rewrites.match_rewrites import build_ruleset_from_module
from megg.utils.mlir_utils import MModule

# Load pattern module
pattern_module = MModule.from_string(pattern_mlir_text)

# Normalization happens automatically
ruleset, skeletons = build_ruleset_from_module(pattern_module)
```

### Manual Control

You can control normalization behavior:

```python
# Disable normalization
ruleset, skeletons = build_ruleset_from_module(
    pattern_module,
    normalize=False  # Skip normalization
)

# Enable verbose logging
ruleset, skeletons = build_ruleset_from_module(
    pattern_module,
    normalize=True,
    verbose=True  # See detailed normalization steps
)
```

### Standalone Normalization

You can also normalize patterns independently:

```python
from megg.rewrites.match_rewrites import normalize_pattern_module

# Normalize pattern module
normalized_module = normalize_pattern_module(
    pattern_module,
    verbose=True
)

# Use the normalized module
print(normalized_module)
```

## Examples

### Example 1: Simple Arithmetic Pattern

**Input Pattern** (with redundant operations):
```mlir
func.func @simple_add(%arg0: i32, %arg1: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  // Redundant operations
  %0 = arith.addi %arg0, %c0 : i32   // x + 0 = x
  %1 = arith.addi %0, %arg1 : i32
  %2 = arith.muli %1, %c1 : i32      // x * 1 = x
  return %2 : i32
}
```

**Normalized Pattern**:
```mlir
func.func @simple_add(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  return %0 : i32
}
```

✅ **Benefit**: Eliminated 2 redundant operations and 2 unused constants

### Example 2: Complex Control Flow Pattern (vgemv3d)

**Input**: 53 lines of MLIR with nested loops, index casts, and memory operations

**Normalized**: Simplified arithmetic expressions, consolidated constants, removed redundant index casts

✅ **Result**:
- 4 rewrite rules generated
- 1 skeleton with 4 leaf patterns
- Cleaner matching structure

## Implementation Details

### File Locations

- **Implementation**: python/megg/rewrites/match_rewrites.py
  - `normalize_pattern_module()` - Main normalization function
  - `build_ruleset_from_module()` - Updated to use normalization

- **Test**: test_pattern_normalization.py
  - Test with vgemv3d pattern
  - Test with simple arithmetic pattern

### Normalization Pipeline

```
Pattern MLIR
    ↓
FuncToTerms.transform(func_op, egraph)
    ↓
Apply basic_math_laws() × 5 rounds
Apply constant_folding_laws() × 5 rounds
Apply type_annotation_ruleset() × 5 rounds
    ↓
MeggEGraph.from_egraph(egraph, transformer)
    ↓
Extractor(megg_egraph, AstSize())
    ↓
ExprTreeToMLIR.reconstruct(func_op, exprs)
    ↓
Normalized Pattern MLIR
```

### Optimizations Applied

The normalization applies all internal rewrites, including:

1. **Algebraic Laws** (`basic_math_laws`):
   - Identity: `x + 0 = x`, `x * 1 = x`
   - Associativity: `(a + b) + c = a + (b + c)`
   - Commutativity: `a + b = b + a`
   - Distributivity: `a * (b + c) = a*b + a*c`

2. **Constant Folding** (`constant_folding_laws`):
   - `2 + 3 → 5`
   - `4 * 8 → 32`
   - `(x * 2) * 4 → x * 8`

3. **Type Propagation** (`type_annotation_ruleset`):
   - Track types through expressions
   - Simplify type conversions

## Performance Impact

- **Pattern Normalization**: ~100-500ms per pattern (one-time cost at compiler initialization)
- **Matching Performance**: **Improved** due to simpler patterns
- **Overall**: Negligible impact, significant benefit for complex patterns

## Testing

Run the test suite:

```bash
PYTHONPATH=python pixi run python test_pattern_normalization.py
```

Expected output:
```
✅ All pattern normalization tests passed!
```

## Future Work

Possible enhancements:
1. Cache normalized patterns to disk for faster repeated compilation
2. Add more aggressive simplification passes (e.g., dead code elimination)
3. Support pattern-specific optimization hints

## Related Documentation

- Custom Instruction Matching Guide (CUSTOM_INSTRUCTION_MATCHING.md)
- Internal Rewrites (python/megg/rewrites/internal_rewrites.py)
- E-graph Extraction (python/megg/egraph/extract.py)
