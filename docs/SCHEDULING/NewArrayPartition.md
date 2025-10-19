# NewArrayPartition Pass Documentation

## Overview

The `NewArrayPartition` pass implements hardware array partitioning optimization for MLIR memref operations. It transforms a single memory array into multiple smaller memory banks that can be accessed in parallel, improving memory bandwidth and enabling loop pipelining in hardware synthesis.

**Location**: `lib/TOR/NewArrayPartition.cpp`

## Purpose

Array partitioning is a critical hardware optimization technique that:
- **Increases memory bandwidth** by creating parallel memory banks
- **Enables loop pipelining** by removing memory access bottlenecks
- **Supports data parallelism** in hardware accelerators
- **Reduces access conflicts** by distributing array elements across banks

## Partition Modes

### 1. Cyclic Partitioning (`cyclic = 1`)

Elements are distributed round-robin across banks.

```
Original array: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
Factor = 4, Cyclic = 1

Bank 0: [0, 4, 8, 12]
Bank 1: [1, 5, 9, 13]
Bank 2: [2, 6, 10, 14]
Bank 3: [3, 7, 11, 15]

Bank selection: bank = index % factor
```

**Use case**: Strided access patterns (e.g., accessing every Nth element)

### 2. Block Partitioning (`cyclic = 0`)

Elements are distributed in contiguous blocks.

```
Original array: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
Factor = 4, Cyclic = 0

Bank 0: [0, 1, 2, 3]
Bank 1: [4, 5, 6, 7]
Bank 2: [8, 9, 10, 11]
Bank 3: [12, 13, 14, 15]

Bank selection: bank = index / (size / factor)
```

**Use case**: Sequential access patterns (e.g., accessing contiguous blocks)

## Pragma Attributes

Arrays are marked for partitioning via MLIR attributes:

```mlir
memref.global @mem_a : memref<16xi32> {
  partition_dim_array = [0 : i32],      // Partition dimension 0
  partition_factor_array = [4 : i32],   // Split into 4 banks
  partition_cyclic_array = [1 : i32],   // Cyclic mode
  var_name = "mem_a"
}
```

### Attribute Meanings

- **`partition_dim_array`**: Which dimension(s) to partition (e.g., `[0]` for first dimension)
  - Value `-1` indicates fully partition (partition all dimensions)
- **`partition_factor_array`**: Number of banks to create (e.g., `[4]` = 4 banks)
  - Value `-1` means completely partition (1 element per bank)
- **`partition_cyclic_array`**: Distribution mode
  - `1` = cyclic (round-robin)
  - `0` = block (contiguous)

## Pass Workflow

### Phase 1: Analysis and Validation (lines 1108-1114)

```cpp
colorAndCheckCallGraph(ModuleOp moduleOp)
```

**Goal**: Build equivalence classes of memrefs that must be partitioned identically across function call boundaries.

#### Step 1.1: Build Call Graph (lines 1009-1022)

```cpp
getReverseGraph(moduleOp, reverseGraph)
```

Creates reverse call graph: `callee → set of callers`

#### Step 1.2: Group Related Arrays (lines 1073-1106)

```cpp
getIdValueMap(moduleOp, idValueMap)
```

Creates groups of memref values that:
- Are passed through function calls
- Must have compatible partitioning schemes
- Flow through caller-callee relationships

**Algorithm**:
1. Start with each `memref.alloca` or function argument with partition attributes
2. Traverse call graph forward (callee direction) via `traverseCalleeArg()`
3. Traverse call graph backward (caller direction) via `traverseCallerArg()`
4. Build transitive closure of related memrefs

#### Step 1.3: Validate Each Group (lines 974-1007)

```cpp
handleOneGroupValueSet(idValueSet)
```

For each equivalence class:

1. **Check access pattern validity** (lines 165-193)
   - All accesses must use `affine.load`/`affine.store`
   - Affine maps must have correct rank
   - No unsupported operations

2. **Extract partition configuration** (lines 960-972)
   - Merge partition attributes from all values in group
   - Handle conflicts (warning if incompatible)
   - Support fully-partitioned arrays

3. **Verify bank assignments are computable** (lines 195-213)
   ```cpp
   rankCanBePartition(arg, rank, bank_factor, cyclic)
   ```
   - Check if bank can be determined at compile time
   - Ensure affine expressions don't have symbolic variables
   - Verify constant folding succeeds

4. **Update or remove attributes**
   - Mark valid partitions
   - Remove invalid partition attributes
   - Set pragma status for reporting

### Phase 2: Global Memref Partitioning (lines 1131-1140)

```cpp
patterns.insert<GlobalOpPattern>()
```

**Target**: `memref.global` operations

#### Step 2.1: Create Partitioned Globals (lines 294-362)

```cpp
createNewArray(op, newArray, partition, memref, factorMap, ...)
```

**Recursive algorithm** to create new global arrays:

1. For each dimension:
   - If partitioned: create `factor` new arrays with reduced dimension size
   - If not partitioned: preserve dimension size

2. Handle uneven divisions:
   ```
   Original: memref<17xi32>, factor=4
   Bank 0: memref<5xi32>  (17 % 4 = 1, gets extra element)
   Bank 1: memref<4xi32>
   Bank 2: memref<4xi32>
   Bank 3: memref<4xi32>
   ```

3. Split initial values (lines 267-293):
   ```cpp
   getInitialValue(memref, rank, flattenedIndex, bank, targetBank, ...)
   ```
   - For `memref.global` with `dense<...>` initializer
   - Recursively distribute elements to correct banks
   - Preserve byte-level data layout

4. Create new global operations:
   ```cpp
   newGlobalOp = rewriter.create<memref::GlobalOp>(
     loc, newMemrefName, visibility, TypeAttr, initValue, ...);
   ```

#### Step 2.2: Rewrite Memory Accesses (lines 438-486)

```cpp
changeMemrefAndOperands(arg, memref, factorMap, cyclicMap, ...)
```

For each `affine.load` or `affine.store`:

1. **Calculate bank index** (lines 427-436):
   ```cpp
   calBankAndChangeOpAttr(loadOrStore, bank, rank, ...)
   ```
   - Compute which bank the access targets
   - Uses affine map analysis
   - Supports multi-dimensional indexing

2. **Transform index expression** (lines 380-392):
   ```cpp
   getDimExpr(map, rank, rewriter, factor, cyclic, rankShape)
   ```

   **Cyclic mode**:
   ```
   Original: array[i]
   Bank:     array_partitioned[bank][i / factor]
   Bank ID:  i % factor
   ```

   **Block mode**:
   ```
   Original: array[i]
   Bank:     array_partitioned[bank][i % (size/factor)]
   Bank ID:  i / (size/factor)
   ```

3. **Update affine maps** (lines 405-414):
   ```cpp
   changeAccessOpAttr(loadOrStore, exprs, rewriter)
   ```
   - Compose new index expression with original affine map
   - Replace dimension expression for partitioned rank
   - Preserve other dimensions unchanged

4. **Replace memref operand** (line 484):
   ```cpp
   part.op->setOperand(operandNum, newArray[part.bank])
   ```
   - Point to correct bank
   - Update all uses

#### Step 2.3: Update Function Calls (lines 467-481)

For `func.call` operations passing partitioned arrays:
- Replace single memref argument with N bank arguments
- Update operand list
- Create new call operation

### Phase 3: Function Argument Partitioning (lines 1141-1147)

```cpp
patterns.insert<FuncOpPattern>()
```

**Target**: `func.func` operations with partitioned arguments

1. **Process each memref argument** (lines 691-718):
   - Extract partition attributes (suffixed with `_argIndex`)
   - Create partition vector
   - Call `partitionFunc()` to split array

2. **Update function signature**:
   ```cpp
   funcOp.insertArgument(index, newMemref, ...)
   ```
   - Replace single argument with N bank arguments
   - Erase original argument
   - Remove partition attributes

3. **Propagate to callers**:
   - Callers were already updated in Phase 2.3
   - Function signature changes matched by call site updates

### Phase 4: Local Alloca Partitioning (lines 1148-1154)

```cpp
patterns.insert<AllocaOpPattern>()
```

**Target**: `memref.alloca` operations

Similar to global partitioning but simpler:
1. Create N new `memref.alloca` operations
2. Rewrite accesses to use appropriate bank
3. No need to handle initial values (stack allocations)

## Key Functions

### Bank Calculation (lines 114-130)

```cpp
int getMemBank(AffineMap map, int rank, MLIRContext *ctx,
               int factor, bool cyclic)
```

**Purpose**: Determine which bank an affine access targets at compile time.

**Algorithm**:
1. Extract affine expression for specified rank
2. Verify no symbolic variables (must be constant-foldable)
3. Apply partitioning formula:
   - Cyclic: `expr % factor`
   - Block: `expr.floorDiv(factor)`
4. Compose and constant-fold
5. Return bank index or `-1` if not compile-time constant

**Example**:
```mlir
// Original: affine.load %mem[%i * 2 + 1]
// Factor = 4, Cyclic mode

map = affine_map<(d0) -> (d0 * 2 + 1)>
expr = d0 * 2 + 1
bank_expr = (d0 * 2 + 1) % 4

// If d0 = 0: bank = 1
// If d0 = 1: bank = 3
// If d0 = 2: bank = 1
```

### New Bank Calculation (lines 258-265)

```cpp
int getNewBank(int bank, int index, int factor, bool cyclic, int rankShape)
```

Used during initial value distribution for multi-dimensional arrays.

**Formula**:
- Cyclic: `bank * factor + (index % bank_factor)`
- Block: `bank * factor + (index / bank_factor)`

Handles dimension nesting in recursive initial value splitting.

## Error Handling

### Warning Messages

1. **Non-standard affine access** (lines 38-42):
   ```
   warning: alloca variable @var_name with applying array_partition pragma
   is failed, because non standard affine access.
   ```
   - Access pattern cannot be analyzed
   - Non-affine operations used
   - Symbolic indices present

2. **Bank cannot be divided** (lines 77-82):
   ```
   warning: global variable @var_name with applying array_partition pragma
   on dim 0 is failed, because bank cannot be divided.
   ```
   - Bank assignment not compile-time constant
   - Complex affine expressions
   - Data-dependent indexing

### Pragma Status Tracking (lines 239, 245, 551, etc.)

```cpp
setPragmaStructureAttrStatusByValue(arg, lineString, status)
```

Marks which pragmas succeeded/failed for user reporting.

## Transformation Example

### Before Partitioning

```mlir
memref.global @array : memref<16xi32> {
  partition_dim_array = [0 : i32],
  partition_factor_array = [4 : i32],
  partition_cyclic_array = [1 : i32]
}

func.func @example() {
  %mem = memref.get_global @array : memref<16xi32>
  %c5 = arith.constant 5 : index
  %val = affine.load %mem[%c5] : memref<16xi32>
  // Bank = 5 % 4 = 1, Index = 5 / 4 = 1
  return
}
```

### After Partitioning

```mlir
memref.global @array_0 : memref<4xi32>  // Bank 0: [0, 4, 8, 12]
memref.global @array_1 : memref<4xi32>  // Bank 1: [1, 5, 9, 13]
memref.global @array_2 : memref<4xi32>  // Bank 2: [2, 6, 10, 14]
memref.global @array_3 : memref<4xi32>  // Bank 3: [3, 7, 11, 15]

func.func @example() {
  %mem_0 = memref.get_global @array_0 : memref<4xi32>
  %mem_1 = memref.get_global @array_1 : memref<4xi32>
  %mem_2 = memref.get_global @array_2 : memref<4xi32>
  %mem_3 = memref.get_global @array_3 : memref<4xi32>

  %c5 = arith.constant 5 : index
  // Access bank 1 (5 % 4 = 1), index 1 (5 / 4 = 1)
  %c1 = arith.constant 1 : index
  %val = affine.load %mem_1[%c1] : memref<4xi32>
  return
}
```

## Multi-Dimensional Partitioning

### Example: 2D Array

```mlir
// Original
memref.global @matrix : memref<8x16xi32> {
  partition_dim_array = [0 : i32, 1 : i32],
  partition_factor_array = [2 : i32, 4 : i32],
  partition_cyclic_array = [0 : i32, 1 : i32]
}

// After partitioning:
// - Dim 0: 2 banks (block mode)
// - Dim 1: 4 banks (cyclic mode)
// Total: 2 * 4 = 8 banks

memref.global @matrix_0 : memref<4x4xi32>  // rows [0-3], cols [0,4,8,12]
memref.global @matrix_1 : memref<4x4xi32>  // rows [0-3], cols [1,5,9,13]
// ... 6 more banks
```

### Bank Indexing Formula

```
bank = bank_dim0 * factor_dim1 + bank_dim1

Access: matrix[i][j]
  bank_dim0 = i / (size_dim0 / factor_dim0)  // Block mode
  bank_dim1 = j % factor_dim1                 // Cyclic mode
  bank = bank_dim0 * 4 + bank_dim1
```

## Limitations

1. **Affine-only**: Only works with affine.load/affine.store operations
2. **Compile-time indices**: Bank selection must be constant-foldable
3. **No symbolic variables**: Affine maps cannot contain symbols
4. **No recursion**: Recursive function calls are not supported
5. **Uniform partitioning**: All uses of an array must partition identically

## Integration with Other Passes

### Pipeline Position

Typically run after:
- Loop transformation passes (trip count, unrolling)
- Affine loop optimization

Typically run before:
- Lower-level memory optimizations
- Conversion to hardware dialects

### Related Passes

- **LoopTripcount**: Analyzes loop bounds for partition validation
- **HlsUnroll**: Loop unrolling creates parallel accesses benefiting from partitioning
- **MemrefReuse**: May analyze partitioned memory access patterns
- **DependenceAnalysis**: Checks if partition enables parallelism

## Performance Considerations

### When to Use Array Partition

**Good candidates**:
- Arrays accessed in tight loops
- Multiple accesses per loop iteration
- Strided or predictable access patterns
- Memory-bound computations

**Poor candidates**:
- Irregular access patterns
- Data-dependent indexing
- Small arrays (overhead > benefit)
- Already-parallel memory systems

### Factor Selection

- **Factor = 2, 4, 8**: Common hardware-friendly values
- **Factor = -1**: Fully partition (1 element per register)
- **Factor = loop_unroll_factor**: Match unroll factor for maximum parallelism

### Mode Selection

- **Cyclic**: Use when accessing with stride > 1
- **Block**: Use for sequential/burst access patterns
- **Mixed**: Different modes per dimension based on access patterns

## Debugging

### Enable Debug Output

```bash
# Build with debug enabled
cmake -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=Debug

# Run with debug flag
mlir-opt -debug-only=new-array-partition input.mlir
```

### Common Issues

1. **"Non-standard affine access"**
   - Check that all accesses use affine.load/store
   - Verify no pointer arithmetic or casts

2. **"Bank cannot be divided"**
   - Ensure index expressions are affine
   - Remove symbolic/non-constant indices
   - Simplify complex affine maps

3. **Incorrect bank assignment**
   - Verify affine map composition
   - Check factor and cyclic attribute values
   - Test with simple constant indices first

## References

- **MLIR Affine Dialect**: https://mlir.llvm.org/docs/Dialects/Affine/
- **HLS Array Partition**: Similar to Xilinx Vitis HLS `#pragma HLS array_partition`
- **Source Code**: `lib/TOR/NewArrayPartition.cpp`
- **Pass Definition**: `include/TOR/Passes.td` (NewArrayPartition)
