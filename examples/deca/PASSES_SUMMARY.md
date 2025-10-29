# MLIR Standardization Passes 总结

## 概述

`pixi run mlir-std` 现在应用完整的标准化 pipeline，生成与 C 代码高度兼容的 MLIR。

## Pass Pipeline

```
CADL Input
    ↓
CADL Frontend (mlir_converter.py)
    ↓
APS Dialect MLIR
    │ • 包含: aps.readrf, aps.memburstload, comb.extract, arith.select
    ↓
--aps-to-standard
    │ • 转换: APS dialect → Standard dialects
    │ • 移除: 硬件特定的寄存器文件和 burst 操作
    ↓
Standard Dialect MLIR (with comb + select)
    │ • 包含: comb.extract, arith.select
    ↓
--comb-extract-to-arith-trunc
    │ • 转换: comb.extract → arith.trunci (+ arith.shrui)
    │ • 移除: CIRCT comb dialect 依赖
    ↓
Standard Dialect MLIR (with select)
    │ • 包含: arith.trunci, arith.select
    ↓
--canonicalize
    │ • 优化: 常量折叠、死代码消除、简化
    ↓
Optimized MLIR (with select)
    │ • 包含: arith.select (更简洁)
    ↓
--arith-select-to-scf-if
    │ • 转换: arith.select → scf.if
    │ • 显式化: 控制流
    ↓
Final Standard MLIR
    │ • 完全标准化，显式控制流
    │ • 只依赖: arith, memref, scf, func
```

## 三个核心 Pass

### 1. CombExtractToArithTrunc

**作用**: 将位提取操作标准化

**转换**:
```mlir
// 之前
%bit_pos = comb.extract %idx from 0 : (i32) -> i3

// 之后
%bit_pos = arith.trunci %idx : i32 to i3
```

**详细文档**: [COMB_EXTRACT_TO_ARITH_TRUNC.md](../../docs/COMB_EXTRACT_TO_ARITH_TRUNC.md)

### 2. ArithSelectToSCFIf

**作用**: 将条件选择转换为显式控制流

**转换**:
```mlir
// 之前
%val = arith.select %cond, %true_val, %false_val : i8

// 之后
%val = scf.if %cond -> (i8) {
  scf.yield %true_val : i8
} else {
  scf.yield %false_val : i8
}
```

**详细文档**: [ARITH_SELECT_TO_SCF_IF.md](../../docs/ARITH_SELECT_TO_SCF_IF.md)

### 3. APSToStandard

**作用**: 转换 APS 硬件方言到标准方言

**转换**:
- `aps.readrf` → 函数参数
- `aps.writerf` → 函数返回值
- `aps.memburstload/store` → 移除（已转换为 memref 参数）
- `memref.get_global` → 函数参数

## 完整示例：DECA 位提取

### CADL 源代码

```cadl
let byte_idx: u32 = idx / 8;
let bit_pos: u8 = idx[2:0];
let mask_byte: u8 = bitmask[byte_idx];
let bit_shifted: u8 = mask_byte >> bit_pos;
let is_nonzero: u1 = bit_shifted[0:0];

let sparse_val: i8 = if is_nonzero { values[vidx] } else { 0 };
```

### Pipeline 各阶段输出

#### Stage 1: CADL Frontend 生成

```mlir
%byte_idx = arith.divui %idx, %c8_i32 : i32
%bit_pos = comb.extract %idx from 0 : (i32) -> i3
%mask_byte = aps.memload %bitmask[%byte_idx] : memref<4xi8>, i32 -> i8
%bit_shifted = arith.shrui %mask_byte, %bit_pos_ext : i8
%is_nonzero = comb.extract %bit_shifted from 0 : (i8) -> i1
%sparse_val = arith.select %is_nonzero, %values_load, %c0_i8 : i8
```

#### Stage 2: 应用 --aps-to-standard

```mlir
%byte_idx = arith.divui %idx, %c8_i32 : i32
%bit_pos = comb.extract %idx from 0 : (i32) -> i3
%byte_idx_cast = arith.index_cast %byte_idx : i32 to index
%mask_byte = memref.load %arg0[%byte_idx_cast] : memref<4xi8>
%bit_shifted = arith.shrui %mask_byte, %bit_pos_ext : i8
%is_nonzero = comb.extract %bit_shifted from 0 : (i8) -> i1
%sparse_val = arith.select %is_nonzero, %values_load, %c0_i8 : i8
```

#### Stage 3: 应用 --comb-extract-to-arith-trunc

```mlir
%byte_idx = arith.divui %idx, %c8_i32 : i32
%bit_pos = arith.trunci %idx : i32 to i3  // ← 转换
%byte_idx_cast = arith.index_cast %byte_idx : i32 to index
%mask_byte = memref.load %arg0[%byte_idx_cast] : memref<4xi8>
%bit_pos_ext = arith.extui %bit_pos : i3 to i8
%bit_shifted = arith.shrui %mask_byte, %bit_pos_ext : i8
%is_nonzero = arith.trunci %bit_shifted : i8 to i1  // ← 转换
%sparse_val = arith.select %is_nonzero, %values_load, %c0_i8 : i8
```

#### Stage 4: 应用 --canonicalize

```mlir
// 常量折叠、简化（基本不变）
%byte_idx = arith.divui %idx, %c8_i32 : i32
%bit_pos = arith.trunci %idx : i32 to i3
%byte_idx_cast = arith.index_cast %byte_idx : i32 to index
%mask_byte = memref.load %arg0[%byte_idx_cast] : memref<4xi8>
%bit_pos_ext = arith.extui %bit_pos : i3 to i8
%bit_shifted = arith.shrui %mask_byte, %bit_pos_ext : i8
%is_nonzero = arith.trunci %bit_shifted : i8 to i1
%sparse_val = arith.select %is_nonzero, %values_load, %c0_i8 : i8
```

#### Stage 5: 应用 --arith-select-to-scf-if（最终）

```mlir
%byte_idx = arith.divui %idx, %c8_i32 : i32
%bit_pos = arith.trunci %idx : i32 to i3
%byte_idx_cast = arith.index_cast %byte_idx : i32 to index
%mask_byte = memref.load %arg0[%byte_idx_cast] : memref<4xi8>
%bit_pos_ext = arith.extui %bit_pos : i3 to i8
%bit_shifted = arith.shrui %mask_byte, %bit_pos_ext : i8
%is_nonzero = arith.trunci %bit_shifted : i8 to i1
%sparse_val = scf.if %is_nonzero -> (i8) {  // ← 转换
  scf.yield %values_load : i8
} else {
  scf.yield %c0_i8 : i8
}
```

## 与 C 代码的对应

### C 代码

```c
uint32_t byte_idx = idx / 8;
uint8_t bit_pos = idx & 0x7;
uint8_t mask_byte = bitmask[byte_idx];
uint8_t bit_shifted = mask_byte >> bit_pos;
uint8_t is_nonzero = bit_shifted & 0x1;

int8_t sparse_val;
if (is_nonzero) {
    sparse_val = values[vidx];
} else {
    sparse_val = 0;
}
```

### Polygeist 生成的 MLIR

```mlir
%byte_idx = arith.divui %idx, %c8 : i32
%bit_pos = arith.andi %idx, %c7 : i32
%bit_pos_trunc = arith.trunci %bit_pos : i32 to i8
%mask_byte = memref.load %bitmask[%byte_idx] : memref<4xi8>
%bit_shifted = arith.shrui %mask_byte, %bit_pos_trunc : i8
%is_nonzero = arith.andi %bit_shifted, %c1 : i8
%is_nonzero_i1 = arith.trunci %is_nonzero : i8 to i1
%sparse_val = scf.if %is_nonzero_i1 -> (i8) {
  %val = memref.load %values[%vidx] : memref<32xi8>
  scf.yield %val : i8
} else {
  scf.yield %c0 : i8
}
```

### 对比

| 特性 | CADL + mlir-std | Polygeist (C) |
|------|----------------|---------------|
| 位提取 | `arith.trunci` | `arith.trunci` ✓ |
| 条件选择 | `scf.if` | `scf.if` ✓ |
| 内存访问 | `memref.load` | `memref.load` ✓ |
| 控制流 | 显式 `scf.if` | 显式 `scf.if` ✓ |

**高度一致！** 便于 megg 指令匹配！

## 使用方法

### 基本使用

```bash
# 直接使用（应用所有 passes）
pixi run mlir-std examples/deca/deca_decompress.cadl
```

### 验证转换

```bash
# 验证没有 comb 操作
pixi run mlir-std examples/deca/deca_decompress.cadl | grep "comb\."
# (应该没有输出)

# 验证没有 arith.select
pixi run mlir-std examples/deca/deca_decompress.cadl | grep "arith.select"
# (应该没有输出)

# 验证有 scf.if
pixi run mlir-std examples/deca/deca_decompress.cadl | grep "scf.if"
# (应该有输出)
```

### 用于 megg 测试

```bash
# 1. 从 CADL 生成标准化 MLIR
pixi run mlir-std examples/deca/deca_decompress.cadl > cadl_pattern.mlir

# 2. 从 C 代码生成 MLIR（假设有 polygeist）
polygeist-opt examples/deca/deca_decompress_simple.c \
  -function=deca_decompress_fused \
  -o c_pattern.mlir

# 3. 使用 megg 进行指令匹配
cd /home/cloud/megg
./megg-opt.py c_pattern.mlir \
  --custom-instructions cadl_pattern.mlir \
  -o optimized.mlir
```

## Pass 顺序的重要性

⚠️ **注意**: Pass 的应用顺序很重要！

### 当前顺序（正确）

```
--aps-to-standard
--comb-extract-to-arith-trunc
--canonicalize
--arith-select-to-scf-if  ← 必须在 canonicalize 之后
```

### 为什么 canonicalize 在中间？

- **在 select-to-if 之前**: 优化简化 IR
- **在 select-to-if 之后**: 会将简单的 `scf.if` 转回 `arith.select`！

**示例**:
```mlir
// arith-select-to-scf-if 转换
%val = scf.if %cond -> (i32) {
  scf.yield %a : i32
} else {
  scf.yield %b : i32
}

// canonicalize 会转回
%val = arith.select %cond, %a, %b : i32
```

因此，`--arith-select-to-scf-if` **必须是最后一个 pass**！

## 依赖的 Dialect

最终输出只依赖这些标准 dialect：

- ✅ `arith` - 算术操作
- ✅ `memref` - 内存操作
- ✅ `scf` - 结构化控制流
- ✅ `func` - 函数定义
- ❌ `comb` - **已移除**
- ❌ `aps` - **已移除**

## 总结

通过三个自定义 pass 的组合，`pixi run mlir-std` 现在生成：

1. ✅ **完全标准化的 MLIR** - 无 CIRCT/APS 依赖
2. ✅ **显式控制流** - 使用 `scf.if` 而非 `arith.select`
3. ✅ **与 C 代码兼容** - 匹配 polygeist 生成的结构
4. ✅ **适合指令匹配** - 便于 megg 端到端测试

这为 CADL → MLIR → C 代码的双向验证和优化提供了坚实的基础！🚀
