# MLIR-STD 更新说明

## 更新内容

`pixi run mlir-std` 现在默认包含以下 passes：

1. **`--aps-to-standard`** - 转换 APS dialect 到标准 MLIR
2. **`--comb-extract-to-arith-trunc`** - 转换 comb.extract 到 arith.trunci ⭐ **新增**
3. **`--canonicalize`** - 规范化优化
4. **`--arith-select-to-scf-if`** - 转换 arith.select 到 scf.if ⭐ **新增**

## 改进

### 之前

```bash
# 旧版本：包含 comb dialect 操作
pixi run mlir-std examples/deca/deca_decompress.cadl
```

输出包含：
```mlir
%3 = comb.extract %arg5 from 0 : (i32) -> i3
%8 = comb.extract %7 from 0 : (i8) -> i1
%11 = arith.select %8, %10, %c0_i8 : i8
```

### 现在

```bash
# 新版本：完全标准化的 MLIR（控制流显式化）
pixi run mlir-std examples/deca/deca_decompress.cadl
```

输出包含：
```mlir
%3 = arith.trunci %arg5 : i32 to i3
%8 = arith.trunci %7 : i8 to i1
%11 = scf.if %8 -> (i8) {
  scf.yield %10 : i8
} else {
  scf.yield %c0_i8 : i8
}
```

## 为什么这很重要？

### 1. 与 Polygeist 生成的代码兼容

Polygeist 从 C 代码生成的 MLIR 使用标准 arith dialect，不使用 CIRCT 的 comb dialect。统一表示方式后，更容易进行指令匹配。

**示例**：

```c
// C 代码
uint8_t bit_pos = idx & 0x7;  // 提取低 3 位
```

Polygeist 生成：
```mlir
%bit_pos = arith.trunci %idx : i32 to i8
```

现在 CADL 也生成相同的模式！

### 2. 适合 megg 端到端测试

```bash
# 从 CADL 生成标准化 MLIR
pixi run mlir-std examples/deca/deca_decompress.cadl > cadl_pattern.mlir

# 从 C 代码生成标准化 MLIR（使用 polygeist）
polygeist-opt deca_decompress.c -o c_pattern.mlir

# 使用 megg 进行指令匹配
./megg-opt.py c_pattern.mlir \
  --custom-instructions cadl_pattern.mlir \
  -o optimized.mlir
```

### 3. 移除 CIRCT 依赖

生成的 MLIR 只依赖标准 dialect：
- ✅ `arith` - 算术操作
- ✅ `memref` - 内存操作
- ✅ `scf` - 结构化控制流
- ✅ `func` - 函数定义
- ❌ `comb` - **不再使用**

## 技术细节

### Pass Pipeline

```
CADL Input
    ↓
CADL Frontend (mlir_converter.py)
    ↓
APS Dialect MLIR (with comb.extract)
    ↓
--aps-to-standard
    ↓
Standard Dialect MLIR (still with comb.extract)
    ↓
--comb-extract-to-arith-trunc  ⭐ 新增
    ↓
Standard Dialect MLIR (arith.trunci)
    ↓
--canonicalize
    ↓
Optimized Standard MLIR
```

### 类型问题修复

之前的 APSToStandard 强制要求所有 memref 使用相同的 element type，但这在 DECA 中不现实：

```mlir
// 不同的 memref 有不同的类型
%bitmask: memref<4xi8>           // i8
%values: memref<32xi8>           // i8
%output: memref<32xi16>          // i16 ← 不同！
```

**解决方案**：移除类型一致性检查，因为 CPU 内存本来就是字节寻址的。

## 使用示例

### 基本使用

```bash
# 生成标准化 MLIR
pixi run mlir-std examples/deca/deca_decompress.cadl

# 保存到文件
pixi run mlir-std examples/deca/deca_decompress.cadl > output.mlir

# 验证没有 comb 操作
pixi run mlir-std examples/deca/deca_decompress.cadl | grep "comb\."
# (应该没有输出)
```

### 与 C 代码对比

```bash
# 1. 从 CADL 生成
pixi run mlir-std examples/deca/deca_decompress.cadl > from_cadl.mlir

# 2. 从 C 代码生成（假设有 polygeist）
polygeist-opt examples/deca/deca_decompress_simple.c \
  -function=deca_decompress_fused \
  -o from_c.mlir

# 3. 比对结构
diff -u from_cadl.mlir from_c.mlir
```

## 相关文档

- [CombExtractToArithTrunc Pass 文档](../../docs/COMB_EXTRACT_TO_ARITH_TRUNC.md)
- [C Reference Implementation](C_REFERENCE_README.md)
- [E2E Testing Guide](E2E_TESTING_GUIDE.md)

## 总结

通过添加 `--comb-extract-to-arith-trunc` pass 到默认 pipeline，`pixi run mlir-std` 现在生成完全标准化的 MLIR，使得：

1. ✅ 与 polygeist 生成的 C 代码兼容
2. ✅ 适合 megg 指令匹配
3. ✅ 移除 CIRCT 依赖
4. ✅ 更容易被标准 MLIR 工具处理

这为端到端的指令匹配测试铺平了道路！🚀
