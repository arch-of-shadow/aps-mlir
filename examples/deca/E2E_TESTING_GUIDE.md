# DECA 端到端测试指南

本文档说明如何使用 megg 进行 DECA 解压缩的端到端指令匹配测试。

## 概述

我们有两套实现：

1. **CADL 实现** (`deca_decompress.cadl`): 手写的自定义指令规范
2. **C 参考实现** (`deca_decompress_simple.c`): 用于 polygeist 转换的 C 代码

测试目标：验证 megg 能否从 C 代码生成的 MLIR 中识别并匹配 CADL 定义的自定义指令模式。

## 文件清单

### APS-MLIR 目录 (`/home/cloud/aps-mlir/examples/deca/`)

| 文件 | 说明 |
|------|------|
| `deca_decompress.cadl` | CADL 指令定义 (32-element tile) |
| `deca_decompress_simple.c` | C 参考实现 (简化版) ⭐ |
| `deca_decompress.c` | C 参考实现 (完整版，含内存) |
| `test_c_reference.c` | C 测试程序 |
| `test_c_reference` | 编译后的测试程序 |
| `C_REFERENCE_README.md` | C 代码使用说明 |
| `E2E_TESTING_GUIDE.md` | 本文档 |

### Megg 目录 (`/home/cloud/megg/tests/benchmarks/deca/`)

| 文件 | 说明 |
|------|------|
| `deca_decmp.cadl` | Megg 版本 (256-element tile) |
| `deca_decmp.c` | Megg 版本 C 实现 |
| `deca_decmp.h` | Megg 版本头文件 |

## 测试流程

### Step 1: 验证 C 代码正确性

```bash
cd /home/cloud/aps-mlir/examples/deca

# 编译测试
gcc -Wall -Wextra -O2 test_c_reference.c -o test_c_reference

# 运行测试
./test_c_reference
```

**预期结果**: 所有测试通过 ✓

### Step 2: 从 CADL 生成 MLIR 指令模式

```bash
cd /home/cloud/aps-mlir

# 生成标准 MLIR (用于指令匹配)
bash scripts/mlir-std.sh examples/deca/deca_decompress.cadl > /tmp/deca_pattern.mlir

# 查看生成的 MLIR
cat /tmp/deca_pattern.mlir
```

**预期输出**: 包含 `func.func @flow_deca_decompress_u1` 的 MLIR 代码

### Step 3: 用 Polygeist 转换 C 代码

```bash
# 注意: 需要先安装 polygeist
# TODO: 添加 polygeist 安装说明

cd /home/cloud/aps-mlir/examples/deca

# 转换 C 代码到 MLIR
polygeist-opt deca_decompress_simple.c \
  -function=deca_decompress_fused \
  -o /tmp/deca_from_c.mlir

# 查看生成的 MLIR
cat /tmp/deca_from_c.mlir
```

### Step 4: 用 Megg 进行指令匹配

```bash
cd /home/cloud/megg

# 运行 megg 优化器
./megg-opt.py /tmp/deca_from_c.mlir \
  --custom-instructions /tmp/deca_pattern.mlir \
  --verbose \
  -o /tmp/deca_optimized.mlir

# 查看优化后的结果
cat /tmp/deca_optimized.mlir
```

**期望结果**:
- megg 识别出 C 代码中的解压缩模式
- 将循环体替换为 `aps.custom_instr "deca_decompress_u1"` 调用
- 输出显示匹配成功的信息

### Step 5: 验证匹配结果

```bash
# 比较优化前后的 MLIR
diff -u /tmp/deca_from_c.mlir /tmp/deca_optimized.mlir

# 检查是否包含自定义指令调用
grep -A 5 "aps.custom_instr" /tmp/deca_optimized.mlir
```

## 关键匹配模式

Megg 需要识别的核心计算模式：

### 模式 1: 位提取

```c
// C 代码
uint32_t byte_idx = idx >> 3;
uint8_t bit_pos = idx & 0x7;
uint8_t is_nonzero = (bitmask[byte_idx] >> bit_pos) & 0x1;
```

```mlir
// 预期 MLIR
%byte_idx = arith.shrui %idx, %c3
%bit_pos = arith.andi %idx, %c7
%mask_byte = memref.load %bitmask[%byte_idx]
%shifted = arith.shrui %mask_byte, %bit_pos
%is_nonzero = arith.andi %shifted, %c1
```

### 模式 2: Q8.8 反量化

```c
// C 代码
int32_t mul_result = (int32_t)val * (int32_t)scale;
int16_t dequant = (int16_t)(mul_result >> 8);
```

```mlir
// 预期 MLIR
%val_ext = arith.extsi %val : i8 to i32
%scale_ext = arith.extsi %scale : i16 to i32
%mul = arith.muli %val_ext, %scale_ext : i32
%shifted = arith.shrsi %mul, %c8 : i32
%result = arith.trunci %shifted : i32 to i16
```

### 模式 3: 稀疏索引更新

```c
// C 代码
vidx += is_nonzero;
```

```mlir
// 预期 MLIR
%vidx_inc = arith.select %is_nonzero, %c1, %c0 : i32
%vidx_next = arith.addi %vidx, %vidx_inc : i32
```

### 模式 4: 完整循环结构

```c
// C 代码
for (uint32_t idx = 0; idx < 32; idx++) {
    uint8_t is_nonzero = extract_bit(bitmask, idx);
    int8_t val = is_nonzero ? sparse_values[vidx] : 0;
    output[idx] = dequantize_q88(val, scale);
    vidx += is_nonzero;
}
```

```mlir
// 预期 MLIR
scf.for %idx = %c0 to %c32 step %c1 iter_args(%vidx = %c0) -> (i32) {
    // ... 位提取 + 条件读取 + Q8.8 乘法 ...
    scf.yield %vidx_next : i32
}
```

## 调试技巧

### 1. 逐步简化 C 代码

如果 megg 无法匹配完整模式，尝试：

```c
// 最简版本：只保留核心计算
uint32_t minimal_pattern(const uint8_t* bitmask, int16_t scale) {
    uint32_t idx = 0;
    uint32_t byte_idx = idx >> 3;
    uint8_t bit_pos = idx & 0x7;
    uint8_t is_nonzero = (bitmask[byte_idx] >> bit_pos) & 0x1;
    return is_nonzero;
}
```

### 2. 查看 megg 的 skeleton 提取

```bash
./megg-opt.py /tmp/deca_from_c.mlir \
  --debug-skeleton \
  --verbose \
  2>&1 | less
```

### 3. 手动验证 MLIR 结构

```bash
# 使用 aps-opt 查看 MLIR 的 IR 结构
./build/tools/aps-opt/aps-opt /tmp/deca_from_c.mlir \
  --mlir-print-ir-tree
```

### 4. 比对 megg 的 DECA 示例

```bash
# 查看 megg 是如何处理 256-element 版本的
cd /home/cloud/megg/tests/benchmarks/deca
cat deca_decmp.cadl
cat deca_decmp.c

# 对比两个版本的差异
diff -u /home/cloud/aps-mlir/examples/deca/deca_decompress_simple.c \
     /home/cloud/megg/tests/benchmarks/deca/deca_decmp.c
```

## 常见问题

### Q1: polygeist 找不到？

需要安装 polygeist:

```bash
# TODO: 添加安装步骤
# 或者使用 megg 内置的 polygeist (如果有)
```

### Q2: megg 无法识别模式？

可能原因：
1. C 代码的计算模式与 CADL 不完全匹配
2. Polygeist 生成的 MLIR 结构与预期不同
3. Megg 的 cost function 没有正确配置

解决方案：
- 使用 `--verbose` 查看详细匹配日志
- 简化 C 代码，逐步增加复杂度
- 参考 megg 的其他成功案例 (mac, rotl32 等)

### Q3: CADL 和 C 的语义差异？

主要差异：

| CADL | C | 说明 |
|------|---|------|
| `static` 数组 | 局部数组 | CADL 是全局 scratchpad |
| `_burst_read` | `memcpy` | CADL 是硬件 burst |
| `with...do` 循环 | `for` 循环 | CADL 是特殊语法 |
| `let` 绑定 | 变量声明 | CADL 是 SSA |

优化建议：
- C 代码尽量使用局部数组（更接近 CADL 的 scratchpad）
- 避免使用指针间接访问（CADL 是直接访问）
- 尽量使用固定长度循环（CADL 的 `while i < CONST`）

## 性能对比

### 理论性能 (32-element tile)

| 版本 | 内存访问 | 计算复杂度 | 硬件资源 |
|------|----------|------------|----------|
| 纯 CPU | 36 + 32 + 64 = 132 次 | O(32) | 通用 ALU |
| CADL ASIP | 3 次 burst (12 次总线) | O(32) 流水线 | 专用单元 |

**加速比估算**:
- 内存: 132 / 12 = 11x
- 计算: 1-2x (流水线)
- **总计**: ~10-20x

### 实测性能 (megg 优化)

TODO: 运行实测后填写

## 下一步

1. ✓ 创建 C 参考实现
2. ✓ 验证 C 代码正确性
3. ⬜ 安装 polygeist
4. ⬜ 转换 C 代码到 MLIR
5. ⬜ 运行 megg 指令匹配
6. ⬜ 分析匹配结果
7. ⬜ 优化匹配成功率
8. ⬜ 性能评估

## 参考资料

- Megg 文档: `/home/cloud/megg/docs/complex_instruction_matching.md`
- Megg DECA 实现: `/home/cloud/megg/tests/benchmarks/deca/`
- APS-MLIR DECA 实现: `/home/cloud/aps-mlir/examples/deca/`
- CADL 语法: `/home/cloud/aps-mlir/CLAUDE.md`
- Polygeist 文档: (TODO)

## 联系信息

如有问题，请查看：
- APS-MLIR 项目 README
- Megg 项目 CLAUDE.md
- 相关论文和文档
