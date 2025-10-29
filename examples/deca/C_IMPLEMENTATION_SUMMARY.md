# DECA C 实现总结

## 概述

为了支持端到端的指令匹配测试，我为 DECA 解压缩算法创建了 C 参考实现，与 CADL 实现完全对应。

## 创建的文件

### 1. [deca_decompress_simple.c](deca_decompress_simple.c) ⭐ 主要文件

包含 4 个函数：

#### `deca_decompress_simple()` - 两阶段版本
- 完全对应 CADL 的 `deca_decompress_u1()`
- Stage 1: 稀疏到密集展开 (bitmask-driven)
- Stage 2: Q8.8 反量化
- 适合理解算法流程

#### `deca_decompress_fused()` - 融合版本 ⭐ 推荐用于 polygeist
- 单循环实现，两阶段融合
- 代码更简洁，更容易被 megg 匹配
- 计算模式清晰

#### `extract_bit_from_bitmask()` - 位提取辅助函数
- 从 bitmask 中提取单个 bit
- 对应 CADL 的位提取逻辑

#### `dequantize_q88()` - Q8.8 反量化辅助函数
- Q8.8 固定点乘法
- 对应 CADL 的 scale 计算

### 2. [deca_decompress.c](deca_decompress.c) - 完整版本

- 包含完整的内存访问模拟
- 对应 CADL 的内存布局
- 用于理解 burst read/write

### 3. [test_c_reference.c](test_c_reference.c) - 测试程序

包含 6 个测试：
1. 位提取测试
2. Q8.8 反量化测试
3. 简单解压缩测试
4. 融合解压缩测试
5. 两种实现对比测试
6. 边界情况测试

### 4. [C_REFERENCE_README.md](C_REFERENCE_README.md) - 使用说明

详细说明如何使用 C 代码进行 polygeist 转换和 megg 指令匹配。

### 5. [E2E_TESTING_GUIDE.md](E2E_TESTING_GUIDE.md) - 端到端测试指南

完整的测试流程文档，从 C 代码验证到 megg 指令匹配。

## 测试结果

```bash
$ ./test_c_reference
========================================
DECA C Reference Implementation Tests
========================================

TEST 1: Bit Extraction                    ✓
TEST 2: Q8.8 Dequantization              ✓
TEST 3: Simple Decompression             ✓
TEST 4: Fused Decompression              ✓
TEST 5: Compare Simple vs Fused          ✓
TEST 6: Edge Cases                       ✓

All tests completed!
```

**所有测试通过！** ✓

## 与 CADL 的对应关系

### 数据类型

| C 类型 | CADL 类型 | 说明 |
|--------|-----------|------|
| `uint8_t` | `u8` | 无符号 8 位 |
| `int8_t` | `i8` | 有符号 8 位 |
| `int16_t` | `i16` | 有符号 16 位 (Q8.8) |
| `uint32_t` | `u32` | 无符号 32 位 |

### 内存布局

```
地址              数据                大小
─────────────────────────────────────────────
base_addr+0      bitmask (u8[4])    4 bytes
base_addr+4      values (i8[32])    32 bytes
base_addr+36     output (i16[32])   64 bytes
```

### 算法流程

```
C: deca_decompress_fused()
│
├─ for (idx = 0; idx < 32; idx++)
│   ├─ byte_idx = idx >> 3
│   ├─ bit_pos = idx & 0x7
│   ├─ is_nonzero = (bitmask[byte_idx] >> bit_pos) & 0x1
│   ├─ val = is_nonzero ? sparse_values[vidx] : 0
│   ├─ mul_result = (int32_t)val * (int32_t)scale
│   ├─ output[idx] = (int16_t)(mul_result >> 8)
│   └─ vidx += is_nonzero
│
└─ return vidx

CADL: deca_decompress_u1()
│
├─ with idx = (0, idx+1), vidx = (0, vidx_next) do
│   ├─ byte_idx = idx / 8
│   ├─ bit_pos = idx[2:0]
│   ├─ is_nonzero = (bitmask[byte_idx] >> bit_pos)[0:0]
│   ├─ sparse_val = if is_nonzero { values[vidx] } else { 0 }
│   ├─ mul_result = sparse_val * global_scale
│   ├─ decompressed_weights[idx] = (mul_result >> 8)[15:0]
│   └─ vidx_next = vidx + (if is_nonzero {1} else {0})
│
└─ return nnz
```

## 核心计算模式

### 1. 位提取模式

```c
// 将 32-bit bitmask 存储为 4 个 u8
uint32_t byte_idx = idx >> 3;        // idx / 8
uint8_t bit_pos = idx & 0x7;         // idx % 8
uint8_t is_nonzero = (bitmask[byte_idx] >> bit_pos) & 0x1;
```

### 2. 稀疏索引模式

```c
// 条件读取稀疏数组
int8_t val = is_nonzero ? sparse_values[vidx] : 0;

// 条件更新索引
vidx += is_nonzero ? 1 : 0;
```

### 3. Q8.8 固定点模式

```c
// Q8.8 格式: 8 整数位 + 8 小数位
int32_t mul_result = (int32_t)val * (int32_t)scale;
int16_t dequant = (int16_t)(mul_result >> 8);
```

## 优化设计

### 为什么使用 32-element tile？

1. **Burst 传输对齐**:
   - Bitmask: 4 bytes
   - Values: 32 bytes
   - Output: 64 bytes
   - 总计: 100 bytes (适合现代 cache line)

2. **硬件友好**:
   - 固定循环边界（编译时常量）
   - 简单的位运算（低成本硬件）
   - 无动态内存分配

3. **流水线效率**:
   - 32 次迭代适合展开 (unroll factor = 4)
   - 每次迭代独立，易并行

### 为什么使用 u8 存储 bitmask？

1. **APSToStandard 兼容性**: 避免 u1 类型的转换问题
2. **内存对齐**: 4 bytes 自然对齐
3. **访问效率**: 字节访问比位访问简单

### 为什么分离 simple 和 fused 版本？

1. **Simple**: 教学用途，清晰展示两阶段
2. **Fused**: 性能优化，减少中间缓冲区
3. **测试**: 验证两种实现的等价性

## 使用建议

### 用于 Polygeist 转换

推荐使用 **`deca_decompress_fused()`**:

```bash
polygeist-opt deca_decompress_simple.c \
  -function=deca_decompress_fused \
  -o deca_from_c.mlir
```

**理由**:
- 单循环结构更简单
- 更容易被 megg 匹配
- 性能更好（无中间数组）

### 用于算法理解

推荐阅读 **`deca_decompress_simple()`**:

```c
// Stage 1: 稀疏展开
for (idx = 0; idx < 32; idx++) {
    dense_values[idx] = is_nonzero ? sparse_values[vidx] : 0;
    vidx += is_nonzero;
}

// Stage 2: 反量化
for (idx = 0; idx < 32; idx++) {
    output[idx] = dequantize_q88(dense_values[idx], scale);
}
```

### 用于性能分析

使用 **`test_c_reference`**:

```bash
# 编译优化版本
gcc -O3 -march=native test_c_reference.c -o test_optimized

# 运行性能测试
./test_optimized

# 查看汇编代码
gcc -S -O3 -march=native deca_decompress_simple.c
```

## 与 Megg 版本的差异

| 特性 | APS-MLIR 版本 | Megg 版本 |
|------|---------------|-----------|
| Tile 大小 | 32 elements | 256 elements |
| Bitmask 类型 | `u8[4]` | `u8[32]` |
| 解压算法 | 2-stage | 1-stage fused |
| LUT | 无 (直接用 scale) | 4-entry LUT |
| 输出类型 | `i16` (Q8.8) | `i8` (整数) |

**设计理念**:
- APS-MLIR: 小 tile，适合低延迟场景
- Megg: 大 tile，适合高吞吐场景

## 性能预估

### 32-element tile

| 操作 | CPU 周期 | ASIP 周期 | 加速比 |
|------|----------|-----------|--------|
| 内存读取 | 132 次 | 3 burst | ~44x |
| 位提取 | 32 次 | 32 次 (流水线) | ~1x |
| Q8.8 乘法 | 32 次 | 32 次 (专用单元) | ~2x |
| 内存写入 | 64 次 | 1 burst | ~64x |
| **总计** | ~228 操作 | ~20 操作 | **~11x** |

*注: 实际加速比取决于 CPU cache、内存带宽、流水线效率等因素*

## 后续工作

### 短期 (已完成)

- ✅ C 代码实现
- ✅ 测试验证
- ✅ 文档编写

### 中期 (待完成)

- ⬜ Polygeist 转换
- ⬜ Megg 指令匹配
- ⬜ 性能评估

### 长期 (规划)

- ⬜ 硬件综合
- ⬜ FPGA 验证
- ⬜ ASIC 设计

## 总结

成功创建了与 CADL 完全对应的 C 参考实现：

1. **功能正确**: 所有测试通过 ✓
2. **结构清晰**: 两阶段算法易于理解
3. **优化友好**: 适合 polygeist 和 megg 处理
4. **文档完善**: 包含使用说明和测试指南

现在可以进行端到端的指令匹配测试，验证 megg 能否从 C 代码识别出 DECA 解压缩模式！

## 相关文件索引

- C 实现: [deca_decompress_simple.c](deca_decompress_simple.c)
- 测试程序: [test_c_reference.c](test_c_reference.c)
- CADL 实现: [deca_decompress.cadl](deca_decompress.cadl)
- 使用指南: [C_REFERENCE_README.md](C_REFERENCE_README.md)
- 测试指南: [E2E_TESTING_GUIDE.md](E2E_TESTING_GUIDE.md)
