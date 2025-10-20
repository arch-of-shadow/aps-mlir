# DECA C Reference Implementations for Polygeist

这个目录包含了 DECA 解压缩算法的 C 参考实现，用于 polygeist 转换和指令匹配测试。

## 文件说明

### 1. [deca_decompress_simple.c](deca_decompress_simple.c)
**推荐用于 polygeist 转换**

包含三个版本的实现：

#### `deca_decompress_simple()`
- 两阶段处理（与 CADL 完全对应）
- Stage 1: 稀疏到密集展开
- Stage 2: Q8.8 反量化
- 适合学习和理解算法

#### `deca_decompress_fused()` ⭐ **推荐**
- 单循环融合版本
- 更简洁，更容易被 megg 匹配
- 适合指令匹配测试

#### 辅助函数
- `extract_bit_from_bitmask()`: 位提取模式
- `dequantize_q88()`: Q8.8 反量化模式

### 2. [deca_decompress.c](deca_decompress.c)
- 完整的内存访问版本
- 包含 burst read/write 模拟
- 与 CADL 的内存布局完全对应
- 用于理解完整的数据流

## 使用方法

### Step 1: 用 Polygeist 转换为 MLIR

```bash
# 转换 simple 版本
polygeist-opt deca_decompress_simple.c \
  -function=deca_decompress_fused \
  -o deca_decompress_fused.mlir

# 或者转换两阶段版本
polygeist-opt deca_decompress_simple.c \
  -function=deca_decompress_simple \
  -o deca_decompress_simple.mlir
```

### Step 2: 用 megg 进行指令匹配

```bash
# 使用你的 CADL 生成的自定义指令定义
cd /home/cloud/megg

# 生成自定义指令模式（从 CADL，使用标准化 MLIR）
# 使用 mlir-std 生成完全标准化的 MLIR（无 comb dialect）
pixi run mlir-std /home/cloud/aps-mlir/examples/deca/deca_decompress.cadl \
  > custom_deca_instr.mlir

# 运行 megg 优化
./megg-opt.py deca_decompress_fused.mlir \
  --custom-instructions custom_deca_instr.mlir \
  -o deca_optimized.mlir
```

### Step 3: 验证匹配结果

检查 `deca_optimized.mlir` 中是否出现了自定义指令调用：

```mlir
// 期望看到类似这样的模式
%result = aps.custom_instr "deca_decompress_u1"(%base_addr, %scale)
  : (i32, i16) -> i32
```

## 对应关系

| C 函数 | CADL 函数 | 特点 |
|--------|-----------|------|
| `deca_decompress_simple()` | `deca_decompress_u1()` | 两阶段，完全对应 |
| `deca_decompress_fused()` | 未来可优化版本 | 单循环，更高效 |
| `dequantize_q88()` | Stage 2 的核心 | Q8.8 乘法模式 |
| `extract_bit_from_bitmask()` | Stage 1 的核心 | 位提取模式 |

## 数据类型对应

| C 类型 | CADL 类型 | 说明 |
|--------|-----------|------|
| `uint8_t` | `u8` | 无符号 8 位 |
| `int8_t` | `i8` | 有符号 8 位 |
| `int16_t` | `i16` | 有符号 16 位 |
| `uint32_t` | `u32` | 无符号 32 位 |

## 算法说明

### 输入数据

```c
// Bitmask: 32 bits stored in 4 bytes
uint8_t bitmask[4] = {0xFF, 0x0F, 0xAA, 0x55};

// Sparse values: only non-zero elements
int8_t sparse_values[N];  // N = popcount(bitmask)

// Scale factor (Q8.8 format)
int16_t scale = 0x0100;  // 1.0 in Q8.8
```

### 输出数据

```c
// Decompressed weights: 32 elements
int16_t output[32];

// Return value: number of non-zero elements
uint32_t nnz = deca_decompress_fused(bitmask, sparse_values, scale, output);
```

### Q8.8 固定点格式

```
Q8.8 = 8 整数位 + 8 小数位

示例:
  0x0100 = 1.0
  0x0080 = 0.5
  0x0200 = 2.0
  0xFF00 = -1.0 (补码)

计算公式:
  result = (value * scale) >> 8
```

## 测试示例

```c
#include "deca_decompress_simple.c"
#include <stdio.h>

int main() {
    // Test data
    uint8_t bitmask[4] = {0xFF, 0xFF, 0x00, 0x00};  // 前16个非零
    int8_t sparse_values[16] = {1, 2, 3, 4, 5, 6, 7, 8,
                                 9, 10, 11, 12, 13, 14, 15, 16};
    int16_t scale = 0x0200;  // 2.0 in Q8.8
    int16_t output[32];

    // Run decompression
    uint32_t nnz = deca_decompress_fused(bitmask, sparse_values, scale, output);

    // Print results
    printf("Non-zero count: %u\n", nnz);
    for (int i = 0; i < 32; i++) {
        printf("output[%d] = %d\n", i, output[i]);
    }

    return 0;
}
```

## Megg 指令匹配策略

### 方案 1: 匹配整个循环体

megg 会尝试匹配整个 `for` 循环的计算模式，将其替换为单个 ASIP 调用。

**优点**: 最大化硬件加速
**缺点**: 模式匹配要求严格

### 方案 2: 匹配子模式

megg 可以匹配循环内的子计算模式（如位提取、Q8.8 乘法），逐步优化。

**优点**: 更灵活，可以部分匹配
**缺点**: 加速效果可能不如完整匹配

### 方案 3: 混合模式

结合两种策略，先尝试完整匹配，失败后回退到子模式匹配。

## 调试技巧

### 1. 查看 Polygeist 生成的 MLIR

```bash
polygeist-opt deca_decompress_simple.c \
  -function=deca_decompress_fused \
  --mlir-print-ir-after-all \
  2>&1 | less
```

### 2. 查看 megg 的匹配过程

```bash
./megg-opt.py input.mlir \
  --custom-instructions custom.mlir \
  --verbose \
  --debug-pattern-matching \
  -o output.mlir
```

### 3. 比对 MLIR 结构

```bash
# C 生成的 MLIR
polygeist-opt deca_decompress_simple.c -o from_c.mlir

# CADL 生成的 MLIR
pixi run mlir examples/deca/deca_decompress.cadl > from_cadl.mlir

# 比对
diff -u from_c.mlir from_cadl.mlir
```

## 相关文件

- CADL 实现: [../deca_decompress.cadl](deca_decompress.cadl)
- megg 文档: `/home/cloud/megg/docs/complex_instruction_matching.md`
- megg DECA 示例: `/home/cloud/megg/tests/benchmarks/deca/`

## 下一步

1. 用 polygeist 转换 C 代码到 MLIR
2. 用 megg 进行指令匹配测试
3. 比对匹配结果和手写 CADL 的性能
4. 迭代优化 C 代码以提高匹配成功率
