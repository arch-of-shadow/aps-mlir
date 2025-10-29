# DECA 仿射变换可行性分析

## 问题概述

当前 `deca_decompress.cadl` 产生 3 个非仿射访问警告：

1. **bitmask**: `array[idx/8]` - 除法导致非仿射
2. **scales**: `array[(idx/16)[4:0]]` - 除法+位提取
3. **values**: `array[vidx]` - 数据依赖索引

本文档分析：**在不改变算法语义的前提下**，能否将这些访问模式改写为仿射形式。

---

## 数学本质分析

### 问题 1: bitmask[idx/8] - 字节索引

**当前代码**:
```cadl
let byte_idx: u32 = idx / 8;           // ❌ 非仿射
let bit_pos: u8 = idx[2:0];            // 等价于 idx % 8
let mask_byte: u8 = bitmask[byte_idx];
```

**数学关系**:
- `idx ∈ [0, 512)` 遍历所有元素
- `byte_idx = ⌊idx / 8⌋` 确定字节位置
- `bit_pos = idx mod 8` 确定字节内位偏移

**仿射变换方案**: ✅ **可以改写**

#### 方案 1A: 双层循环展开 (推荐)

将单层循环 `idx ∈ [0, 512)` 改写为双层嵌套循环：

```cadl
// 外层: 遍历 64 个字节
with byte_i: u32 = (0, byte_i + 1) do {
    let mask_byte: u8 = bitmask[byte_i];  // ✅ 仿射: array[i]

    // 内层: 遍历字节内 8 个位
    [[unroll(8)]]
    with bit_i: u8 = (0, bit_i + 1), vidx: u32 = (0, vidx_next) do {
        let is_nonzero: u1 = (mask_byte >> bit_i)[0:0];

        let idx: u32 = byte_i * 8 + bit_i;  // 重构全局索引
        let group_idx: u8 = idx / 16;       // 仍需处理 (见后文)

        // ... 其余逻辑

        let bit_i_: u8 = bit_i + 1;
    } while bit_i_ < 8;

    let byte_i_: u32 = byte_i + 1;
} while byte_i_ < 64;
```

**优势**:
- `bitmask[byte_i]` 变为 ✅ **仿射访问** `array[i]`
- 内层循环展开后，`mask_byte` 可以复用 8 次
- 符合硬件优化：按字节流水处理

**劣势**:
- 增加一层嵌套
- 需要重构全局索引 `idx = byte_i * 8 + bit_i`（但这是仿射表达式）

---

#### 方案 1B: 位扩展缓冲区

预先将 bitmask 扩展为 512 位的密集数组：

```cadl
// 新增缓冲区
static dense_mask: [u1; 512];

// 第一阶段：扩展 bitmask (可完全展开)
[[unroll(64)]]
with byte_i: u32 = (0, byte_i + 1) do {
    let mask_byte: u8 = bitmask[byte_i];  // ✅ 仿射

    // 展开 8 位写入
    dense_mask[byte_i * 8 + 0] = mask_byte[0:0];
    dense_mask[byte_i * 8 + 1] = mask_byte[1:1];
    dense_mask[byte_i * 8 + 2] = mask_byte[2:2];
    dense_mask[byte_i * 8 + 3] = mask_byte[3:3];
    dense_mask[byte_i * 8 + 4] = mask_byte[4:4];
    dense_mask[byte_i * 8 + 5] = mask_byte[5:5];
    dense_mask[byte_i * 8 + 6] = mask_byte[6:6];
    dense_mask[byte_i * 8 + 7] = mask_byte[7:7];

    let byte_i_: u32 = byte_i + 1;
} while byte_i_ < 64;

// 第二阶段：使用密集 mask
with idx: u32 = (0, idx + 1), vidx: u32 = (0, vidx_next) do {
    let is_nonzero: u1 = dense_mask[idx];  // ✅ 仿射: array[idx]
    // ... 其余逻辑
}
```

**优势**:
- `dense_mask[idx]` 完全仿射
- 原始循环结构不变

**劣势**:
- 需要额外 512 位缓冲区（64 字节）
- 增加一个预处理循环（延迟 +64 周期）

---

### 问题 2: scales[(idx/16)[4:0]] - 组索引

**当前代码**:
```cadl
let group_idx: u8 = (idx / 16)[4:0];  // ❌ 除法 + 位提取
let scale: i16 = scales[group_idx];
```

**数学关系**:
- 512 个元素分为 32 组，每组 16 个元素
- `group_idx = ⌊idx / 16⌋ mod 32`
- 由于 `idx < 512 = 16 × 32`，所以 `⌊idx / 16⌋ < 32`，位提取 `[4:0]` 实际上是冗余的

**仿射变换方案**: ✅ **可以改写**

#### 方案 2A: 三层循环分组 (推荐)

将循环改为 **外层遍历组，内层遍历组内元素**：

```cadl
// 外层: 遍历 32 个组
with group: u32 = (0, group + 1) do {
    let scale: i16 = scales[group];  // ✅ 仿射: array[group]

    // 内层: 遍历组内 16 个元素
    [[unroll(16)]]
    with elem: u32 = (0, elem + 1), vidx: u32 = (vidx_in, vidx_out) do {
        let idx: u32 = group * 16 + elem;  // ✅ 仿射: group*16 + elem

        let byte_idx: u32 = idx / 8;  // 仍需处理 (见方案组合)
        let bit_pos: u8 = idx[2:0];
        let mask_byte: u8 = bitmask[byte_idx];
        let is_nonzero: u1 = (mask_byte >> bit_pos)[0:0];

        let sparse_val: i8 = values[vidx];  // 仍然数据依赖
        let mul_result: i32 = sparse_val * scale;
        let dequant: i16 = (mul_result >> 8)[15:0];

        let final_val: i16 = if is_nonzero {dequant} else {0};
        decompressed_weights[idx] = final_val;  // ✅ 仿射

        let inc_val: u32 = if is_nonzero {1} else {0};
        let vidx_out: u32 = vidx + inc_val;

        let elem_: u32 = elem + 1;
    } while elem_ < 16;

    let group_: u32 = group + 1;
} while group_ < 32;
```

**优势**:
- `scales[group]` 变为 ✅ **仿射访问**
- `scale` 在内层循环中可复用 16 次（寄存器保存）
- 硬件优化：减少 scale 读取次数（从 512 次降到 32 次）

**劣势**:
- 增加嵌套深度
- 需要传递 `vidx` 跨内层循环（通过 iter_args）

---

#### 方案 2B: 查找表 (LUT)

如果接受常量数组的代价，可以预生成查找表：

```cadl
// 编译时生成的常量数组
static group_lut: [u8; 512] = [
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  // idx 0-15 → group 0
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,  // idx 16-31 → group 1
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,  // idx 32-47 → group 2
    // ... (共 512 个元素)
];

// 在循环中使用
let group_idx: u8 = group_lut[idx];  // ✅ 仿射: array[idx]
let scale: i16 = scales[group_idx];  // ❌ 仍然非仿射 (group_idx 是从数组读取的)
```

**问题**: 虽然 `group_lut[idx]` 是仿射的，但 `scales[group_idx]` 仍然是**数据依赖访问**（即使数据是常量）。

**结论**: ❌ **此方案无效** - 间接索引仍然非仿射。

---

### 问题 3: values[vidx] - 稀疏值索引

**当前代码**:
```cadl
with idx: u32 = (0, idx + 1)
     vidx: u32 = (0, vidx_next)
do {
    let sparse_val: i8 = values[vidx];  // ❌ 数据依赖
    // ...
    let inc_val: u32 = if is_nonzero {1} else {0};
    let vidx_next: u32 = vidx + inc_val;  // vidx 基于 bitmask 更新
}
```

**数学本质**:
- `vidx` 是稀疏索引：只计数非零元素
- `vidx_next = vidx + popcount(is_nonzero)`
- `vidx = sum_{i=0}^{idx-1} bitmask[i]` （前缀和）

**仿射可行性**: ❌ **无法改写为仿射**

#### 不可行原因

仿射访问要求索引是循环变量的**多项式函数**（通常是一次多项式）：
```
array[a*i + b*j + c]  // a, b, c 是常量
```

但 `vidx` 是 `bitmask` 的**前缀和**：
```
vidx(idx) = Σ_{i=0}^{idx-1} bitmask[i]
```

这是**数据依赖函数**，不能表示为 `idx` 的仿射函数。

#### 替代方案：密集访问（改变算法）

唯一的仿射方案是**放弃稀疏格式**：

```cadl
// 将 values 扩展为 512 元素的密集数组
static dense_values: [i8; 512];

with idx: u32 = (0, idx + 1) do {
    let sparse_val: i8 = dense_values[idx];  // ✅ 仿射
    // 对于 bitmask[idx] == 0 的位置，dense_values[idx] = 0
}
```

**代价**:
- 内存从 256 字节增加到 512 字节（2x 膨胀）
- 违反了 "不改变算法" 的前提（从稀疏变为密集）

**结论**: ❌ **在保持稀疏格式的前提下，`values[vidx]` 无法仿射化**

---

## 组合方案：最优仿射改写

结合上述分析，这里是**在不改变算法语义的前提下**，能达到的最优仿射程度：

### 推荐方案：三层嵌套循环 + 位展开

```cadl
#[opcode(7'b0101011)]
#[funct7(7'b0000001)]
rtype deca_decompress_affine(rs1: u5, rs2: u5, rd: u5) {
    let base_addr: u32 = _irf[rs1];
    let out_addr: u32 = _irf[rs2];

    // Burst read (不变)
    bitmask[0 +: ] = _burst_read[base_addr +: 64];
    values[0 +: ] = _burst_read[base_addr + 64 +: 256];
    scales[0 +: ] = _burst_read[base_addr + 320 +: 32];

    // 三层嵌套循环
    // Layer 1: 遍历 32 个组
    with group: u32 = (0, group + 1), vidx_g: u32 = (0, vidx_g_out) do {
        let scale: i16 = scales[group];  // ✅ 仿射: scales[group]

        // Layer 2: 遍历组内 2 个字节 (每组 16 元素 = 2 字节)
        [[unroll(2)]]
        with byte_in_group: u32 = (0, byte_in_group + 1),
             vidx_b: u32 = (vidx_g, vidx_b_out) do {

            let byte_idx: u32 = group * 2 + byte_in_group;  // ✅ 仿射
            let mask_byte: u8 = bitmask[byte_idx];          // ✅ 仿射: bitmask[group*2 + byte_in_group]

            // Layer 3: 遍历字节内 8 个位 (完全展开)
            [[unroll(8)]]
            with bit_i: u8 = (0, bit_i + 1), vidx: u32 = (vidx_b, vidx_next) do {
                let is_nonzero: u1 = (mask_byte >> bit_i)[0:0];

                // 重构全局索引
                let idx: u32 = group * 16 + byte_in_group * 8 + bit_i;  // ✅ 仿射

                // ❌ 仍然非仿射：values[vidx]
                let sparse_val: i8 = values[vidx];

                let mul_result: i32 = sparse_val * scale;
                let dequant: i16 = (mul_result >> 8)[15:0];
                let final_val: i16 = if is_nonzero {dequant} else {0};

                decompressed_weights[idx] = final_val;  // ✅ 仿射

                let inc_val: u32 = if is_nonzero {1} else {0};
                let vidx_next: u32 = vidx + inc_val;

                let bit_i_: u8 = bit_i + 1;
            } while bit_i_ < 8;

            let vidx_b_out: u32 = vidx;  // 传递到下一个字节
            let byte_in_group_: u32 = byte_in_group + 1;
        } while byte_in_group_ < 2;

        let vidx_g_out: u32 = vidx_b;  // 传递到下一个组
        let group_: u32 = group + 1;
    } while group_ < 32;

    _burst_write[out_addr +: 512] = decompressed_weights[0 +: ];
    _irf[rd] = vidx_g;
}
```

### 仿射性分析

| 数组访问 | 原始代码 | 改写后 | 仿射? |
|---------|---------|--------|-------|
| `bitmask[idx/8]` | `idx/8` | `group*2 + byte_in_group` | ✅ 仿射 |
| `scales[(idx/16)[4:0]]` | `idx/16` | `group` | ✅ 仿射 |
| `values[vidx]` | `vidx` | `vidx` | ❌ 非仿射（无法避免） |
| `decompressed_weights[idx]` | `idx` | `group*16 + byte_in_group*8 + bit_i` | ✅ 仿射 |

### 改进效果

- **警告数量**: 从 3 个减少到 **1 个**（只剩 `values`）
- **bitmask**: ✅ **已解决** - 改为 `array[group*2 + byte_in_group]`
- **scales**: ✅ **已解决** - 改为 `array[group]`
- **values**: ❌ **无法解决** - 稀疏格式的本质限制

### 性能优化副作用

此改写还带来额外的硬件优化：

1. **减少 scale 读取**：从 512 次降到 32 次（每组读一次）
2. **提高 bitmask 复用**：每个 `mask_byte` 被 8 个位共享
3. **更好的流水线**：内层循环展开后，减少分支

---

## 最终结论

### 可行性总结

| 非仿射访问 | 可否改写为仿射？ | 改写方法 | 代价 |
|-----------|----------------|---------|------|
| `bitmask[idx/8]` | ✅ **可以** | 嵌套循环：外层遍历字节，内层遍历位 | 增加嵌套深度 |
| `scales[(idx/16)[4:0]]` | ✅ **可以** | 嵌套循环：外层遍历组，内层遍历元素 | 增加嵌套深度 |
| `values[vidx]` | ❌ **不可以** | 无法在保持稀疏格式时仿射化 | - |

### 推荐实施方案

**方案 A: 三层嵌套循环（推荐）**
- 实现上述组合方案
- 警告数量：3 → 1
- 代码复杂度：中等
- 硬件性能：**提升**（减少 scale 和 bitmask 读取）

**方案 B: 接受现状**
- 保持当前代码
- 警告数量：3
- 说明：`values` 的非仿射访问是稀疏压缩的本质特征，无法避免
- 实际影响：警告不影响综合，只是数组分区可能不是最优

### 语义等价性保证

上述改写方案满足以下保证：

1. ✅ **功能等价**：输出的 `decompressed_weights` 完全一致
2. ✅ **数值等价**：所有计算的中间结果相同
3. ✅ **顺序等价**：元素处理顺序可能改变（先按组再按元素），但最终结果不变
4. ✅ **接口兼容**：输入输出格式完全一致

### 实现建议

1. **优先级**：如果警告影响综合质量（QoR），实施方案 A
2. **测试**：对比改写前后的硬件仿真结果
3. **基准**：对比资源使用（LUT/FF/BRAM）和时序（Fmax）

---

## 附录：为什么 values[vidx] 本质上非仿射

### 反证法

假设存在仿射函数 `f` 使得 `vidx = f(idx)`，即：
```
vidx = a*idx + b  // a, b 为常量
```

考虑两种情况的 bitmask：

**Case 1**: 所有位为 1
```
vidx(0) = 0
vidx(1) = 1
vidx(2) = 2
...
```
推导：`vidx = 1*idx + 0`

**Case 2**: 只有偶数位为 1
```
vidx(0) = 0
vidx(1) = 0  (bit 1 = 0)
vidx(2) = 1  (bit 2 = 1)
vidx(3) = 1  (bit 3 = 0)
vidx(4) = 2  (bit 4 = 1)
```
推导：`vidx = 0.5*idx + 0`

两个常量系数矛盾！**证明不存在统一的仿射函数**。

### 图论视角

将索引映射建模为图：
- 节点：`idx` 和 `vidx`
- 边：`idx → vidx` 的映射

**仿射映射的图**是一条直线（或平行线簇）。
**稀疏映射的图**是阶梯函数（非连续，斜率变化）。

因此，稀疏索引**拓扑上无法嵌入仿射空间**。

---

## 参考资料

- **Polyhedral Model**: 仿射循环变换的理论基础
- **Affine Map 定义**: `f(i₁, i₂, ..., iₙ) = a₁i₁ + a₂i₂ + ... + aₙiₙ + c`
- **MLIR Affine Dialect**: https://mlir.llvm.org/docs/Dialects/Affine/
