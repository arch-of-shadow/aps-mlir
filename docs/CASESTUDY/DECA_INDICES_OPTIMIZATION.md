# DECA 索引预计算优化方案分析

## 优化思路

**核心想法**：将稀疏解压缩分为两个阶段

### 当前方案（单阶段）
```
遍历 512 个位置 {
    读取 bitmask → 判断是否非零 → 计算 vidx → 读取 values[vidx] → 解压缩
}
```
- **问题**：`vidx` 数据依赖，导致 `values[vidx]` 非仿射访问

### 提议方案（两阶段）
```
阶段 1: 构建索引表
    遍历 bitmask → 生成 indices[0..nnz] = [所有非零位置]

阶段 2: 并行解压缩
    遍历 indices → values[i] 对应 decompressed_weights[indices[i]]
```
- **优势**：`values[i]` 是仿射访问！

---

## 方案 A: 密集索引数组（推荐）

### 实现代码

```cadl
// 新增：索引数组（最多 256 个非零元素）
#[partition_dim_array([0])]
#[partition_factor_array([4])]
#[partition_cyclic_array([1])]
static indices: [u16; 256];

#[opcode(7'b0101011)]
#[funct7(7'b0000001)]
rtype deca_decompress_indexed(rs1: u5, rs2: u5, rd: u5) {
    let base_addr: u32 = _irf[rs1];
    let out_addr: u32 = _irf[rs2];

    // Burst read compressed data
    bitmask[0 +: ] = _burst_read[base_addr +: 64];
    values[0 +: ] = _burst_read[base_addr + 64 +: 256];
    scales[0 +: ] = _burst_read[base_addr + 320 +: 32];

    // ========================================================================
    // 阶段 1: 构建索引数组（扫描 bitmask）
    // ========================================================================
    // 嵌套循环：外层遍历 64 个字节，内层遍历 8 个位
    [[unroll(2)]]  // 每周期处理 2 个字节
    with byte_i: u32 = (0, byte_i + 1), nnz: u32 = (0, nnz_out) do {
        let mask_byte: u8 = bitmask[byte_i];  // ✅ 仿射访问

        // 展开 8 个位的检查（每个位独立判断）
        [[unroll(8)]]
        with bit_i: u8 = (0, bit_i + 1), nnz_local: u32 = (nnz, nnz_next) do {
            let is_set: u1 = (mask_byte >> bit_i)[0:0];
            let global_idx: u16 = (byte_i * 8 + bit_i)[15:0];

            // 条件写入索引
            if is_set {
                indices[nnz_local] = global_idx;  // ✅ 仿射写入
            }

            // 更新非零计数
            let inc: u32 = if is_set {1} else {0};
            let nnz_next: u32 = nnz_local + inc;

            let bit_i_: u8 = bit_i + 1;
        } while bit_i_ < 8;

        let nnz_out: u32 = nnz_local;
        let byte_i_: u32 = byte_i + 1;
    } while byte_i_ < 64;

    // nnz_out 现在是非零元素总数

    // ========================================================================
    // 阶段 2: 并行解压缩（使用索引数组）
    // ========================================================================
    [[unroll(4)]]  // 并行处理 4 个非零元素
    with i: u32 = (0, i + 1) do {
        // 读取预计算的位置索引
        let pos: u16 = indices[i];  // ✅ 仿射访问: indices[i]

        // 计算组索引（仍然有除法，但可以进一步优化）
        let group_idx: u8 = (pos / 16)[4:0];
        let scale: i16 = scales[group_idx];

        // 读取稀疏值
        let sparse_val: i8 = values[i];  // ✅ 仿射访问: values[i]

        // Q8.8 反量化
        let mul_result: i32 = sparse_val * scale;
        let dequant: i16 = (mul_result >> 8)[15:0];

        // 写入到正确的位置
        decompressed_weights[pos] = dequant;  // ❌ 仍然非仿射（pos 来自数组）

        let i_: u32 = i + 1;
    } while i_ < nnz_out;

    // 填充零元素（可选：如果需要密集输出）
    with idx: u32 = (0, idx + 1) do {
        // 检查 idx 是否在 indices 中（需要额外逻辑）
        // 或者在阶段2中同时标记，这里跳过简化
    }

    _burst_write[out_addr +: 512] = decompressed_weights[0 +: ];
    _irf[rd] = nnz_out;
}
```

---

## 方案 B: 完全展开 + 稀疏-密集转换

### 核心思想
预先将稀疏 `values` 扩展为密集数组 `dense_values[512]`，非零位置填充实际值，零位置填充 0。

```cadl
// 新增：密集值数组
static dense_values: [i8; 512];

// 阶段 1: 稀疏到密集转换
[[unroll(2)]]
with byte_i: u32 = (0, byte_i + 1), vidx: u32 = (0, vidx_out) do {
    let mask_byte: u8 = bitmask[byte_i];  // ✅ 仿射

    [[unroll(8)]]
    with bit_i: u8 = (0, bit_i + 1), v: u32 = (vidx, v_next) do {
        let idx: u32 = byte_i * 8 + bit_i;
        let is_set: u1 = (mask_byte >> bit_i)[0:0];

        if is_set {
            dense_values[idx] = values[v];  // ✅ 两侧都仿射
        } else {
            dense_values[idx] = 0;
        }

        let inc: u32 = if is_set {1} else {0};
        let v_next: u32 = v + inc;

        let bit_i_: u8 = bit_i + 1;
    } while bit_i_ < 8;

    let vidx_out: u32 = v;
    let byte_i_: u32 = byte_i + 1;
} while byte_i_ < 64;

// 阶段 2: 密集解压缩（完全仿射！）
with group: u32 = (0, group + 1) do {
    let scale: i16 = scales[group];  // ✅ 仿射

    [[unroll(16)]]
    with elem: u32 = (0, elem + 1) do {
        let idx: u32 = group * 16 + elem;  // ✅ 仿射

        let sparse_val: i8 = dense_values[idx];  // ✅ 仿射: dense_values[group*16+elem]
        let mul_result: i32 = sparse_val * scale;
        let dequant: i16 = (mul_result >> 8)[15:0];

        decompressed_weights[idx] = dequant;  // ✅ 仿射

        let elem_: u32 = elem + 1;
    } while elem_ < 16;

    let group_: u32 = group + 1;
} while group_ < 32;
```

---

## 效率分析

### 计算复杂度对比

| 方案 | 阶段1（扫描） | 阶段2（解压） | 总操作数 |
|-----|-------------|--------------|---------|
| 原始单阶段 | - | 512 次迭代 | 512 |
| 方案A（索引） | 512 次检查 | nnz 次解压 | 512 + nnz |
| 方案B（密集） | 512 次转换 | 512 次解压 | 1024 |

**稀疏度影响**：
- 如果 `nnz < 256`（稀疏度 < 50%）：方案A 更少操作
- 如果 `nnz ≈ 512`（几乎密集）：方案B 操作数固定但简单

---

### 内存开销对比

| 方案 | 额外数组 | 大小 | 总内存 |
|-----|---------|------|--------|
| 原始 | - | - | 64 + 256 + 64 + 1024 = **1408 字节** |
| 方案A | `indices[256]` | 512 字节 | 1408 + 512 = **1920 字节** |
| 方案B | `dense_values[512]` | 512 字节 | 1408 + 512 = **1920 字节** |

**增加 36% 内存**（512 / 1408）

---

### 并行度对比

#### 原始方案
```
主循环 unroll(4):
  - bitmask[idx/8]: ❌ 非仿射 → 可能冲突
  - values[vidx]: ❌ 非仿射 → 串行化
  - decompressed_weights[idx]: ✅ 仿射 → 并行

实际并行度：受限于 values 访问（可能 II=4）
```

#### 方案A（索引预计算）
```
阶段1 unroll(2) × unroll(8) = 16-way:
  - bitmask[byte_i]: ✅ 仿射 → 并行
  - indices[nnz]: ✅ 仿射写入 → 并行（有条件）

阶段2 unroll(4):
  - indices[i]: ✅ 仿射 → 并行
  - values[i]: ✅ 仿射 → 并行！
  - decompressed_weights[pos]: ❌ 非仿射（pos来自数组）

实际并行度：values 访问改善，但输出仍有冲突
```

#### 方案B（密集转换）
```
阶段1 unroll(2) × unroll(8) = 16-way:
  - bitmask[byte_i]: ✅ 仿射 → 并行
  - values[v]: ❌ 仍然非仿射（v 是数据依赖）
  - dense_values[idx]: ✅ 仿射写入 → 并行

阶段2 unroll(16):
  - scales[group]: ✅ 仿射 → 并行
  - dense_values[group*16+elem]: ✅ 完全仿射 → 完美并行！
  - decompressed_weights[idx]: ✅ 仿射 → 完美并行！

实际并行度：阶段2 完全仿射，理想流水线 II=1
```

---

### 延迟对比（估算）

假设：
- `nnz = 128`（25% 稀疏度）
- 展开因子 `unroll(4)`
- BRAM 读取延迟 = 1 周期
- II（理想）= 1

| 方案 | 阶段1 | 阶段2 | 总延迟 | 仿射访问 |
|-----|-------|-------|--------|---------|
| 原始 | - | 512 / 4 = 128 周期（II≈2-4 因非仿射） | **256-512 周期** | ❌ |
| 方案A | 512 / 16 = 32 周期 | 128 / 4 = 32 周期 | **64 周期** | 部分 ✅ |
| 方案B | 512 / 16 = 32 周期 | 512 / 16 = 32 周期 | **64 周期** | 完全 ✅ |

**加速比**：**4x - 8x**（主要来自阶段2的并行化）

---

## 仿射性分析

### 方案A 的仿射改进

| 数组访问 | 原始 | 方案A（阶段1） | 方案A（阶段2） | 仿射？ |
|---------|------|---------------|---------------|-------|
| `bitmask[idx/8]` | ❌ | `bitmask[byte_i]` | - | ✅ |
| `scales[(idx/16)[4:0]]` | ❌ | - | `scales[pos/16]` | ❌（仍有除法）|
| `values[vidx]` | ❌ | - | `values[i]` | ✅ |
| `indices[nnz]` | - | ✅ | `indices[i]` | ✅ |
| `decompressed_weights[idx]` | ✅ | - | `[pos]` | ❌（pos来自数组）|

**改进**：3/3 → 2/5（新增 indices 访问是仿射的）

---

### 方案B 的仿射改进

| 数组访问 | 原始 | 方案B（阶段1） | 方案B（阶段2） | 仿射？ |
|---------|------|---------------|---------------|-------|
| `bitmask[idx/8]` | ❌ | `bitmask[byte_i]` | - | ✅ |
| `scales[(idx/16)[4:0]]` | ❌ | - | `scales[group]` | ✅ |
| `values[vidx]` | ❌ | `values[v]` | - | ❌（仍数据依赖）|
| `dense_values[idx]` | - | `[byte_i*8+bit_i]` | `[group*16+elem]` | ✅ |
| `decompressed_weights[idx]` | ✅ | - | `[group*16+elem]` | ✅ |

**改进**：阶段1 仍有 1 个非仿射，但阶段2 **完全仿射**！

---

## 推荐方案：方案B（密集转换）

### 理由

1. **阶段2 完全仿射**
   - `dense_values[group*16+elem]` ✅
   - `scales[group]` ✅
   - `decompressed_weights[group*16+elem]` ✅
   - 可以达到 **II=1** 的完美流水线

2. **代码规整性**
   - 阶段2 是标准的双层嵌套循环
   - 编译器更容易优化

3. **内存访问规律**
   - 阶段2 的访问是顺序的（group × 16）
   - 利于 burst 访问和缓存

4. **与硬件特性匹配**
   - 阶段1 的稀疏访问无法避免（这是算法本质）
   - 将复杂性隔离在阶段1，阶段2 高效执行

---

## 完整实现（方案B优化版）

```cadl
// 新增密集值数组
#[partition_dim_array([0])]
#[partition_factor_array([16])]  // 匹配组大小
#[partition_cyclic_array([1])]
static dense_values: [i8; 512];

#[opcode(7'b0101011)]
#[funct7(7'b0000001)]
rtype deca_decompress_optimized(rs1: u5, rs2: u5, rd: u5) {
    let base_addr: u32 = _irf[rs1];
    let out_addr: u32 = _irf[rs2];

    // Burst read
    bitmask[0 +: ] = _burst_read[base_addr +: 64];
    values[0 +: ] = _burst_read[base_addr + 64 +: 256];
    scales[0 +: ] = _burst_read[base_addr + 320 +: 32];

    // ========================================================================
    // 阶段 1: 稀疏到密集转换（消除 vidx 数据依赖）
    // ========================================================================
    [[unroll(2)]]  // 每周期处理 2 个字节 = 16 个元素
    with byte_i: u32 = (0, byte_i + 1), vidx: u32 = (0, vidx_out) do {
        let mask_byte: u8 = bitmask[byte_i];  // ✅ 仿射: bitmask[byte_i]

        // 完全展开 8 个位
        [[unroll(8)]]
        with bit_i: u8 = (0, bit_i + 1), v: u32 = (vidx, v_next) do {
            let idx: u32 = byte_i * 8 + bit_i;
            let is_set: u1 = (mask_byte >> bit_i)[0:0];

            // 条件赋值
            let val: i8 = if is_set { values[v] } else { 0 };  // values[v] 仍非仿射
            dense_values[idx] = val;  // ✅ 仿射: dense_values[byte_i*8+bit_i]

            let inc: u32 = if is_set {1} else {0};
            let v_next: u32 = v + inc;

            let bit_i_: u8 = bit_i + 1;
        } while bit_i_ < 8;

        let vidx_out: u32 = v;
        let byte_i_: u32 = byte_i + 1;
    } while byte_i_ < 64;

    // ========================================================================
    // 阶段 2: 密集解压缩（完全仿射！）
    // ========================================================================
    // 外层：32 个组
    with group: u32 = (0, group + 1) do {
        let scale: i16 = scales[group];  // ✅ 仿射: scales[group]

        // 内层：每组 16 个元素（完全展开）
        [[unroll(16)]]
        with elem: u32 = (0, elem + 1) do {
            let idx: u32 = group * 16 + elem;  // ✅ 仿射索引

            let sparse_val: i8 = dense_values[idx];  // ✅ 仿射: dense_values[group*16+elem]

            // Q8.8 反量化
            let mul_result: i32 = sparse_val * scale;
            let dequant: i16 = (mul_result >> 8)[15:0];

            decompressed_weights[idx] = dequant;  // ✅ 仿射: [group*16+elem]

            let elem_: u32 = elem + 1;
        } while elem_ < 16;

        let group_: u32 = group + 1;
    } while group_ < 32;

    // Burst write
    _burst_write[out_addr +: 512] = decompressed_weights[0 +: ];

    // 返回非零元素数
    _irf[rd] = vidx_out;
}
```

---

## 性能估算

### 阶段1（稀疏转换）

```
展开因子：unroll(2) × unroll(8) = 16-way 并行
迭代次数：64 / 2 = 32 次外层迭代
每次迭代：8 次内层（完全展开）

理想延迟：32 周期
实际延迟：~40-50 周期（考虑 values[v] 的冲突）
```

### 阶段2（密集解压）

```
展开因子：unroll(16)（完全展开内层）
迭代次数：32 次外层迭代

每次迭代处理 16 个元素：
  - 1 次 scales 读取（寄存器保存）
  - 16 次 dense_values 读取（完全分区 → 并行）
  - 16 次 decompressed_weights 写入（分区 → 并行）

理想 II：1
实际延迟：32 周期（几乎理想）
```

### 总延迟

```
总延迟 = 阶段1 + 阶段2
      ≈ 50 + 32
      = 82 周期

对比原始方案（256-512 周期）：
加速比 = 3x - 6x
```

---

## Warning 消除效果

### 运行 affine pipeline 后

**预期结果**：

```
✅ 阶段2 无 warning（完全仿射）
⚠️ 阶段1 仍有 1 个 warning：values[v]
  - 这是算法本质，但影响有限（只在阶段1）
```

**对比原始**：
- 原始：3 个 warning（bitmask, scales, values）
- 方案B：1 个 warning（只有阶段1的 values）
- 改善：**67% 减少**

---

## 总结

### 您的想法非常正确！

预计算索引（或转换为密集）确实能**显著提升性能**：

1. **仿射性改善**：从 3 个非仿射访问减少到 1 个
2. **并行度提升**：阶段2 可以达到 II=1 的完美流水线
3. **加速比**：3x - 6x（根据稀疏度）
4. **代价**：额外 512 字节内存（+36%）

### 推荐实施方案B（密集转换）

- **优势**：阶段2 完全仿射，代码规整
- **适用**：稀疏度 < 80% 时收益明显
- **风险**：如果内存极度紧张，考虑方案A

### 下一步

1. 实现上述代码
2. 运行 affine pipeline 验证 warning 减少
3. 使用 HLS 工具测量实际延迟和资源使用
