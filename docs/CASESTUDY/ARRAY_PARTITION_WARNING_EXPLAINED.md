# Array Partition Warning 详解

## Warning 信息

```
warning: global variable "bitmask" with applying array_partition pragma is failed,
because non standard affine access.
```

## Warning 的含义

### 这个 warning 在说什么？

编译器在执行 `--new-array-partition` pass 时，尝试根据你的 partition pragma 将数组分割成多个存储体（memory banks），但发现数组的访问模式**不符合仿射形式**，导致无法自动完成分区优化。

### 触发条件

查看源码 [lib/TOR/NewArrayPartition.cpp:202-220](../lib/TOR/NewArrayPartition.cpp#L202-L220)：

```cpp
bool rankCanBePartition(Value arg, size_t rank, unsigned bank_factor,
                        bool cyclic, MLIRContext *ctx) {
  for (auto *op : arg.getUsers()) {
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      if (getMemBank(load.getAffineMap(), rank, ctx, bank_factor, cyclic) == -1) {
        return false;  // ❌ 无法确定 bank 编号
      }
    }
    // ... store 同理
  }
  return true;
}
```

**关键逻辑**：
- 编译器遍历数组的所有访问操作（load/store）
- 对每个访问，尝试计算它访问的是哪个 bank（`getMemBank`）
- 如果**无法在编译期确定 bank 编号**，返回 `-1`，触发 warning

### 什么情况下无法确定 bank？

查看 [lib/TOR/NewArrayPartition.cpp:116-132](../lib/TOR/NewArrayPartition.cpp#L116-L132)：

```cpp
int getMemBank(AffineMap map, int rank, MLIRContext *ctx, int factor, bool cyclic) {
  auto expr = map.getResult(rank);

  // 条件 1: 表达式包含符号变量（symbolic）
  if (!expr || hasSymbolic(expr)) {
    return -1;  // ❌ 不能有符号变量
  }

  // 计算 bank 编号
  if (cyclic) {
    expr = expr % factor;     // 循环分区：bank = index % factor
  } else {
    expr = expr.floorDiv(factor);  // 块分区：bank = index / factor
  }

  auto compose_map = AffineMap::get(map.getNumDims(), 0, expr, ctx);

  // 条件 2: bank 编号必须是编译期常量
  if (compose_map.isConstant()) {
    return compose_map.getConstantResults()[0];  // ✅ 返回常量 bank 编号
  }

  return -1;  // ❌ bank 编号不是常量
}
```

**失败原因总结**：

1. **访问索引包含符号变量**（非循环归纳变量）
   ```mlir
   // 例如：index 来自函数参数或数据依赖值
   %val = affine.load %array[%symbol_idx] : memref<64xi8>
   ```

2. **Bank 编号无法化简为常量**
   ```mlir
   // 例如：idx / 8，其中 idx 是循环变量
   // cyclic=0 时，bank = idx / 8 / factor = idx / (8*factor)
   // 这不是常量！
   %val = affine.load %array[%idx floordiv 8] : memref<64xi8>
   ```

---

## 对 deca_decompress.cadl 的具体分析

### 案例 1: bitmask[idx/8]

**CADL 代码**：
```cadl
let byte_idx: u32 = idx / 8;
let mask_byte: u8 = bitmask[byte_idx];
```

**Partition 配置**：
```cadl
#[partition_dim_array([0])]
#[partition_factor_array([64])]  // 完全分区：64 个 bank
#[partition_cyclic_array([0])]   // 块分区（非循环）
```

**MLIR 表示**：
```mlir
%c8 = arith.constant 8 : i32
%byte_idx = arith.divui %idx, %c8 : i32        // 非仿射操作
%val = aps.memload %bitmask[%byte_idx] : ...   // 索引是 divui 的结果
```

**为什么失败？**

1. `idx / 8` 被表示为 `arith.divui`（无符号整数除法），这**不是 affine.apply 操作**
2. 即使是仿射除法 `affine_map<(d0) -> (d0 floordiv 8)>`：
   - Bank 计算（块分区）：`bank = (idx / 8) / 64 = idx / 512`
   - 对于 `idx ∈ [0, 512)`，`bank` 不是常量（是 0 或 1）
   - 编译器无法为**每个访问**静态分配 bank

**实际影响**：
- 编译器**放弃**对 `bitmask` 的自动分区
- `bitmask` 保持为**单个存储体**（未分区）

---

### 案例 2: scales[(idx/16)[4:0]]

**CADL 代码**：
```cadl
let group_idx: u8 = (idx / 16)[4:0];
let scale: i16 = scales[group_idx];
```

**Partition 配置**：
```cadl
#[partition_factor_array([32])]  // 完全分区：32 个 bank
#[partition_cyclic_array([0])]   // 块分区
```

**MLIR 表示**：
```mlir
%c16 = arith.constant 16 : i32
%div = arith.divui %idx, %c16 : i32
%extract = comb.extract %div from 0 : (i32) -> i5  // 位提取 [4:0]
%ext = arith.extui %extract : i5 to i8
%val = aps.memload %scales[%ext] : ...
```

**为什么失败？**

1. 索引经过多步非仿射变换：除法 → 位提取 → 类型转换
2. 最终索引无法表示为循环变量的仿射映射

**实际影响**：
- `scales` 保持单个存储体

---

### 案例 3: values[vidx]

**CADL 代码**：
```cadl
with idx: u32 = (0, idx + 1)
     vidx: u32 = (0, vidx_next)
do {
    let sparse_val: i8 = values[vidx];
    let inc_val: u32 = if is_nonzero {1} else {0};
    let vidx_next: u32 = vidx + inc_val;
}
```

**Partition 配置**：
```cadl
#[partition_factor_array([4])]   // 部分分区：4 个 bank
#[partition_cyclic_array([1])]   // 循环分区
```

**MLIR 表示**：
```mlir
%result = scf.for %idx = ... iter_args(%vidx = %c0) -> (i32) {
  %val = aps.memload %values[%vidx] : memref<256xi8>
  %inc = arith.select %cond, %c1, %c0 : i32
  %vidx_next = arith.addi %vidx, %inc : i32
  scf.yield %vidx_next : i32
}
```

**为什么失败？**

1. `vidx` 是 **loop-carried variable**（循环携带变量）
2. `vidx` 的值依赖于 `bitmask` 的数据（数据依赖）
3. 不能表示为 `idx` 的函数：`vidx ≠ f(idx)`（见前文数学证明）

**Bank 分配问题**：
- 循环分区：`bank = vidx % 4`
- 由于 `vidx` 是运行时值，编译器**无法静态确定**访问哪个 bank
- 即使能动态计算，编译器也无法**证明访问不会冲突**

**实际影响**：
- `values` 保持单个存储体（或分区失败）

---

## 不管 Warning 会怎样？

### 短期影响：✅ **没有致命问题**

1. **代码仍然正确**
   - Warning 不影响功能正确性
   - 生成的硬件行为与预期一致
   - 只是**性能优化**没有生效

2. **综合仍然成功**
   - HLS 工具（如 Vivado HLS、Catapult）仍然能综合代码
   - 生成的 RTL 是有效的

3. **Pragma 被忽略**
   - 数组不会被分区（或只应用了部分分区）
   - 保持为单个 BRAM/URAM

---

### 长期影响：⚠️ **性能损失**

#### 1. **存储体冲突（Bank Conflict）**

**未分区的数组**：
```
bitmask: 单个 64x8bit BRAM
  - 每周期只能读取 1 个元素
  - 如果循环展开（unroll），多个访问串行化
```

**分区后的数组**（假设能分区）：
```
bitmask: 64 个独立 BRAM（完全分区）
  - 每周期可以并行读取 64 个元素
  - 循环展开时，多个访问可并行
```

**性能对比（假设 unroll(8)）**：
| 场景 | 未分区 | 完全分区 |
|-----|--------|----------|
| 每周期访问数 | 1 | 8 |
| 总周期数 | 64 | 8 |
| 加速比 | 1x | **8x** |

---

#### 2. **流水线停顿（Pipeline Stall）**

HLS 工具在流水化循环时，需要确保：
- **读后写（RAW）无冲突**
- **写后读（WAR）无冲突**
- **写后写（WAW）无冲突**

**未分区**时：
```verilog
// 假设 II=1（每周期启动一次新迭代）
// 迭代 0: 读 bitmask[0]
// 迭代 1: 读 bitmask[1]  ← 需要等待 BRAM 空闲（冲突）
```
→ **Initiation Interval (II) > 1**（流水线停顿）

**分区后**：
```verilog
// 每个元素有独立 BRAM
// 迭代 0: 读 bank_0
// 迭代 1: 读 bank_1  ← 无冲突
```
→ **II = 1**（完美流水线）

**性能影响**：
- II=8 vs II=1 → **8x 延迟差异**

---

#### 3. **资源利用率低下**

**单个大 BRAM**：
```
bitmask: 64 字节 → 使用 1 个 BRAM18K
  - BRAM 利用率：64B / 2KB = 3.125%
  - 浪费剩余 1984 字节
```

**完全分区**：
```
bitmask: 64 个独立寄存器
  - 使用 64 个 8-bit FF（触发器）
  - BRAM 使用：0（释放 BRAM 资源）
  - FF 使用：+512 FF（通常充足）
```

**优势**：
- 小数组（如 `scales[32]`）完全分区后变为寄存器
- 释放宝贵的 BRAM 资源用于大数组

---

### 对 deca_decompress 的具体影响

假设你的目标 FPGA 是 **Xilinx UltraScale+ ZU9EG**：

| 数组 | 大小 | 未分区资源 | 分区后资源 | 性能影响 |
|-----|------|-----------|-----------|---------|
| `bitmask` | 64B | 1 BRAM | 64 FF (完全分区) | 无影响（访问已经非仿射） |
| `scales` | 64B | 1 BRAM | 32 FF (完全分区) | **中等**（可复用在组内） |
| `values` | 256B | 1 BRAM | 4 BRAM (factor=4) | **高**（稀疏访问，分区减少冲突） |
| `decompressed_weights` | 1KB | 1 BRAM | 4 BRAM (factor=4) | **高**（写入密集） |

**未分区的后果**：
1. **bitmask**: 由于访问本来就是非仿射的（`idx/8`），即使分区也无法并行 → 影响小
2. **scales**: 在嵌套循环方案中，每组读一次，分区可提速 → 影响中等
3. **values**: 稀疏访问，单 bank 会严重冲突 → **影响高**
4. **decompressed_weights**: 密集写入，单 bank 限制吞吐 → **影响高**

**估算性能损失**：
- 未分区的主循环 II：可能是 4-8（BRAM 冲突）
- 分区后的主循环 II：可能是 1-2
- **总体加速比**: 2x - 4x（仅通过分区）

---

## 什么时候可以忽略 Warning？

### ✅ 可以忽略的情况

1. **原型验证阶段**
   - 先验证功能正确性
   - 性能优化延后

2. **非性能关键路径**
   - 数组访问不在热点代码中
   - 占总执行时间 < 5%

3. **数组太小**
   - 如 `bitmask[64]`，分区收益有限
   - 访问本身就很快（几个周期）

4. **访问已经串行**
   - 循环没有展开（unroll=1）
   - 没有流水线指令（`#pragma HLS PIPELINE`）
   - 分区也无法并行 → 无收益

5. **BRAM 资源紧张**
   - 分区会增加 BRAM 使用
   - 如果 BRAM 使用率 > 80%，保持未分区反而好

---

### ⚠️ 必须解决的情况

1. **性能关键循环**
   - 主计算循环（如 512 次解压缩）
   - 带 `[[unroll(N)]]` 或 `[[pipeline(1)]]`

2. **并行访问需求**
   - 循环展开因子 > 1
   - 多个数组同时访问

3. **HLS 报告显示瓶颈**
   - II > 目标值
   - BRAM conflicts 警告
   - Latency 不符合要求

4. **已知访问模式可优化**
   - 如 `scales[group]`（可改为嵌套循环）
   - 如 `bitmask[byte_i]`（可改为双层循环）

---

## 如何判断 Warning 的严重性？

### 步骤 1: 查看 HLS 报告

综合后查看 Vivado HLS 报告：

```bash
# 查看 Latency Report
cat solution1/syn/report/deca_decompress_csynth.rpt

# 关键指标：
# 1. Loop Latency
# 2. Initiation Interval (II)
# 3. BRAM Usage
```

**关键问题**：
- II 是否达到目标？（通常希望 II=1）
- Latency 是否满足需求？
- BRAM 使用是否合理？

---

### 步骤 2: 比较分区前后

```cadl
// 实验 1: 去掉所有 partition pragma
// 实验 2: 保留 partition pragma（有 warning）
// 实验 3: 改写为仿射形式（无 warning）
```

对比三个版本的：
- **Latency**（延迟）
- **Throughput**（吞吐量）
- **Resource Usage**（资源使用）

**如果 实验1 和 实验2 的性能相同** → Warning 影响已经体现，无需担心

**如果 实验2 和 实验3 的性能差距大** → 值得修复

---

### 步骤 3: 使用 Co-Simulation 测量

```bash
# 运行 C/RTL Co-Simulation
vivado_hls -f run_cosim.tcl

# 测量实际执行时间
# 对比理论值 vs 实际值
```

**判断标准**：
- 实际 Latency / 理论 Latency < 2x → 可接受
- 实际 Latency / 理论 Latency > 5x → **必须优化**

---

## 实际建议

### 对于 deca_decompress.cadl

#### 方案 A: 接受现状（快速原型）

```cadl
// 保持当前代码不变
// 先验证功能正确性
// 性能够用就不管 warning
```

**适用场景**：
- 只需要功能验证（非性能敏感）
- BRAM 资源紧张（分区会增加使用）
- 时间紧迫（改写需要验证）

---

#### 方案 B: 部分修复（性能平衡）

只修复**收益最大**的部分：

1. **修复 scales** → 使用嵌套循环（见 DECA_AFFINE_TRANSFORMATION.md）
   - 代价：中等（增加嵌套）
   - 收益：**高**（32 次 scale 读取，复用友好）

2. **保留 bitmask 和 values** → 接受 warning
   - 原因：本质非仿射，改写复杂
   - 影响：中等（访问本来就不是并行的）

**预期效果**：
- Warning: 3 → 2
- 性能提升：20% - 40%（主要来自 scale 复用）

---

#### 方案 C: 完全修复（最优性能）

实施完整的三层嵌套循环（见 DECA_AFFINE_TRANSFORMATION.md）：

```cadl
with group: u32 = (0, group + 1) do {        // 外层：32 组
    with byte_in_group: u32 = (0, ...) do {  // 中层：2 字节/组
        [[unroll(8)]]
        with bit_i: u8 = (0, ...) do {       // 内层：8 位/字节
            // ...
        }
    }
}
```

**代价**：
- 代码复杂度：**高**
- 验证成本：需要详细测试

**收益**：
- Warning: 3 → 1（只剩 values）
- 性能提升：50% - 100%（bitmask 和 scales 都优化）
- 资源优化：scales 变为寄存器（释放 BRAM）

---

## 总结

### Warning 本质

- 编译器无法为数组访问**静态分配存储体**
- 导致 partition pragma **被忽略**
- 数组保持为**单个存储体**

### 影响程度

| 影响类型 | 严重性 | 说明 |
|---------|--------|------|
| 功能正确性 | ✅ 无影响 | 代码仍然正确 |
| 综合成功率 | ✅ 无影响 | 仍能生成硬件 |
| 性能（串行） | ⚠️ 小影响 | 未展开循环几乎无差 |
| 性能（并行） | ❌ **高影响** | 展开循环会严重受限 |
| 资源使用 | ⚠️ 中影响 | 可能浪费 BRAM |

### 决策树

```
是否在性能关键路径？
├─ 否 → ✅ 忽略 warning
└─ 是 → 循环是否展开/流水线？
    ├─ 否 → ✅ 忽略 warning
    └─ 是 → HLS 报告 II 是否达标？
        ├─ 是 → ✅ 忽略 warning
        └─ 否 → ❌ 必须修复（改写为仿射）
```

### 推荐行动

1. **立即**：先完成功能验证（接受 warning）
2. **短期**：运行 HLS 综合，查看 II 和 Latency
3. **中期**：如果性能不达标，实施部分修复（scales）
4. **长期**：如果需要极致性能，实施完全修复

**最重要的原则**：用**实测数据**而不是 warning 数量来指导优化！
