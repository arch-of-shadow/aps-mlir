# CADL 定制指令匹配方案（重新设计）

## 基于学长反馈的重新思考

### 核心认识转变

**之前的误解**：认为 CADL/APS MLIR 是"底层硬件细节"，需要从 C 语义开始。

**正确的理解**：CADL 本身就是"高层硬件语义"，比 C 更适合表达硬件定制指令。

---

## CADL vs C：为什么 CADL 更合适？

### 对比分析

| 特性 | C 语言 | CADL | 结论 |
|-----|--------|------|------|
| **控制流** | ✅ for/if/while | ✅ with/if/while | 等价 |
| **算术运算** | ✅ +/-/*// | ✅ +/-/*// | 等价 |
| **任意位宽** | ❌ 只有 int8/16/32/64 | ✅ u17, bit<5> | **CADL 胜** |
| **Burst 语义** | ❌ 只能写循环，需要识别 | ✅ `_burst_read[addr +: len]` | **CADL 胜** |
| **寄存器文件** | ❌ 无法表达 | ✅ `_irf[rs1]` | **CADL 胜** |
| **硬件属性** | ❌ 无法表达 | ✅ `#[impl("1rw")]` | **CADL 胜** |
| **位操作** | ❌ 笨拙的位掩码 | ✅ `data[5:10]` 直接切片 | **CADL 胜** |

**结论**：CADL 的表达力 > C 语言（在硬件建模方面）

### 关键例子：Burst 传输

#### C 语言的困境

```c
// 用 C 表达 burst：只能写循环
void vector_add_with_burst(int* cpu_a, int* cpu_b, int len) {
    int local_a[16];
    int local_b[16];

    // 1. "Burst" load（但编译器看不出来这是 burst）
    for (int i = 0; i < len; i++) {
        local_a[i] = cpu_a[i];  // 普通内存拷贝
        local_b[i] = cpu_b[i];
    }

    // 2. 计算
    for (int i = 0; i < len; i++) {
        local_a[i] = local_a[i] + local_b[i];
    }

    // 3. "Burst" store
    for (int i = 0; i < len; i++) {
        cpu_a[i] = local_a[i];
    }
}
```

**问题**：
- 编译器无法区分 burst 循环和计算循环
- 需要额外的 pattern 识别："这个循环是连续访问 → 可能是 burst"
- 信息丢失：用户想表达 DMA，但只能写成循环

#### CADL 的表达

```cadl
rtype vector_add_with_burst(rs1: u5, rs2: u5, rd: u5) {
    let cpu_a: u64 = _irf[rs1];
    let cpu_b: u64 = _irf[rs2];

    static mem_a: [u32; 16];
    static mem_b: [u32; 16];

    // 1. Burst load（语义明确！）
    mem_a[0 +: ] = _burst_read[cpu_a +: 16];
    mem_b[0 +: ] = _burst_read[cpu_b +: 16];

    // 2. 计算（和 C 一样）
    with i: u32 = (0, i_) do {
        let a: u32 = mem_a[i];
        let b: u32 = mem_b[i];
        mem_a[i] = a + b;
        let i_: u32 = i + 1;
    } while (i_ < 16);

    // 3. Burst store（语义明确！）
    _burst_write[cpu_a +: 16] = mem_a[0 +: ];

    _irf[rd] = 0;
}
```

**优势**：
- ✅ Burst 语义显式表达，无需猜测
- ✅ 寄存器文件访问清晰
- ✅ 内存属性可标注

---

## 重新定义："抽象" vs "Raise"

### 真正糟糕的 Raise（信息丢失）

```
汇编 → C 源码
├─ 丢失：寄存器分配信息
├─ 丢失：指令调度信息
└─ 结果：无法准确重建

LLVM IR → MLIR
├─ 丢失：优化历史
├─ 丢失：类型推导信息
└─ 结果：信息不可逆
```

### 合理的抽象（语义投影）

```
CADL/APS MLIR → 匹配层抽象
├─ 保留：计算逻辑（算法核心）
├─ 抽象：硬件接口（I/O 层）
└─ 结果：语义等价的匹配 pattern

类比：
  完整电影 → 预告片
  ├─ 丢失：具体剧情细节
  ├─ 保留：主题和核心情节
  └─ 目的：让观众识别"这是那部电影"
```

---

## 修正后的方案：两层架构（保留 CADL 优势）

### 架构设计

```
┌────────────────────────────────────────────────────────────────┐
│                   修正后的两层架构                              │
└────────────────────────────────────────────────────────────────┘

Layer 1: 完整定义（CADL → APS MLIR）
┌─────────────────────────────────────────────────────────────┐
│ 包含完整的硬件语义：                                         │
│ - aps.readrf/writerf（寄存器文件）                          │
│ - aps.memburstload/store（DMA 传输）                        │
│ - aps.memdeclare（硬件内存）                                │
│ - 计算核心（算法逻辑）                                       │
└─────────────────────────────────────────────────────────────┘
                    ↓ (抽象投影)
Layer 2: 匹配层（自动生成）
┌─────────────────────────────────────────────────────────────┐
│ 只保留计算核心：                                             │
│ - 移除 I/O 层（readrf, writerf, burst）                    │
│ - 抽象内存操作（aps.memload → memref.load）                │
│ - 保留控制流和算术（scf.for, arith.addi）                  │
└─────────────────────────────────────────────────────────────┘
                    ↓ (Pattern matching)
用户代码（标准 MLIR）
┌─────────────────────────────────────────────────────────────┐
│ 从 C 编译的标准 MLIR：                                       │
│ - memref.load/store                                         │
│ - scf.for                                                   │
│ - arith.addi                                                │
└─────────────────────────────────────────────────────────────┘
```

### 关键设计决策

#### 决策 1: Pattern 定义使用 CADL（完整语义）

**原因**：
1. ✅ CADL 表达力更强（burst, 寄存器, 位宽）
2. ✅ 无信息丢失（保留所有硬件特性）
3. ✅ 用户熟悉（硬件工程师已经在用 CADL）
4. ✅ 自然的开发流程（定义指令 → 测试 → 用于匹配）

**示例**：
```cadl
// patterns/vector_add.cadl - 完整的 CADL 定义
rtype vector_add_16(rs1: u5, rs2: u5, rd: u5) {
    let addr_a: u64 = _irf[rs1];
    let addr_b: u64 = _irf[rs2];

    static mem_a: [u32; 16];
    static mem_b: [u32; 16];

    // Burst load
    mem_a[0 +: ] = _burst_read[addr_a +: 16];
    mem_b[0 +: ] = _burst_read[addr_b +: 16];

    // 计算核心（匹配这部分！）
    with i: u32 = (0, i_) do {
        let a: u32 = mem_a[i];
        let b: u32 = mem_b[i];
        mem_a[i] = a + b;
        let i_: u32 = i + 1;
    } while (i_ < 16);

    // Burst store
    _burst_write[addr_a +: 16] = mem_a[0 +: ];
    _irf[rd] = 0;
}
```

#### 决策 2: 自动生成匹配层（抽象投影）

**自动抽象规则**：

```python
class CADLPatternAbstractor:
    """从 CADL/APS MLIR 生成匹配层抽象"""

    def abstract_for_matching(self, aps_pattern: MOperation) -> MOperation:
        """
        生成匹配层 pattern

        输入：完整的 APS MLIR（包含所有硬件细节）
        输出：匹配层 pattern（只保留计算核心）
        """

        # 阶段 1: 识别计算核心
        compute_core = self._extract_compute_core(aps_pattern)
        # 找到包含 scf.for/arith.* 但不包含 aps.readrf/writerf 的 block

        # 阶段 2: 抽象硬件内存操作
        abstracted_core = self._abstract_memory_ops(compute_core)
        # aps.memload → memref.load
        # aps.memstore → memref.store

        # 阶段 3: 调整函数签名
        match_pattern = self._create_match_function(abstracted_core)
        # 原始: (rs1: i5, rs2: i5, rd: i5)
        # 抽象: (mem_a: memref<16xi32>, mem_b: memref<16xi32>)

        return match_pattern

    def _extract_compute_core(self, func: MOperation) -> MBlock:
        """提取计算核心（去除 I/O 层）"""
        for block in func.get_blocks():
            # 跳过包含 I/O 操作的 statement
            if self._has_io_ops(block):
                continue

            # 跳过 burst 操作
            if self._has_burst_ops(block):
                continue

            # 保留计算逻辑
            if self._has_compute_ops(block):
                return block

        raise ValueError("No compute core found")

    def _has_io_ops(self, block: MBlock) -> bool:
        """检查是否包含 I/O 操作"""
        io_ops = {'aps.readrf', 'aps.writerf'}
        return any(op.name in io_ops for op in block.operations)

    def _has_burst_ops(self, block: MBlock) -> bool:
        """检查是否包含 Burst 操作"""
        burst_ops = {'aps.memburstload', 'aps.memburststore'}
        return any(op.name in burst_ops for op in block.operations)

    def _has_compute_ops(self, block: MBlock) -> bool:
        """检查是否包含计算操作"""
        compute_patterns = ['scf.for', 'scf.if', 'arith.']
        for op in block.operations:
            if any(pattern in op.name for pattern in compute_patterns):
                return True
        return False

    def _abstract_memory_ops(self, block: MBlock) -> MBlock:
        """抽象内存操作"""
        abstracted = block.clone()

        for op in abstracted.operations:
            # aps.memload → memref.load
            if op.name == 'aps.memload':
                new_op = self._create_memref_load(op)
                abstracted.replace_op(op, new_op)

            # aps.memstore → memref.store
            elif op.name == 'aps.memstore':
                new_op = self._create_memref_store(op)
                abstracted.replace_op(op, new_op)

        return abstracted
```

**生成的匹配层 pattern**：
```mlir
// 自动生成的匹配 pattern
func.func @vector_add_16_match(
  %mem_a: memref<16xi32>,
  %mem_b: memref<16xi32>
) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index

  // 只保留计算核心
  scf.for %i = %c0 to %c16 step %c1 {
    %a = memref.load %mem_a[%i] : memref<16xi32>
    %b = memref.load %mem_b[%i] : memref<16xi32>
    %sum = arith.addi %a, %b : i32
    memref.store %sum, %mem_a[%i] : memref<16xi32>
  }

  return
}
```

#### 决策 3: 匹配后保留完整定义（用于代码生成）

```python
class Skeleton:
    """定制指令的骨架结构"""

    instr_name: str
    root: SkeletonNode

    # 🆕 保存完整的 APS MLIR 定义
    full_definition: MOperation  # 包含所有硬件细节

    # 🆕 保存抽象的匹配 pattern
    match_pattern: MOperation    # 只包含计算核心

    # 现有字段
    leaf_patterns: Dict[str, Term]
    arg_vars: List
    # ...
```

**使用流程**：
```python
def build_ruleset_from_module(module: MModule):
    """从 CADL/APS MLIR 构建 ruleset"""

    abstractor = CADLPatternAbstractor()

    for func in module.get_functions():
        # 1. 保存完整定义
        full_definition = func

        # 2. 生成匹配层 pattern
        match_pattern = abstractor.abstract_for_matching(func)

        # 3. 从匹配 pattern 构建 skeleton
        skeleton, simple_pattern = _build_skeleton_from_func(match_pattern)

        # 4. 保存完整定义（用于后续代码生成）
        if skeleton:
            skeleton.full_definition = full_definition
            skeleton.match_pattern = match_pattern

        # 5. 生成 rewrite rules
        # ... (现有逻辑)
```

---

## 完整工作流程

### 步骤 1: 定义定制指令（CADL）

```cadl
// patterns/my_instructions.cadl

rtype vector_add_16(rs1: u5, rs2: u5, rd: u5) {
    // 完整的硬件定义（包含 I/O、DMA、计算）
    let addr_a: u64 = _irf[rs1];
    static mem_a: [u32; 16];
    mem_a[0 +: ] = _burst_read[addr_a +: 16];

    // 计算核心
    with i: u32 = (0, i_) do {
        let a: u32 = mem_a[i];
        let b: u32 = mem_b[i];
        mem_a[i] = a + b;
        let i_: u32 = i + 1;
    } while (i_ < 16);

    _burst_write[addr_a +: 16] = mem_a[0 +: ];
    _irf[rd] = 0;
}
```

### 步骤 2: CADL → APS MLIR（CADL 前端）

```bash
# CADL 编译器生成 APS MLIR
cadl-frontend patterns/my_instructions.cadl -o patterns/my_instructions.mlir
```

**生成的 APS MLIR**（包含完整硬件语义）：
```mlir
func.func @vector_add_16(%rs1: i5, %rs2: i5, %rd: i5) {
  %addr_a = aps.readrf %rs1 : i5 -> i64
  %mem_a = aps.memdeclare : memref<16xi32>
  aps.memburstload %addr_a, %mem_a[%c0], %c16 : ...

  scf.for %i = %c0 to %c16 step %c1 : i32 {
    %a = aps.memload %mem_a[%i] : ...
    %b = aps.memload %mem_b[%i] : ...
    %sum = arith.addi %a, %b : i32
    aps.memstore %sum, %mem_a[%i] : ...
  }

  aps.memburststore %mem_a[%c0], %addr_a, %c16 : ...
  aps.writerf %rd, %c0 : i5, i32
  return
}
```

### 步骤 3: Megg 自动生成匹配 Pattern

```bash
# Megg 接收 APS MLIR，自动生成匹配层
./megg-opt user_code.mlir \
  --custom-instructions patterns/my_instructions.mlir \
  -o optimized.mlir
```

**Megg 内部处理**：
```python
# 1. 加载 APS MLIR pattern
aps_pattern = load_mlir("patterns/my_instructions.mlir")

# 2. 自动生成匹配层
abstractor = CADLPatternAbstractor()
match_pattern = abstractor.abstract_for_matching(aps_pattern)

# 3. 构建 skeleton（用匹配层）
skeleton = build_skeleton_from_func(match_pattern)
skeleton.full_definition = aps_pattern  # 保存完整定义

# 4. Pattern matching（用匹配层）
matches = megg_egraph.match_skeleton(skeleton)

# 5. 替换为自定义指令（引用完整定义）
for match in matches:
    custom_instr = create_custom_instr(skeleton.full_definition, match)
    replace_region(match, custom_instr)
```

### 步骤 4: 代码生成（使用完整定义）

**Megg 输出**（标记了定制指令）：
```mlir
func.func @my_function(%a: memref<16xi32>, %b: memref<16xi32>) {
  // 使用完整的 APS MLIR 定义
  %result = "megg.custom_instr"(%a, %b) {
    instr_name = "vector_add_16",
    full_definition = @vector_add_16  // 引用完整定义
  } : (memref<16xi32>, memref<16xi32>) -> ()

  return
}
```

**后端处理**（可选，如果需要进一步 lowering）：
```bash
# 后端可以直接使用 full_definition 生成代码
aps-backend optimized.mlir --output-format cadl -o output.cadl
```

---

## 这个方案为什么合理？

### 1. 保留 CADL 的表达力

**不需要从 C 开始**，因为 CADL 本身就比 C 更适合表达硬件语义。

```
C 的问题：
  - Burst 只能写成循环（需要识别）
  - 无法表达寄存器文件
  - 位宽受限

CADL 的优势：
  - Burst 是原生语义
  - 寄存器文件是一等公民
  - 任意位宽支持
```

### 2. "抽象"不是"Raise"

**这不是逆向工程**，而是**语义投影**：

```
完整定义（CADL/APS MLIR）
    ↓ 投影
匹配层（计算核心）
    ↓ 匹配
用户代码（标准 MLIR）
```

类比：
- 完整定义 = 完整的食谱（包括采购、准备、烹饪、摆盘）
- 匹配层 = 核心烹饪步骤（只关心炒菜的过程）
- 用户代码 = 另一个食谱的核心步骤
- 匹配成功 = "这两道菜的烹饪方法一样！"

### 3. 信息无损

**完整定义始终保留**，只是在匹配时"投影"到计算核心：

```python
skeleton.full_definition  # 完整的 APS MLIR（所有硬件细节）
skeleton.match_pattern    # 匹配层（只有计算核心）

# 匹配用 match_pattern
# 代码生成用 full_definition
```

### 4. 符合编译器设计

**每一层都在 Lower**：

```
CADL 源码
    ↓ (CADL frontend lower)
APS MLIR（完整定义）
    ↓ (投影到匹配层，非 raise！)
匹配层 pattern
    ↓ (Pattern matching)
标记的 MLIR
    ↓ (Backend lower)
CADL 汇编 / 二进制
```

**"投影"不是"Raise"**：
- Raise = 从低层重建高层（信息丢失）
- 投影 = 从完整信息中提取子集（信息保留）

---

## 总结

### 学长的观点（正确）

1. **CADL 比 C 更适合表达硬件语义**
   - Burst 是原生概念，不是需要识别的循环
   - 位宽灵活，寄存器文件是一等公民

2. **不需要从 C 开始**
   - C 无法表达 burst、寄存器、位宽
   - 强行用 C 反而增加复杂度

3. **"Raise" 可以接受（如果是语义投影）**
   - 保留完整定义 (full_definition)
   - 投影到匹配层 (match_pattern)
   - 匹配成功后使用完整定义

### 修正后的方案

```
1. Pattern 定义：CADL（完整硬件语义）
   └─> 编译为 APS MLIR

2. Megg 自动生成匹配层（计算核心）
   └─> 抽象投影：去除 I/O，保留计算

3. Pattern matching（用匹配层）
   └─> 在用户的标准 MLIR 中查找

4. 代码生成（用完整定义）
   └─> 引用原始 APS MLIR 的所有硬件细节
```

### 关键理解

**这不是 "Raise"，而是 "语义投影"**：
- 完整定义始终存在（无信息丢失）
- 匹配层是投影视图（只看计算核心）
- 匹配成功后使用完整定义（恢复所有细节）

---

你觉得这个修正后的方案如何？我现在认同学长的观点，这个方向是合理的！