# tests/diff_match 纯计算模式建议

## 概述

本文档提出一组**纯计算模式**（无控制流），用于补充 `tests/diff_match` 中现有的控制流密集型模式。这些模式旨在：

1. **充分利用 e-graph 能力**：代数化简、公共子表达式消除（CSE）、强度削减
2. **代表经典 ASIP 应用**：DSP、图形、机器学习加速器中的常见模式
3. **适度简单**：易于理解但足够非平凡以展示价值
4. **填补当前测试覆盖的空白**：专注于纯计算模式

---

## 现状分析

### 现有 tests/diff_match 模式
所有当前模式都是**控制流密集型**的：
- `v3ddist_vs`: 循环 + 3D 距离计算（平方差之和）
- `v3ddist_vv`: 向量-向量变体
- `vcovmat3d_vs`: 带循环的协方差矩阵计算
- `vgemv3d`: 嵌套循环的矩阵-向量乘法

### 现有 tests/match/match_cases 模式
存在简单模式但都很基础：
- `01-arithmetic`: 简单的加/减/乘链
- `02-bitwise`: and/or/xor 操作
- `03-mux`: 比较 + 选择

### 发现的空白
**缺失**：能够展示 e-graph 在以下方面优势的经典计算模式：
- 识别代数等价形式
- 跨复杂表达式的公共子表达式消除
- 强度削减（例如 `x*2` → `x<<1`）
- 跨同一算法不同实现的模式匹配

---

## 提出的模式分类

### 类别 1: 融合乘加变体 ⭐ **最高优先级**

**理由**：FMA 是 DSP/ML/图形 ASIP 中最常见的自定义指令

#### 模式 1.1: 简单 FMA（融合乘加）
```mlir
func.func @fma(%a: i32, %b: i32, %c: i32) -> i32 {
  %mul = arith.muli %a, %b : i32
  %add = arith.addi %mul, %c : i32
  return %add : i32
}
```
**应用场景**：`result = a * b + c`（矩阵运算、卷积中无处不在）

**E-graph 优势**：即使代码计算 `c + a*b` 也能匹配（交换律重写）

---

#### 模式 1.2: MAC（乘累加）
```mlir
func.func @mac(%acc: i32, %a: i32, %b: i32) -> i32 {
  %mul = arith.muli %a, %b : i32
  %new_acc = arith.addi %acc, %mul : i32
  return %new_acc : i32
}
```
**应用场景**：基于累加器的操作（点积、卷积）

**与 FMA 的区别**：语义上强调累加模式

---

#### 模式 1.3: MSUB（乘减）
```mlir
func.func @msub(%a: i32, %b: i32, %c: i32) -> i32 {
  %mul = arith.muli %a, %b : i32
  %sub = arith.subi %mul, %c : i32
  return %sub : i32
}
```
**应用场景**：`result = a * b - c`（数值方法中常见）

**E-graph 优势**：可以从 `c - a*b` 重写为 `-(a*b - c)`

---

#### 模式 1.4: NMSUB（负乘减）
```mlir
func.func @nmsub(%a: i32, %b: i32, %c: i32) -> i32 {
  %mul = arith.muli %a, %b : i32
  %sub = arith.subi %c, %mul : i32  // c - a*b
  return %sub : i32
}
```
**应用场景**：`result = c - a * b`（FIR 滤波器、误差计算）

---

### 类别 2: 算术优化模式 ⭐ **最高优先级**

**理由**：展示强度削减和代数恒等式

#### 模式 2.1: 平均值（算术平均）
```mlir
func.func @avg_shift(%a: i32, %b: i32) -> i32 {
  %sum = arith.addi %a, %b : i32
  %c1 = arith.constant 1 : i32
  %avg = arith.shrui %sum, %c1 : i32  // (a+b) >> 1
  return %avg : i32
}
```
**应用场景**：无需除法的快速平均（图像处理、插值）

**E-graph 优势**：通过强度削减重写可以匹配等价的 `(a+b)/2`

**替代形式**（测试代数等价）：
```mlir
func.func @avg_div(%a: i32, %b: i32) -> i32 {
  %sum = arith.addi %a, %b : i32
  %c2 = arith.constant 2 : i32
  %avg = arith.divui %sum, %c2 : i32
  return %avg : i32
}
```

---

#### 模式 2.2: 平方差
```mlir
func.func @sqdiff(%a: i32, %b: i32) -> i32 {
  %diff = arith.subi %a, %b : i32
  %sq = arith.muli %diff, %diff : i32
  return %sq : i32
}
```
**应用场景**：MSE 计算、距离度量（v3ddist 中使用但在循环内）

**E-graph 优势**：
- 如果 `%diff` 在其他地方使用，CSE
- 可以展开为 `a*a + b*b - 2*a*b` 并匹配该形式

**展开的代数形式**（测试等价检测）：
```mlir
func.func @sqdiff_expanded(%a: i32, %b: i32) -> i32 {
  %a2 = arith.muli %a, %a : i32
  %b2 = arith.muli %b, %b : i32
  %ab = arith.muli %a, %b : i32
  %c2 = arith.constant 2 : i32
  %ab2 = arith.muli %ab, %c2 : i32
  %sum = arith.addi %a2, %b2 : i32
  %result = arith.subi %sum, %ab2 : i32
  return %result : i32
}
```

---

#### 模式 2.3: 3 项点积（小向量点积）
```mlir
func.func @dot3(%a0: i32, %a1: i32, %a2: i32,
                %b0: i32, %b1: i32, %b2: i32) -> i32 {
  %p0 = arith.muli %a0, %b0 : i32
  %p1 = arith.muli %a1, %b1 : i32
  %p2 = arith.muli %a2, %b2 : i32
  %sum01 = arith.addi %p0, %p1 : i32
  %sum012 = arith.addi %sum01, %p2 : i32
  return %sum012 : i32
}
```
**应用场景**：3D 向量点积（图形、物理）

**E-graph 优势**：
- 乘法的 CSE
- 可以匹配不同的结合顺序：`(p0+p1)+p2` vs `p0+(p1+p2)`

---

#### 模式 2.4: 线性插值（LERP）
```mlir
func.func @lerp(%a: i32, %b: i32, %t: i32) -> i32 {
  %diff = arith.subi %b, %a : i32
  %scaled = arith.muli %diff, %t : i32
  %result = arith.addi %a, %scaled : i32
  return %result : i32  // a + t*(b-a)
}
```
**应用场景**：动画、颜色插值、数值方法

**E-graph 优势**：可以匹配等价形式：
- `a + t*b - t*a`（分配律）
- `(1-t)*a + t*b`（重写后）

---

### 类别 3: 位运算计算模式

**理由**：嵌入式系统、加密、数据处理中常见

#### 模式 3.1: 清除最低置位位
```mlir
func.func @clear_lowest_bit(%x: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %x_minus_1 = arith.subi %x, %c1 : i32
  %result = arith.andi %x, %x_minus_1 : i32
  return %result : i32  // x & (x-1)
}
```
**应用场景**：位操作、人口计数算法

**E-graph 优势**：经典位操作惯用法识别

---

#### 模式 3.2: 隔离最低置位位
```mlir
func.func @isolate_lowest_bit(%x: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %neg_x = arith.subi %c0, %x : i32  // -x（补码）
  %result = arith.andi %x, %neg_x : i32
  return %result : i32  // x & (-x)
}
```
**应用场景**：快速位操作、位上的二分搜索

---

#### 模式 3.3: 字节交换（16 位）
```mlir
func.func @bswap16(%x: i32) -> i32 {
  %c8 = arith.constant 8 : i32
  %c0xff = arith.constant 255 : i32
  %c0xff00 = arith.constant 65280 : i32

  %low_byte = arith.andi %x, %c0xff : i32
  %high_byte = arith.andi %x, %c0xff00 : i32

  %low_shifted = arith.shli %low_byte, %c8 : i32
  %high_shifted = arith.shrui %high_byte, %c8 : i32

  %result = arith.ori %low_shifted, %high_shifted : i32
  return %result : i32
}
```
**应用场景**：字节序转换、网络协议

**E-graph 优势**：具有多个子表达式的复杂模式

---

### 类别 4: 饱和与截断模式

**理由**：信号处理和图像/视频编解码器中常见

#### 模式 4.1: 饱和加法（带 min/max）
```mlir
func.func @sat_add(%a: i32, %b: i32, %max_val: i32) -> i32 {
  %sum = arith.addi %a, %b : i32
  %cmp = arith.cmpi ult, %sum, %max_val : i32
  %result = arith.select %cmp, %sum, %max_val : i32
  return %result : i32  // min(a+b, max_val)
}
```
**应用场景**：定点算术、音频处理

---

#### 模式 4.2: 截断（到范围）
```mlir
func.func @clamp(%x: i32, %min_val: i32, %max_val: i32) -> i32 {
  %cmp_min = arith.cmpi slt, %x, %min_val : i32
  %clamped_low = arith.select %cmp_min, %min_val, %x : i32

  %cmp_max = arith.cmpi sgt, %clamped_low, %max_val : i32
  %result = arith.select %cmp_max, %max_val, %clamped_low : i32
  return %result : i32  // clamp(x, min, max)
}
```
**应用场景**：颜色量化、传感器数据归一化

**E-graph 优势**：两阶段选择模式（无 scf.if 的嵌套条件）

---

### 类别 5: 多项式求值模式

**理由**：数学函数近似、校验和

#### 模式 5.1: Horner 方法（2 次）
```mlir
func.func @horner_deg2(%x: i32, %c0: i32, %c1: i32, %c2: i32) -> i32 {
  // 计算：c2*x^2 + c1*x + c0
  // Horner 形式：c0 + x*(c1 + x*c2)
  %inner = arith.muli %x, %c2 : i32
  %inner_plus_c1 = arith.addi %inner, %c1 : i32
  %outer = arith.muli %x, %inner_plus_c1 : i32
  %result = arith.addi %outer, %c0 : i32
  return %result : i32
}
```
**应用场景**：快速多项式求值（泰勒级数、校验和）

**E-graph 优势**：可以匹配 Horner 和朴素 `c0 + c1*x + c2*x*x` 两种形式

---

#### 模式 5.2: CRC 风格 XOR 链
```mlir
func.func @crc_step(%crc: i32, %data: i32, %poly: i32) -> i32 {
  %xor1 = arith.xori %crc, %data : i32
  %c1 = arith.constant 1 : i32
  %shifted = arith.shrui %xor1, %c1 : i32
  %xor2 = arith.xori %shifted, %poly : i32
  return %xor2 : i32
}
```
**应用场景**：校验和、错误检测码

---

### 类别 6: 专用 DSP 模式

**理由**：信号处理 ASIP 中常见

#### 模式 6.1: 绝对差
```mlir
func.func @abs_diff(%a: i32, %b: i32) -> i32 {
  %diff = arith.subi %a, %b : i32
  %c0 = arith.constant 0 : i32
  %is_neg = arith.cmpi slt, %diff, %c0 : i32
  %neg_diff = arith.subi %c0, %diff : i32
  %result = arith.select %is_neg, %neg_diff, %diff : i32
  return %result : i32  // |a - b|
}
```
**应用场景**：视频编码中的 SAD（绝对差之和）

---

#### 模式 6.2: 符号扩展（8 位到 32 位）
```mlir
func.func @sign_extend_i8(%x: i32) -> i32 {
  %c0xff = arith.constant 255 : i32
  %c0x80 = arith.constant 128 : i32
  %c0xffffff00 = arith.constant -256 : i32

  %low_byte = arith.andi %x, %c0xff : i32
  %is_neg = arith.cmpi uge, %low_byte, %c0x80 : i32
  %extended = arith.ori %low_byte, %c0xffffff00 : i32
  %result = arith.select %is_neg, %extended, %low_byte : i32
  return %result : i32
}
```
**应用场景**：数据类型转换、打包算术

---

## 推荐实现优先级

### 第一梯队：必须实现（立即价值）⭐⭐⭐
1. **模式 1.1**：FMA（最常见的 ASIP 指令）
2. **模式 1.2**：MAC（累加器变体）
3. **模式 2.2**：平方差（补充现有 v3ddist）
4. **模式 2.3**：3 项点积（常见向量操作）

**理由**：这些在实际 ASIP 设计中无处不在，展示明确价值

---

### 第二梯队：高价值（良好演示）⭐⭐
5. **模式 2.1**：平均值（强度削减展示）
6. **模式 2.4**：LERP（代数等价）
7. **模式 4.2**：截断（嵌套选择模式）
8. **模式 6.1**：绝对差（DSP 常见）

**理由**：展示不同的 e-graph 能力（强度削减、等价性、嵌套模式）

---

### 第三梯队：锦上添花（完整性）⭐
9. **模式 3.1**：清除最低位（经典位技巧）
10. **模式 5.1**：Horner 方法（多项式求值）
11. **模式 1.3**：MSUB（FMA 变体）

**理由**：教育价值，展示模式多样性

---

## 实现指南

### 目录结构
```
tests/diff_match/
├── fma/                    # 模式 1.1
│   ├── fma.c
│   ├── fma.mlir           # 模式定义
│   ├── fma.cadl
│   ├── fma.json
│   └── compile.sh
├── mac/                    # 模式 1.2
├── sqdiff/                 # 模式 2.2
├── dot3/                   # 模式 2.3
└── ...
```

### 文件模板

**示例：`fma.c`**
```c
#include <stdint.h>

uint8_t fma(uint32_t *rs1, uint32_t *rs2) {
    uint32_t a = rs1[0];
    uint32_t b = rs1[1];
    uint32_t c = rs1[2];

    // 模式：a * b + c
    uint32_t result = a * b + c;

    rs2[0] = result;
    return 0;
}
```

**示例：`fma.cadl`（最小化）**
```cadl
instruction fma (rs1: uint32[3], rs2: uint32[1]) -> uint8 {
    let a = rs1[0];
    let b = rs1[1];
    let c = rs1[2];
    rs2[0] = a * b + c;
    return 0u8;
}
```

**示例：`fma.json`**（编码）
```json
{
  "fma": {
    "opcode": "0001011",
    "funct3": "000",
    "funct7": "0000001"
  }
}
```

---

## E-graph 优势分析

### 每个模式如何展示 E-graph 价值

| 模式 | E-graph 能力 | 演示内容 |
|------|-------------|---------|
| FMA | 交换律 | 匹配 `a*b+c` 和 `c+a*b` |
| MAC | CSE | 重用中间结果 `%mul` |
| 平方差 | 代数展开 | `(a-b)²` ↔ `a²+b²-2ab` |
| Dot3 | 结合律 | 不同求和顺序 |
| 平均值 | 强度削减 | `(a+b)/2` ↔ `(a+b)>>1` |
| LERP | 分配律 | `a+t*(b-a)` ↔ `a+t*b-t*a` |
| 截断 | 嵌套模式 | 组合选择链 |
| 绝对差 | CSE + 选择 | 分支中的公共子表达式 |

---

## 测试策略

### 测试用例结构
对于每个模式，创建：
1. **精确匹配**：代码与模式相同
2. **代数变体**：等价但形式不同（测试 e-graph 重写）
3. **CSE 变体**：嵌入有共享子表达式的模式

**FMA 示例**：

**测试 1**：`fma_exact.mlir`（与模式相同）
```mlir
%mul = arith.muli %a, %b : i32
%add = arith.addi %mul, %c : i32
```

**测试 2**：`fma_commuted.mlir`（测试交换律）
```mlir
%mul = arith.muli %a, %b : i32
%add = arith.addi %c, %mul : i32  // c + a*b 而不是 a*b + c
```

**测试 3**：`fma_with_cse.mlir`（测试 CSE）
```mlir
%mul = arith.muli %a, %b : i32
%add1 = arith.addi %mul, %c : i32   // FMA
%add2 = arith.addi %mul, %d : i32   // 重用 %mul
```

---

## 成功指标

当满足以下条件时，模式成功实现：

1. ✅ **模式提取**：Megg 正确构建骨架/模式
2. ✅ **精确匹配**：相同代码 100% 匹配
3. ✅ **代数匹配**：等价形式匹配（重写后）
4. ✅ **CSE 优势**：正确处理共享子表达式
5. ✅ **RISC-V 生成**：生成正确的 `.insn` 编码
6. ✅ **反汇编验证**：`.asm` 中可见自定义指令

---

## 与现有测试的集成

### 与 tests/match/match_cases 的关系
- **match_cases**：测试*匹配引擎*正确性（控制流、边界情况）
- **diff_match**：测试*实际 ASIP 模式*的端到端编译

### 与现有 diff_match 模式的关系
- **当前**：控制流密集型（循环 + 计算）
- **提议**：纯计算（无循环，利用代数重写）
- **互补**：一起提供 ASIP 模式类型的完整覆盖

---

## 未来扩展

一旦建立纯计算模式：

1. **混合模式**：将计算模式与最小控制流结合
   - 示例：简单 `scf.if` 内的 FMA（谓词执行）

2. **链式模式**：按顺序多个计算模式
   - 示例：`FMA → LERP → CLAMP` 流水线

3. **SIMD 变体**：标量模式的向量版本
   - 示例：在向量上操作的 `vec_fma`

4. **浮点变体**：FP32/FP64 版本
   - 示例：用于 `f32` FMA 的 `fmaf`

---

## 结论

这些纯计算模式将：

✅ **填补空白**：当前测试覆盖中的纯计算模式
✅ **展示 e-graph 优势**：代数重写、CSE、强度削减
✅ **代表真实 ASIP 需求**：FMA、MAC、点积无处不在
✅ **易于实现**：无控制流复杂性
✅ **具有教育意义**：Horner 等经典算法、位技巧

**推荐起点**：首先实现**第一梯队**模式（FMA、MAC、平方差、Dot3）以建立价值，然后根据需求扩展到第二/第三梯队。

---

**文档版本**：1.0
**日期**：2025-01-09
**状态**：建议（等待审查）
