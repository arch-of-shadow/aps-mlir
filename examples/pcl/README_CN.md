# PCL加速器指令集架构（ISA）设计

## 概述

本目录包含针对点云处理（Point Cloud Library, PCL）算法的自定义RISC-V扩展指令集。所有指令均采用CADL（计算机架构描述语言）实现，支持硬件综合与优化。

## 指令集列表

### 1. VGEMV3D - 3D矩阵-向量乘法
**操作码**: `opcode=0x2B, funct7=0x31`
**功能**: 计算4×4矩阵与4D齐次向量的乘积
**公式**: `y = M × x`，其中M为4×4矩阵，x为4D向量

**应用场景**:
- 3D仿射变换（平移、旋转、缩放）
- 投影变换（透视投影、正交投影）
- 坐标系转换

**关键特性**:
- 完全阵列分割：矩阵16个元素、向量4个元素全部并行访问
- 双层循环展开实现并行计算
- 使用全局累加器避免循环依赖

---

### 2. V3DDIST.VV - 向量模式3D距离计算
**操作码**: `opcode=0x2B, funct7=0x28`
**功能**: 计算16对3D点之间的欧氏距离平方
**公式**: `dist² = (x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²`

**应用场景**:
- 点云配准（ICP算法）
- K近邻搜索
- 点云分割

**关键特性**:
- SOA内存布局（Structure-of-Arrays）: X、Y、Z坐标分别连续存储
- 循环分割因子=4，支持向量化并行计算
- Burst传输最小化内存访问延迟
- 内存布局: `[X: 0+64] [Y: 64+64] [Z: 128+64]`，每个数组占用64字节（2^6，16×u32）

---

### 3. V3DDIST.VS - 向量-标量模式3D距离计算
**操作码**: `opcode=0x2B, funct7=0x29`
**功能**: 计算16个点到单个参考点的欧氏距离平方
**公式**: `dist² = (x-ref_x)² + (y-ref_y)² + (z-ref_z)²`

**应用场景**:
- 最远点采样（FPS, Farthest Point Sampling）
- 半径搜索
- 离群点检测

**关键特性**:
- 标量广播：参考点坐标复用于所有向量元素
- 减少内存访问：参考点仅读取一次
- 部分循环展开（unroll=4）平衡面积与性能

---

### 4. VCOVMAT3D - 3D协方差矩阵计算
**操作码**: `opcode=0x2B, funct7=0x30`
**功能**: 计算两个3D点差值的外积（3×3对称矩阵）
**公式**: `C = (p₁-p₂)(p₁-p₂)ᵀ`

**应用场景**:
- 法向量估计（主成分分析）
- 曲面特征提取
- 点云平滑与去噪

**关键特性**:
- 仅存储上三角6个元素（利用对称性）
- 完全分割：6个输入、6个输出同时并行访问
- 单次Burst传输减少延迟

---

### 5. VFPSMAX - 向量最大值归约与索引跟踪
**操作码**: `opcode=0x2B, funct7=0x2A`
**功能**: 找到向量中的最大值及其索引
**公式**: `(max_val, max_idx) = argmax(v[0..15])`

**应用场景**:
- 最远点采样（FPS）算法核心
- 优先级队列操作
- 特征点选择

**关键特性**:
- 4级树状归约结构（16→8→4→2→1）
- 双缓冲避免读写冲突
- 哨兵值处理（值=0表示已采样点）
- 完全展开实现O(log N)延迟

---

## 硬件优化技术

### 1. 阵列分割（Array Partitioning）
- **完全分割**: 所有元素映射到独立存储单元（寄存器）
- **循环分割**: 元素按循环方式分配到多个存储Bank
- **目标**: 消除结构性冒险，实现真并行访问

### 2. Burst传输
- 连续地址传输仅产生一次延迟
- 减少内存访问开销90%以上
- SOA布局优化Burst效率
- **约束**: Burst传输长度必须为2^n字节，因此数组间预留空隙地址以防越界写

### 3. 循环展开
- **完全展开**: 消除循环控制开销，暴露指令级并行
- **部分展开**: 平衡硬件资源与性能
- **Affine分析**: 编译器自动推导内存访问模式

### 4. 数据流优化
- 流水线寄存器插入减少关键路径
- 双缓冲消除数据依赖
- 累加器复用减少寄存器压力

---

## MLIR编译流程

```
CADL源码
  ↓ [cadl_frontend/parser.py]
AST抽象语法树
  ↓ [mlir_converter.py]
MLIR IR (SCF dialect)
  ↓ [raise-scf-to-affine]
Affine dialect
  ↓ [infer-affine-mem-access (多维affine索引推导)]
Affine load/store
  ↓ [hls-unroll (循环展开)]
展开后的affine.for
  ↓ [new-array-partition (阵列分割)]
分割后的memref (多Bank)
  ↓ [CIRCT lowering]
RTL硬件描述 (Verilog)
```

### 关键Pass说明
- **InferAffineMemAccess**: 自动推导多维affine访问模式（如`i*4+j`）
- **HlsUnroll**: 基于`[[unroll(N)]]`指令展开循环
- **NewArrayPartition**: 根据pragma将数组分割为并行Bank

---

## 技术指标

| 指令 | 向量长度 | 展开因子 | 分割因子 | 理论峰值吞吐 |
|------|----------|----------|----------|--------------|
| VGEMV3D | 4 | 4×4 | 16+4+4 | 16 MAC/周期 |
| V3DDIST.VV | 16 | 4 | 4×7 | 4 点对/周期 |
| V3DDIST.VS | 16 | 4 | 4×4 | 4 点/周期 |
| VCOVMAT3D | 1 | - | 6+6 | 6 MAC/周期 |
| VFPSMAX | 16 | 4 | 4×5 | log₂(16)=4周期 |

---

## 验证状态

✅ 所有指令通过MLIR编译流程
✅ 阵列分割成功（无警告）
✅ Affine循环分析通过
✅ 生成硬件可综合代码

---

## 参考文献

- Point Cloud Library (PCL) Documentation
- RISC-V Vector Extension Specification v1.0
- MLIR Affine Dialect Documentation
- Vivado HLS Array Partitioning Guide
