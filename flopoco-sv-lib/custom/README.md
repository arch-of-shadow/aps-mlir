# FloPoCo 一键生成工具

基于 FloPoCo 5.0.0 的浮点硬件模块生成工具，支持从 VHDL 到 SystemVerilog 的完整转换流程。

## 快速开始

```bash
cd /home/xys/aps-mlir/flopoco-sv-lib/custom

# 查看帮助
make help

# 生成 FP32 FMA (IEEE格式, 200MHz)
make OP=IEEEFPFMA WE=8 WF=23 FREQUENCY=200

# 生成 FP32 加法器
make OP=FPAdd WE=8 WF=23 FREQUENCY=200

# 生成 FP16 除法器 (3级流水线)
make OP=FPDiv WE=5 WF=10 PIPELINE=3

# 查看当前配置
make info
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `OP` | FPAdd | 运算类型 |
| `WE` | 8 | 指数位宽 |
| `WF` | 23 | 尾数位宽 |
| `FREQUENCY` | 200 | 目标频率 (MHz)，自动计算流水线深度 |
| `PIPELINE` | - | 直接指定流水线级数 (优先于 FREQUENCY) |
| `NAME` | 自动 | 模块名，默认: `<OP>_w<WE>e<WF>f` |
| `OUT_DIR` | ./generated | 输出目录 |
| `TARGET` | Virtex6 | 目标 FPGA |

## 支持的运算符

### 浮点运算 (FloPoCo 格式)

| OP | 说明 | 输入 | 输出 |
|----|------|------|------|
| FPAdd | 浮点加法 | X, Y | R |
| FPMult | 浮点乘法 | X, Y | R |
| FPDiv | 浮点除法 | X, Y | R |
| FPSqrt | 平方根 | X | R |
| FPExp | 指数函数 | X | R |
| FPLog | 对数函数 | X | R |
| FPComparator | 比较器 | X, Y | gt, lt, eq |

### IEEE 754 格式运算

| OP | 说明 | 输入 | 输出 |
|----|------|------|------|
| **IEEEFPFMA** | 融合乘加 (a*b+c) | A, B, C, negateAB, negateC, RndMode | R |
| IEEEFPExp | 指数函数 | X | R |

### 格式转换

| OP | 说明 | 输入 | 输出 |
|----|------|------|------|
| InputIEEE | IEEE → FloPoCo | X (IEEE) | R (FloPoCo) |
| OutputIEEE | FloPoCo → IEEE | X (FloPoCo) | R (IEEE) |
| Fix2FP | 定点 → 浮点 | I (定点) | O (浮点) |
| FP2Fix | 浮点 → 定点 | I (浮点) | O (定点) |

## 常用浮点格式

| 格式 | WE | WF | 总位宽 |
|------|----|----|--------|
| FP16 (半精度) | 5 | 10 | 16 |
| FP32 (单精度) | 8 | 23 | 32 |
| FP64 (双精度) | 11 | 52 | 64 |

## 输出文件

```
generated/
├── IEEEFPFMA_w8e23f.vhdl   # 中间 VHDL (可删除)
├── IEEEFPFMA_w8e23f.v      # Verilog
└── IEEEFPFMA_w8e23f.sv     # SystemVerilog (带时间刻度)
```

## 生成流程

```
FloPoCo 生成 VHDL → Yosys+GHDL 转换 → Verilog → SystemVerilog
```

1. **gen-vhdl**: FloPoCo 生成 VHDL
2. **convert-verilog**: Yosys + GHDL 插件转换为 Verilog
3. **convert-sv**: 添加时间刻度和 Verilator lint 指令

## 批量生成

```bash
# 生成所有精度的某运算
make OP=FPAdd gen-fp16    # FP16
make OP=FPAdd gen-fp32    # FP32
make OP=FPAdd gen-fp64    # FP64
make OP=FPAdd gen-all-formats  # 全部

# 频率扫描
make OP=FPAdd gen-freq-sweep  # 100-400MHz
```

## 安装信息

| 组件 | 版本 | 路径 |
|------|------|------|
| FloPoCo | 5.0.0 | `/home/xys/aps-mlir/flopoco/build/bin/flopoco` |
| Sollya | 8.0 | `/home/xys/aps-mlir/sollya-install/` |
| PAGSuite | 2.1.0 | `/home/xys/aps-mlir/pagsuite-install/` |

## 依赖

- **FloPoCo 5.0.0** - 浮点硬件生成器
- **Sollya** - 多项式逼近库
- **PAGSuite** - 流水线加法图优化
- **Yosys + GHDL** - VHDL 到 Verilog 转换

## 预生成模块

已有大量预生成模块位于 `/home/xys/aps-mlir/flopoco-sv-lib/flopoco/`:

```
IEEEFMA/   - FP16: H1-H5, FP32: S1-S6 (FMA)
FPDiv/     - FP16: H1-H6, FP32: S2-S12 (除法)
FPSqrt/    - FP16: H1-H5, FP32: S2-S12 (平方根)
FPLog/     - FP16: H1-H7, FP32: S2-S9 (对数)
IEEEExp/   - FP16: H1-H6, FP32: S3-S9 (指数)
Fix2FP/    - FP16: H0-H2, FP32: S0-S2 (定点→浮点)
FP2Fix/    - FP16: H0-H1, FP32: S0-S1 (浮点→定点)
IEEE2FP/   - FP16: H0, FP32: S0 (IEEE→FloPoCo)
FP2IEEE/   - FP16: H0, FP32: S0 (FloPoCo→IEEE)
FPComp/    - FP16: H0, FP32: S0 (比较器)
```

命名规则: `<运算>_<精度><流水线级数>` (H=FP16, S=FP32)

## 参考链接

- [FloPoCo 官网](https://flopoco.org/)
- [FloPoCo GitLab](https://gitlab.inria.fr/flopoco/flopoco)
- [Sollya](https://sollya.org/)
- [PAGSuite](https://gitlab.com/kumm/pagsuite)
