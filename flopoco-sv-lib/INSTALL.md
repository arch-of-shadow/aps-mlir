# FloPoCo-SV-Lib 安装与使用指南

本文档提供完整的依赖安装、构建和使用说明。

## 目录

- [概述](#概述)
- [快速开始](#快速开始)
- [使用 Pixi 构建 (推荐)](#使用-pixi-构建-推荐)
- [手动构建](#手动构建)
- [使用方法](#使用方法)
- [目录结构](#目录结构)
- [常见问题](#常见问题)

## 概述

flopoco-sv-lib 是一个 FloPoCo 浮点运算模块的 SystemVerilog 库，包含：
- 已生成的 Verilog 浮点运算模块（FMA、除法、平方根等）
- SystemVerilog 包装器
- 模块生成脚本

### 依赖关系

```
FloPoCo (浮点硬件生成器)
├── GMP/GMPXX (多精度算术)      ← pixi 提供
├── MPFR (多精度浮点)           ← pixi 提供
├── MPFI (多精度区间)           ← pixi 提供
├── LAPACK (线性代数)           ← pixi 提供
├── Boost (C++ 库)              ← pixi 提供
├── FLEX (词法分析)             ← pixi 提供
├── Sollya (函数逼近)           ← 自动克隆构建
└── PAGSuite (常数乘法优化)     ← 自动克隆构建

VHDL → Verilog 转换
├── Yosys                       ← OSS CAD Suite 提供
└── GHDL + ghdl-yosys-plugin    ← OSS CAD Suite 提供 (无需 sudo)
```

## 快速开始

如果只需使用已生成的模块，**无需安装任何依赖**：

```bash
git clone <repo-url> flopoco-sv-lib
cd flopoco-sv-lib

# 已生成的 Verilog 模块
ls flopoco/IEEEFMA/   # FMA
ls flopoco/FPDiv/     # 除法
ls flopoco/FPSqrt/    # 平方根
```

---

## 使用 Pixi 构建 (推荐)

pixi.toml 已配置所有依赖和构建任务，只需几个命令即可完成构建。

### 前提条件

安装 [Pixi](https://pixi.sh)：
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### 一键构建

```bash
# 克隆仓库
git clone <repo-url>
cd aps-mlir

# 安装 pixi 环境（自动安装所有 conda 依赖）
pixi install

# 安装 OSS CAD Suite（包含 GHDL + Yosys，无需 sudo）
pixi run setup-oss-cad-suite

# 构建 FloPoCo 及其依赖（自动克隆 + 构建 Sollya → PAGSuite → FloPoCo）
pixi run build-flopoco

# 验证安装
pixi run flopoco --help
pixi run ghdl --version
```

### 构建任务说明

| 任务 | 说明 | 依赖 |
|------|------|------|
| `setup-oss-cad-suite` | 下载 OSS CAD Suite (GHDL + Yosys) | - |
| `clone-sollya` | 克隆 Sollya 源码 | - |
| `clone-pagsuite` | 克隆 PAGSuite 源码 | - |
| `clone-flopoco` | 克隆 FloPoCo 源码 | - |
| `build-sollya` | 构建 Sollya | clone-sollya |
| `build-pagsuite` | 构建 PAGSuite | clone-pagsuite |
| `build-flopoco` | 构建 FloPoCo | clone-flopoco, build-sollya, build-pagsuite |

单独运行：
```bash
pixi run setup-oss-cad-suite  # 下载 GHDL + Yosys（约 400MB）
pixi run build-sollya         # 自动克隆 + 构建 Sollya
pixi run build-pagsuite       # 自动克隆 + 构建 PAGSuite
pixi run build-flopoco        # 自动克隆全部 + 构建全部
```

**注意**：如果目录已存在，克隆/下载步骤会自动跳过。

### pixi.toml 已提供的依赖

无需手动安装，pixi 自动管理：
- gmp, mpfr, mpfi, fplll, libxml2, boost
- flex, lapack, autoconf, automake, libtool
- yosys, verilator, cmake, make, ninja

---

## 手动构建

如果不使用 pixi，按以下步骤手动构建。

### 步骤 1: 安装系统依赖

#### Ubuntu/Debian

```bash
# 基础工具
sudo apt update
sudo apt install -y build-essential cmake git flex bison autoconf automake libtool

# FloPoCo 必需库
sudo apt install -y libgmp-dev libmpfr-dev libmpfi-dev libboost-all-dev \
    liblapack-dev libxml2-dev libfplll-dev

# VHDL 转 Verilog 工具
sudo apt install -y yosys ghdl ghdl-yosys-plugin

# Lint 工具 (可选)
sudo apt install -y verilator
```

### 步骤 2: 构建 Sollya

```bash
export INSTALL_PREFIX=$HOME/.local

# 克隆
git clone https://gitlab.inria.fr/sollya/sollya.git sollya-src
cd sollya-src

# 构建
./autogen.sh
./configure --prefix=$INSTALL_PREFIX
make -j$(nproc) || true

# 安装
mkdir -p $INSTALL_PREFIX/{lib,include}
cp .libs/libsollya.* $INSTALL_PREFIX/lib/
cp *.h $INSTALL_PREFIX/include/

# 环境变量
export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$INSTALL_PREFIX:$CMAKE_PREFIX_PATH

cd ..
```

### 步骤 3: 构建 PAGSuite

```bash
# 克隆
git clone https://gitlab.com/kumm/pagsuite.git
cd pagsuite

# 构建
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX
make -j$(nproc)

# 安装
mkdir -p $INSTALL_PREFIX/include/pagsuite
cp lib/*.so $INSTALL_PREFIX/lib/
cp ../paglib/inc/pagsuite/*.h $INSTALL_PREFIX/include/pagsuite/
cp ../rpag/inc/pagsuite/*.h $INSTALL_PREFIX/include/pagsuite/
cp ../oscm/inc/pagsuite/*.h $INSTALL_PREFIX/include/pagsuite/

cd ../..
```

### 步骤 4: 构建 FloPoCo

```bash
# 克隆
git clone https://gitlab.com/flopoco/flopoco.git
cd flopoco

# 构建
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=$INSTALL_PREFIX \
    -DPAG_PREFIX_DIR=$INSTALL_PREFIX
make -j$(nproc)

# 添加到 PATH
export PATH=$PWD/bin:$PATH

# 验证
flopoco --help

cd ../..
```

---

## 使用方法

### 1. 使用已生成的模块

```bash
cd flopoco-sv-lib

# 模块命名规则: <运算>_<精度><流水线级数>
# H = FP16, S = FP32, D = FP64
# 例如: IEEEFMA_S3 = FP32 FMA, 3级流水线

ls flopoco/IEEEFMA/   # FMA 模块
ls flopoco/FPDiv/     # 除法模块
ls flopoco/FPSqrt/    # 平方根模块
```

### 2. 重新生成模块

```bash
cd flopoco-sv-lib

# 使用 pixi
pixi run flopoco --help

# 生成所有模块（需要 flopoco + yosys + ghdl）
make gen-flopoco

# 生成 Bender.yml 并 lint
make all
```

### 3. 自定义模块生成

编辑 `flopoco/gen.py`：

```python
# 支持的精度
tasks = [
    {'type': 'H', 'exp': 5, 'frac': 10},   # FP16
    {'type': 'S', 'exp': 8, 'frac': 23},   # FP32
    {'type': 'D', 'exp': 11, 'frac': 52},  # FP64 (取消注释启用)
]

# 目标频率 (MHz) - 影响流水线深度
for frequency in [100, 150, 200, 250, 300, 400]:
    gen_fma(frequency, task)
```

### 4. 直接使用 FloPoCo

```bash
# 生成 FP32 FMA (200MHz)
flopoco allRegistersWithAsyncReset=1 IEEEFPFMA wE=8 wF=23 name=MyFMA frequency=200
# 输出: flopoco.vhdl

# 转换为 Verilog
yosys -m ghdl -p "ghdl --std=08 flopoco.vhdl -e MyFMA; write_verilog MyFMA.v"
```

### 5. FloPoCo 常用命令

| 运算 | 命令 | 说明 |
|------|------|------|
| FMA | `IEEEFPFMA wE=8 wF=23` | 融合乘加 |
| 除法 | `FPDiv wE=8 wF=23` | 浮点除法 |
| 平方根 | `FPSqrt wE=8 wF=23` | 浮点平方根 |
| 对数 | `FPLog wE=8 wF=23` | 浮点对数 |
| 指数 | `IEEEFPExp wE=8 wF=23` | 浮点指数 |
| 比较 | `FPComparator wE=8 wF=23` | 浮点比较 |

参数：
- `wE`: 指数位宽 (FP16=5, FP32=8, FP64=11)
- `wF`: 尾数位宽 (FP16=10, FP32=23, FP64=52)
- `frequency`: 目标频率 MHz
- `allRegistersWithAsyncReset=1`: 异步复位

---

## 目录结构

上传到 GitHub 的内容：
```
aps-mlir/
├── pixi.toml                   # Pixi 配置（依赖 + 构建任务）
└── flopoco-sv-lib/             # FloPoCo SV 库
    ├── flopoco/                # 已生成的 Verilog 模块
    │   ├── gen.py              # 模块生成脚本
    │   ├── IEEEFMA/            # FMA 模块
    │   ├── FPDiv/              # 除法模块
    │   ├── FPSqrt/             # 平方根模块
    │   └── ...
    ├── rtl/                    # SystemVerilog 包装器
    ├── Makefile                # 构建脚本
    ├── Bender.yml              # Bender 依赖管理
    └── INSTALL.md              # 本文档
```

运行 pixi 任务后自动生成：
```
aps-mlir/
├── oss-cad-suite/              # GHDL + Yosys (setup-oss-cad-suite)
├── flopoco/                    # FloPoCo 源码 (自动克隆)
├── sollya-src/                 # Sollya 源码 (自动克隆)
├── pagsuite/                   # PAGSuite 源码 (自动克隆)
├── sollya-install/             # Sollya 安装目录
└── pagsuite-install/           # PAGSuite 安装目录
```

---

## 常见问题

### Q: pixi run build-flopoco 失败？

```bash
# 检查 Sollya 是否构建成功
ls sollya-install/lib/libsollya.so

# 检查 PAGSuite 是否构建成功
ls pagsuite-install/lib/libpag.so

# 手动重新构建
pixi run build-sollya
pixi run build-pagsuite
pixi run build-flopoco
```

### Q: GHDL 找不到？

运行 OSS CAD Suite 安装任务：

```bash
pixi run setup-oss-cad-suite

# 验证
pixi run ghdl --version
```

### Q: flopoco 命令找不到？

```bash
# 使用 pixi 环境
pixi shell
flopoco --help

# 或直接运行
pixi run flopoco --help
```

### Q: 如何选择流水线深度？

- 更高频率 = 更多流水线级数 = 更高延迟但更高吞吐量
- FP32 FMA @ 200MHz 通常是 3 级流水线
- 根据目标 FPGA 频率选择合适的模块

---

## 参考链接

- FloPoCo: https://flopoco.org / https://gitlab.com/flopoco/flopoco
- Sollya: https://www.sollya.org / https://gitlab.inria.fr/sollya/sollya
- PAGSuite: https://gitlab.com/kumm/pagsuite
- Pixi: https://pixi.sh

## 许可证

- FloPoCo: 修改版 AGPL (生成代码为 LGPL)
- PAGSuite: 研究用途免费，商业用途需授权
- flopoco-sv-lib 硬件代码: Solderpad Hardware License 0.51
