# FloPoCo VHDL 到 SystemVerilog 转换指南

本文档描述如何使用 flopoco-sv-lib 工具从 FloPoCo 生成 VHDL 并转换为 Verilog/SystemVerilog。

## 目录结构

```
flopoco-sv-lib/
├── flopoco/                    # FloPoCo 生成和转换工具
│   ├── gen.py                  # 主生成脚本
│   ├── rename.tcl              # Yosys 模块重命名脚本
│   ├── IEEEFMA/                # 生成的 FMA 模块
│   ├── FPDiv/                  # 生成的除法模块
│   ├── FPSqrt/                 # 生成的平方根模块
│   └── ...                     # 其他生成的模块
├── rtl/                        # SystemVerilog wrapper
│   ├── wrappers/               # FloPoCo 模块的 SV 包装器
│   └── *.sv                    # IEEE-754 辅助模块
├── Makefile                    # 构建脚本
└── Bender.yml                  # Bender 依赖管理
```

## 完整工作流程

### 步骤 1: 安装依赖

```bash
# 安装 FloPoCo
sudo apt install flopoco

# 安装 Yosys + GHDL 插件 (用于 VHDL → Verilog 转换)
sudo apt install yosys ghdl ghdl-yosys-plugin
```

### 步骤 2: 使用 FloPoCo 生成 VHDL

FloPoCo 命令格式：
```bash
flopoco [选项] <运算类型> wE=<指数位宽> wF=<尾数位宽> name=<模块名> frequency=<目标频率>
```

示例 - 生成 FP32 FMA (目标 200MHz):
```bash
flopoco allRegistersWithAsyncReset=1 IEEEFPFMA wE=8 wF=23 name=IEEEFMA_S frequency=200
# 输出: flopoco.vhdl
```

**关键参数**:
| 参数 | 说明 |
|------|------|
| `wE` | 指数位宽 (FP16=5, FP32=8, FP64=11) |
| `wF` | 尾数位宽 (FP16=10, FP32=23, FP64=52) |
| `frequency` | 目标频率 MHz (影响流水线深度) |
| `allRegistersWithAsyncReset=1` | 使用异步复位寄存器 |

### 步骤 3: 解析流水线深度

FloPoCo 输出会包含流水线信息:
```
Pipeline depth = 3, register count = 156
# 或格式: c3, xxx registers
```

gen.py 脚本自动解析此信息:
```python
stages = out.splitlines()[-1]
stages = stages.split(',')[0].split('c')[1]  # 提取流水线级数
```

### 步骤 4: VHDL → Verilog 转换 (Yosys + GHDL)

使用 Yosys 和 GHDL 插件进行转换:

```tcl
# run.tcl - Yosys 转换脚本
yosys -import;
ghdl --std=08 -fsynopsys -fexplicit IEEEFMA_S3.vhdl -e IEEEFMA_S;
hierarchy -top IEEEFMA_S;
yosys rename -top IEEEFMA_S3;
write_verilog IEEEFMA/IEEEFMA_S3.v;
```

执行:
```bash
yosys -m ghdl -C run.tcl
```

### 步骤 5: 模块名后缀处理

rename.tcl 脚本为所有子模块添加后缀，避免命名冲突:

```tcl
# 获取模块列表
yosys tee -q -o "modules.rpt" ls

# 为每个模块名添加后缀
foreach module $modules {
    exec sed -i -e "s/${module}/${module}${suffix}/g" $file_name
}
```

### 步骤 6: 添加 Verilator 兼容性注释

gen.py 最后一步为生成的 Verilog 添加 lint 指令:

```python
content = content.replace('module ', '/* verilator lint_off CASEOVERLAP*/\nmodule ')
content = content.replace('endmodule', 'endmodule\n/* verilator lint_on CASEOVERLAP*/')
```

## 一键生成

使用 Makefile:
```bash
cd flopoco-sv-lib

# 生成所有模块
make gen-flopoco

# 生成 Bender.yml + lint 检查
make all
```

## 支持的运算类型

| 运算 | FloPoCo 命令 | 输出模块 |
|------|-------------|----------|
| FMA | `IEEEFPFMA` | IEEEFMA_H*, IEEEFMA_S* |
| 除法 | `FPDiv` | FPDiv_H*, FPDiv_S* |
| 平方根 | `FPSqrt` | FPSqrt_H*, FPSqrt_S* |
| 对数 | `FPLog` | FPLog_H*, FPLog_S* |
| 指数 | `IEEEFPExp` | IEEEExp_H*, IEEEExp_S* |
| 比较 | `FPComparator` | FPComp_H*, FPComp_S* |
| IEEE→FloPoCo | `InputIEEE` | IEEE2FP_H*, IEEE2FP_S* |
| FloPoCo→IEEE | `OutputIEEE` | FP2IEEE_H*, FP2IEEE_S* |
| 定点→浮点 | `Fix2FP` | Fix2FP_H*, Fix2FP_S* |
| 浮点→定点 | `FP2Fix` | FP2Fix_H*, FP2Fix_S* |

**命名规则**: `<运算>_<精度><流水线级数>`
- H = Half (FP16)
- S = Single (FP32)
- 数字 = 流水线级数

例如: `IEEEFMA_S3` = FP32 FMA, 3 级流水线

## 生成的目录示例

```
flopoco/IEEEFMA/
├── IEEEFMA_H1.v    # FP16 FMA, 1 级流水线 (100MHz)
├── IEEEFMA_H2.v    # FP16 FMA, 2 级流水线 (150MHz)
├── IEEEFMA_H3.v    # FP16 FMA, 3 级流水线 (200MHz)
├── IEEEFMA_H5.v    # FP16 FMA, 5 级流水线 (300MHz)
├── IEEEFMA_S1.v    # FP32 FMA, 1 级流水线
├── IEEEFMA_S2.v    # FP32 FMA, 2 级流水线
└── IEEEFMA_S3.v    # FP32 FMA, 3 级流水线
```

## 自定义生成

修改 gen.py 中的 tasks 和 frequency 列表:

```python
tasks = [
    {'type': 'H', 'exp': 5, 'frac': 10},   # FP16
    {'type': 'S', 'exp': 8, 'frac': 23},   # FP32
    {'type': 'D', 'exp': 11, 'frac': 52},  # FP64 (可选)
]

# 目标频率列表 (MHz)
for frequency in [100, 150, 200, 250, 300, 400]:
    gen_fma(frequency, task)
    # ...
```

## 注意事项

1. **FloPoCo 格式**: FloPoCo 内部使用的浮点格式比 IEEE 754 多 2 位，需要使用 `InputIEEE`/`OutputIEEE` 进行转换

2. **异步复位**: 使用 `allRegistersWithAsyncReset=1` 确保生成异步复位寄存器

3. **版本兼容**: 部分模块 (如 Fix2FP) 需要使用 FloPoCo 4.1.1 版本

4. **流水线深度**: 更高的目标频率会生成更深的流水线，增加延迟但提高吞吐量
