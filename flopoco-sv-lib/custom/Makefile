# =============================================================================
# FloPoCo VHDL to SystemVerilog 一键生成 Makefile
# =============================================================================
# 用法:
#   make                          # 使用默认参数生成
#   make OP=FPAdd WE=8 WF=23      # 自定义参数
#   make help                     # 查看帮助
# =============================================================================

# =============================================================================
# 可配置参数
# =============================================================================

# 运算类型
# 可选: FPAdd, FPMult, FPDiv, FPSqrt, FPLog, IEEEFPFMA, FPComparator,
#       InputIEEE, OutputIEEE, Fix2FP, FP2Fix, IEEEFPExp
OP ?= FPAdd

# 浮点格式参数
# FP16: WE=5, WF=10
# FP32: WE=8, WF=23
# FP64: WE=11, WF=52
WE ?= 8
WF ?= 23

# 目标频率 MHz (FloPoCo 自动计算流水线深度)
FREQUENCY ?= 200

# 或直接指定流水线级数 (取消注释使用, 优先于 FREQUENCY)
# PIPELINE ?= 3

# 模块命名
NAME ?= $(OP)_w$(WE)e$(WF)f
SUFFIX ?= _custom

# 路径配置
# 本地编译版本: /home/xys/aps-mlir/flopoco/build/bin/flopoco (v5.0.0)
# Docker 备选: ./flopoco-docker.sh (旧版)
FLOPOCO_ROOT ?= /home/xys/aps-mlir
FLOPOCO_LIBS ?= $(FLOPOCO_ROOT)/sollya-install/lib:$(FLOPOCO_ROOT)/pagsuite-install/lib:$(FLOPOCO_ROOT)/.pixi/envs/default/lib
FLOPOCO_BIN ?= LD_LIBRARY_PATH=$(FLOPOCO_LIBS):$$LD_LIBRARY_PATH $(FLOPOCO_ROOT)/flopoco/build/bin/flopoco
FLOPOCO_411 ?= LD_LIBRARY_PATH=$(FLOPOCO_LIBS):$$LD_LIBRARY_PATH $(FLOPOCO_ROOT)/flopoco/build/bin/flopoco
SRC_DIR ?= $(CURDIR)
OUT_DIR ?= $(CURDIR)/generated

# 使用异步复位 (1=是, 0=否)
ASYNC_RESET ?= 1

# 目标 FPGA (Virtex6, Zynq7000, StratixV, etc.)
TARGET ?= Virtex6

# 工具路径
YOSYS ?= yosys
GHDL ?= ghdl

# =============================================================================
# 内部变量 (不要修改)
# =============================================================================

VHDL_FILE = $(SRC_DIR)/$(NAME).vhdl
VERILOG_FILE = $(OUT_DIR)/$(NAME).v
SV_FILE = $(OUT_DIR)/$(NAME).sv

# FloPoCo 全局选项 (在运算符名称之前)
FLOPOCO_GLOBAL_OPTS := name=$(NAME)

ifdef PIPELINE
  FLOPOCO_GLOBAL_OPTS += pipeline=$(PIPELINE)
else
  FLOPOCO_GLOBAL_OPTS += frequency=$(FREQUENCY)
endif

ifdef TARGET
  FLOPOCO_GLOBAL_OPTS += target=$(TARGET)
endif

# FloPoCo 运算符参数 (在运算符名称之后)
FLOPOCO_OP_OPTS := wE=$(WE) wF=$(WF)

# 特殊运算的额外参数
MSB_VAL := $(shell expr $(WE) + $(WF))
FIX2FP_OPTS := LSB=0 MSB=$(MSB_VAL)
FP2FIX_OPTS := LSB=0 MSB=$(MSB_VAL)
IEEE2FP_OPTS := wEIn=$(WE) wFIn=$(WF) wEOut=$(WE) wFOut=$(WF)
FP2IEEE_OPTS := wEIn=$(WE) wFIn=$(WF) wEOut=$(WE) wFOut=$(WF)

# =============================================================================
# 主目标
# =============================================================================

.PHONY: all clean help gen-vhdl convert-verilog convert-sv info

all: $(SV_FILE)
	@echo "=========================================="
	@echo "生成完成: $(SV_FILE)"
	@echo "=========================================="

# =============================================================================
# 步骤 1: 生成 VHDL
# =============================================================================

.PHONY: gen-vhdl
gen-vhdl: $(VHDL_FILE)

$(VHDL_FILE): | $(OUT_DIR)
	@echo "=========================================="
	@echo "步骤 1: 生成 FloPoCo VHDL"
	@echo "运算: $(OP), 格式: wE=$(WE) wF=$(WF)"
	@echo "=========================================="
ifeq ($(OP),FPAdd)
	$(FLOPOCO_BIN) $(FLOPOCO_GLOBAL_OPTS) FPAdd $(FLOPOCO_OP_OPTS)
else ifeq ($(OP),FPMult)
	$(FLOPOCO_BIN) $(FLOPOCO_GLOBAL_OPTS) FPMult $(FLOPOCO_OP_OPTS)
else ifeq ($(OP),FPDiv)
	$(FLOPOCO_BIN) $(FLOPOCO_GLOBAL_OPTS) FPDiv $(FLOPOCO_OP_OPTS)
else ifeq ($(OP),FPSqrt)
	$(FLOPOCO_BIN) $(FLOPOCO_GLOBAL_OPTS) FPSqrt $(FLOPOCO_OP_OPTS)
else ifeq ($(OP),FPLog)
	$(FLOPOCO_BIN) $(FLOPOCO_GLOBAL_OPTS) FPLog $(FLOPOCO_OP_OPTS) method=0
else ifeq ($(OP),IEEEFPFMA)
	$(FLOPOCO_BIN) $(FLOPOCO_GLOBAL_OPTS) IEEEFPFMA $(FLOPOCO_OP_OPTS)
else ifeq ($(OP),FPComparator)
	$(FLOPOCO_BIN) $(FLOPOCO_GLOBAL_OPTS) FPComparator $(FLOPOCO_OP_OPTS)
else ifeq ($(OP),IEEEFPExp)
	$(FLOPOCO_BIN) $(FLOPOCO_GLOBAL_OPTS) IEEEFPExp $(FLOPOCO_OP_OPTS)
else ifeq ($(OP),InputIEEE)
	$(FLOPOCO_BIN) $(FLOPOCO_GLOBAL_OPTS) InputIEEE $(IEEE2FP_OPTS)
else ifeq ($(OP),OutputIEEE)
	$(FLOPOCO_BIN) $(FLOPOCO_GLOBAL_OPTS) OutputIEEE $(FP2IEEE_OPTS)
else ifeq ($(OP),Fix2FP)
	$(FLOPOCO_411) $(FLOPOCO_GLOBAL_OPTS) Fix2FP $(FIX2FP_OPTS) $(FLOPOCO_OP_OPTS)
else ifeq ($(OP),FP2Fix)
	$(FLOPOCO_BIN) $(FLOPOCO_GLOBAL_OPTS) FP2Fix $(FP2FIX_OPTS) $(FLOPOCO_OP_OPTS)
else
	$(error 不支持的运算类型: $(OP))
endif
	@mv flopoco.vhdl $(VHDL_FILE) 2>/dev/null || true
	@echo "VHDL 生成完成: $(VHDL_FILE)"

# =============================================================================
# 步骤 2: VHDL -> Verilog
# =============================================================================

.PHONY: convert-verilog
convert-verilog: $(VERILOG_FILE)

$(VERILOG_FILE): $(VHDL_FILE) | $(OUT_DIR)
	@echo "=========================================="
	@echo "步骤 2: VHDL -> Verilog 转换"
	@echo "=========================================="
	@echo "yosys -import;" > $(SRC_DIR)/run.tcl
	@echo "ghdl --std=08 -fsynopsys -fexplicit $(VHDL_FILE) -e $(NAME);" >> $(SRC_DIR)/run.tcl
	@echo "hierarchy -top $(NAME);" >> $(SRC_DIR)/run.tcl
	@echo "write_verilog $(VERILOG_FILE);" >> $(SRC_DIR)/run.tcl
	@echo "exit;" >> $(SRC_DIR)/run.tcl
	$(YOSYS) -m ghdl -c $(SRC_DIR)/run.tcl
	@# 添加 Verilator lint 指令
	@sed -i '1s/^/\/* verilator lint_off CASEOVERLAP *\/\n/' $(VERILOG_FILE)
	@echo '/* verilator lint_on CASEOVERLAP */' >> $(VERILOG_FILE)
	@rm -f $(SRC_DIR)/run.tcl
	@echo "Verilog 生成完成: $(VERILOG_FILE)"

# =============================================================================
# 步骤 3: Verilog -> SystemVerilog
# =============================================================================

.PHONY: convert-sv
convert-sv: $(SV_FILE)

$(SV_FILE): $(VERILOG_FILE)
	@echo "=========================================="
	@echo "步骤 3: 生成 SystemVerilog"
	@echo "=========================================="
	@cp $(VERILOG_FILE) $(SV_FILE)
	@sed -i '1i\// Auto-generated by FloPoCo + Yosys' $(SV_FILE)
	@sed -i '2i\// Parameters: OP=$(OP) WE=$(WE) WF=$(WF) FREQ=$(FREQUENCY)MHz' $(SV_FILE)
	@sed -i '3i\`timescale 1ns/1ps' $(SV_FILE)
	@echo "SystemVerilog 生成完成: $(SV_FILE)"

# =============================================================================
# 辅助目标
# =============================================================================

$(OUT_DIR):
	@mkdir -p $(OUT_DIR)

clean:
	rm -rf $(OUT_DIR)
	rm -f $(SRC_DIR)/*.vhdl $(SRC_DIR)/*.tcl $(SRC_DIR)/*.rpt $(SRC_DIR)/*.out

info:
	@echo "=========================================="
	@echo "当前配置:"
	@echo "=========================================="
	@echo "运算类型:     $(OP)"
	@echo "指数位宽:     $(WE)"
	@echo "尾数位宽:     $(WF)"
	@echo "目标频率:     $(FREQUENCY) MHz"
	@echo "模块名:       $(NAME)"
	@echo "输出目录:     $(OUT_DIR)"
	@echo "VHDL 文件:    $(VHDL_FILE)"
	@echo "Verilog 文件: $(VERILOG_FILE)"
	@echo "SV 文件:      $(SV_FILE)"
	@echo "异步复位:     $(ASYNC_RESET)"
	@echo "=========================================="

help:
	@echo "=========================================="
	@echo "FloPoCo VHDL to SV 生成工具"
	@echo "=========================================="
	@echo ""
	@echo "用法: make [目标] [参数=值]"
	@echo ""
	@echo "目标:"
	@echo "  all             完整生成流程 (默认)"
	@echo "  gen-vhdl        仅生成 VHDL"
	@echo "  convert-verilog 转换为 Verilog"
	@echo "  convert-sv      转换为 SystemVerilog"
	@echo "  clean           清理生成文件"
	@echo "  info            显示当前配置"
	@echo "  help            显示此帮助"
	@echo ""
	@echo "参数:"
	@echo "  OP=<运算>       运算类型 (默认: FPAdd)"
	@echo "                  可选: FPAdd, FPMult, FPDiv, FPSqrt, FPLog,"
	@echo "                        IEEEFPFMA, FPComparator, IEEEFPExp,"
	@echo "                        InputIEEE, OutputIEEE, Fix2FP, FP2Fix"
	@echo "  WE=<n>          指数位宽 (默认: 8)"
	@echo "  WF=<n>          尾数位宽 (默认: 23)"
	@echo "  FREQUENCY=<n>   目标频率 MHz (默认: 200)"
	@echo "  PIPELINE=<n>    流水线级数 (可选, 优先于 FREQUENCY)"
	@echo "  NAME=<name>     模块名 (默认: 自动生成)"
	@echo "  OUT_DIR=<path>  输出目录 (默认: ./generated)"
	@echo "  ASYNC_RESET=1   使用异步复位 (默认: 1)"
	@echo "  TARGET=<fpga>   目标 FPGA (默认: Virtex6)"
	@echo ""
	@echo "示例:"
	@echo "  make OP=FPAdd WE=8 WF=23 FREQUENCY=300"
	@echo "  make OP=FPDiv WE=11 WF=52 NAME=fp64_div"
	@echo "  make OP=IEEEFPFMA PIPELINE=5"
	@echo "  make OP=FPSqrt WE=5 WF=10 OUT_DIR=/tmp/output"
	@echo ""

# =============================================================================
# 批量生成
# =============================================================================

.PHONY: gen-fp16 gen-fp32 gen-fp64 gen-all-formats gen-freq-sweep

gen-fp16:
	$(MAKE) WE=5 WF=10 NAME=$(OP)_fp16

gen-fp32:
	$(MAKE) WE=8 WF=23 NAME=$(OP)_fp32

gen-fp64:
	$(MAKE) WE=11 WF=52 NAME=$(OP)_fp64

gen-all-formats: gen-fp16 gen-fp32 gen-fp64

gen-freq-sweep:
	@for freq in 100 150 200 250 300 400; do \
		$(MAKE) FREQUENCY=$$freq NAME=$(OP)_$(WE)e$(WF)f_$${freq}MHz; \
	done
