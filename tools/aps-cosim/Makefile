VERILATOR ?= verilator
CXX ?= g++
JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
VERILATE_JOBS ?= $(JOBS)
BUILD_JOBS ?= $(JOBS)
RTL ?= example/crypto_pqc.sv
TOP ?= main
OBJ_DIR ?= obj_dir
SPIKE_OBJ_DIR ?= obj_dir_spike
SPIKE_DIR ?= ../../thirdparty/chipyard/toolchains/riscv-tools/riscv-isa-sim
SPIKE_BUILD ?= $(SPIKE_DIR)/build
SPIKE_OBJ_SPIKE_DIR := ../../../thirdparty/chipyard/toolchains/riscv-tools/riscv-isa-sim
SPIKE_OBJ_SPIKE_BUILD := $(SPIKE_OBJ_SPIKE_DIR)/build

COMMON_SRCS := \
  aps_rtl_model.cc \
  csr_args.cc \
  csr_file.cc \
  csr_runtime_bridge.cc \
  fake_dma.cc \
  memory_bridge.cc

DIRECT_SRCS := \
  $(COMMON_SRCS) \
  run_direct.cc

SPIKE_SRCS := \
  $(COMMON_SRCS) \
  aps_rocc_extension.cc \
  spike_memory_bridge.cc \
  spike_aps_main.cc

SPIKE_CFLAGS := -std=c++17 -I.. -I$(SPIKE_OBJ_SPIKE_DIR) -I$(SPIKE_OBJ_SPIKE_BUILD)
SPIKE_LIBS := \
  $(SPIKE_OBJ_SPIKE_BUILD)/libriscv.a \
  $(SPIKE_OBJ_SPIKE_BUILD)/libdisasm.a \
  $(SPIKE_OBJ_SPIKE_BUILD)/libsoftfloat.a \
  $(SPIKE_OBJ_SPIKE_BUILD)/libfesvr.a \
  $(SPIKE_OBJ_SPIKE_BUILD)/libfdt.a \
  -ldl -lpthread -lz

.PHONY: all direct spike clean FORCE

all: direct

direct: $(OBJ_DIR)/V$(TOP)

spike: $(SPIKE_OBJ_DIR)/V$(TOP)

FORCE:

$(OBJ_DIR)/rtl_path.stamp: FORCE
	mkdir -p $(OBJ_DIR)
	@rtl_path="$$(realpath $(RTL))"; \
	if ! test -f $@ || ! printf '%s\n' "$$rtl_path" | cmp -s - $@; then \
		printf '%s\n' "$$rtl_path" > $@; \
		rm -f $(OBJ_DIR)/V$(TOP); \
	fi

$(SPIKE_OBJ_DIR)/rtl_path.stamp: FORCE
	mkdir -p $(SPIKE_OBJ_DIR)
	@rtl_path="$$(realpath $(RTL))"; \
	if ! test -f $@ || ! printf '%s\n' "$$rtl_path" | cmp -s - $@; then \
		printf '%s\n' "$$rtl_path" > $@; \
		rm -f $(SPIKE_OBJ_DIR)/V$(TOP); \
	fi

$(OBJ_DIR)/V$(TOP): $(OBJ_DIR)/rtl_path.stamp $(DIRECT_SRCS) $(RTL) Makefile
	$(VERILATOR) --cc --exe --build --timing -Wall -Wno-fatal \
		--verilate-jobs $(VERILATE_JOBS) --build-jobs $(BUILD_JOBS) \
		-Wno-DECLFILENAME -Wno-PINCONNECTEMPTY -Wno-UNUSEDSIGNAL \
		-Wno-UNDRIVEN -Wno-WIDTH \
		--public-flat-rw --trace \
		--top-module $(TOP) \
		-Mdir $(OBJ_DIR) \
		-CFLAGS "-std=c++17 -I.. -I." \
		$(RTL) $(DIRECT_SRCS)

$(SPIKE_OBJ_DIR)/V$(TOP): $(SPIKE_OBJ_DIR)/rtl_path.stamp $(SPIKE_SRCS) $(RTL) Makefile
	$(VERILATOR) --cc --exe --build --timing -Wall -Wno-fatal \
		--verilate-jobs $(VERILATE_JOBS) --build-jobs $(BUILD_JOBS) \
		-Wno-DECLFILENAME -Wno-PINCONNECTEMPTY -Wno-UNUSEDSIGNAL \
		-Wno-UNDRIVEN -Wno-WIDTH \
		--public-flat-rw --trace \
		--top-module $(TOP) \
		-Mdir $(SPIKE_OBJ_DIR) \
		-CFLAGS "$(SPIKE_CFLAGS)" \
		-LDFLAGS "$(SPIKE_LIBS)" \
		$(RTL) $(SPIKE_SRCS)

clean:
	rm -rf $(OBJ_DIR) $(SPIKE_OBJ_DIR)
