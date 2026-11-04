ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
PIXI ?= pixi
NAME := $(basename $(notdir $(CADL)))
OUT ?= $(ROOT)/build/examples/$(notdir $(EXAMPLE_DIR))/$(NAME)
RESOURCE ?= $(ROOT)/examples/resource_ihp130.json

.DEFAULT_GOAL := help
.PHONY: help rtl compile build
help:
	@echo 'make CADL=$(CADL) rtl'
	@echo 'make CADL=<instruction.cadl> C=<caller.c> compile'
	@echo 'OUT and RESOURCE override output and HLS resources; make build builds tools.'
rtl:
	mkdir -p "$(OUT)"
	cd "$(ROOT)" && $(PIXI) run mlir "$(EXAMPLE_DIR)/$(CADL)" "$(OUT)/$(NAME).mlir"
	cd "$(ROOT)" && $(PIXI) run build/tools/aps-opt/aps-e2e -i "$(OUT)/$(NAME).mlir" --resource "$(RESOURCE)" --export sv -o "$(OUT)/$(NAME).sv"
compile:
	@test -n "$(C)" || { echo 'Set C to the matching C input relative to $(EXAMPLE_DIR)' >&2; exit 1; }
	mkdir -p "$(OUT)"
	cd "$(ROOT)" && $(PIXI) run compile "$(EXAMPLE_DIR)/$(CADL)" "$(EXAMPLE_DIR)/$(C)" "$(OUT)/$(NAME).riscv"
build:
	cd "$(ROOT)" && $(PIXI) run build
