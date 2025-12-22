# Copyright 2025 Tobias Senti
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

.PHONY: all lint

all: bender lint
lint: lint-verilator lint-verible

# Generate Flopoco Modules
gen-flopoco: flopoco/gen.py
	cd flopoco && ./gen.py
	
# Generate Bender File
bender: generate_bender.py gen-flopoco
	./generate_bender.py

# Verible Lint
lint-verible:
	verible-verilog-lint rtl/*.sv rtl/**/*.sv

# Verilator Lint
lint-verilator:
	bender script verilator > verilator.f
	verilator --lint-only -Wno-MULTITOP -f verilator.f

# Clean Generated Files
clean:
	rm -f Bender.yml
	rm -f flopoco/**/*.v
	rm -f flopoco/run.tcl
	rm -f flopoco/modules.rpt
	rm -f *.f
