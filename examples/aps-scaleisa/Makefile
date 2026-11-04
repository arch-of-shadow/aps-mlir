ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
PIXI ?= pixi
CASE ?= bitcount
CASES ?= $(CASE)
OUT ?= $(ROOT)/build/aps-scaleisa-examples
RUNNER = $(PIXI) run python examples/aps-scaleisa/run.py

.DEFAULT_GOAL := help
.PHONY: help list plan run rocket build
help:
	@echo 'make list | make CASE=bitcount run | make CASES="octree iss" rocket'
	@echo 'Use CASE=all for all 16 cases; OUT overrides the output directory.'
	@echo 'make plan prints commands; make build builds the project tools.'
list:
	cd "$(ROOT)" && $(RUNNER) --list
plan:
	cd "$(ROOT)" && $(RUNNER) --cases $(CASES) --output-dir "$(OUT)" --plan
run:
	cd "$(ROOT)" && $(RUNNER) --cases $(CASES) --output-dir "$(OUT)"
rocket:
	cd "$(ROOT)" && $(RUNNER) --cases $(CASES) --output-dir "$(OUT)" --chipyard
build:
	cd "$(ROOT)" && $(PIXI) run build
