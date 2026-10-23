# Top-level build for deployed-cnn-fpga
#   make env    - create Python environments
#   make lint   - Verilator lint of all portable RTL
#   make test   - cocotb unit tests (Verilator)
#   make fpga   - build the Tang Nano 20K bitstream (TOP=... to choose)

SHELL := /bin/bash
OSS   ?= $(HOME)/oss-cad-suite
TOOLS := export PATH="$(OSS)/bin:$$PATH" &&
PY    := $(CURDIR)/.venv/bin/python
PYTF  := $(CURDIR)/.venv-tf/bin/python
TOP   ?= colorbars_top

# Portable RTL (no vendor primitives); rtl/gowin is synthesis-only.
RTL_PORTABLE := $(shell find rtl -name '*.sv' -not -path 'rtl/gowin/*' | sort)

.PHONY: help env lint test fpga clean

help:
	@echo "env      create .venv (sim) and .venv-tf (TensorFlow)"
	@echo "lint     verilator --lint-only -Wall on each portable RTL file"
	@echo "test     run all cocotb tests"
	@echo "fpga     build bitstream for TOP=$(TOP)"

env: .venv/.ok .venv-tf/.ok

.venv/.ok: requirements-sim.txt
	python3 -m venv .venv
	.venv/bin/pip install -q -r requirements-sim.txt uv
	touch $@

.venv-tf/.ok: requirements-tf.txt .venv/.ok
	.venv/bin/uv venv -q -p 3.12 .venv-tf
	.venv/bin/uv pip install -q -p $(PYTF) -r requirements-tf.txt
	touch $@

lint:
	@for f in $(RTL_PORTABLE); do \
		$(TOOLS) verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-MULTITOP $(RTL_PORTABLE) --top-module $$(basename $$f .sv) \
			|| exit 1; \
		echo "lint ok: $$f"; \
	done

test:
	$(TOOLS) cd tb && $(PY) -m pytest -q

fpga:
	$(MAKE) -C fpga/tangnano20k TOP=$(TOP)

clean:
	rm -rf build fpga/tangnano20k/build
