# Top-level build for deployed-cnn-fpga
#   make env      - create Python environments
#   make model    - train the MNIST CNN (optional; a trained model is committed)
#   make ir       - export the TFLite model to IR + golden tensors (TensorFlow env)
#   make compile  - compile IR to NPU memory images (build/npu)
#   make pkg      - regenerate rtl/npu/npu_pkg.sv from sw/npu/isa.py
#   make lint     - Verilator lint of all portable RTL
#   make test     - software tests + cocotb RTL tests
#   make fpga     - build the Tang Nano 20K bitstream (TOP=... to choose)

SHELL := /bin/bash
OSS   ?= $(HOME)/oss-cad-suite
TOOLS := export PATH="$(OSS)/bin:$$PATH" &&
PY    := $(CURDIR)/.venv/bin/python
PYTF  := $(CURDIR)/.venv-tf/bin/python
TOP   ?= colorbars_top
MODEL ?= models/mnist_int8.tflite

# Portable RTL (no vendor primitives); rtl/gowin is synthesis-only. Packages go first.
RTL_PKGS     := $(shell find rtl -name '*_pkg.sv' -not -path 'rtl/gowin/*' | sort)
RTL_MODULES  := $(shell find rtl -name '*.sv' -not -name '*_pkg.sv' -not -path 'rtl/gowin/*' | sort)

.PHONY: help env model ir compile pkg lint test test-sw test-rtl fpga clean

help:
	@sed -n 's/^#   make /  make /p' Makefile

env: .venv/.ok .venv-tf/.ok

.venv/.ok: requirements-sim.txt
	python3 -m venv .venv
	.venv/bin/pip install -q -r requirements-sim.txt uv
	touch $@

.venv-tf/.ok: requirements-tf.txt .venv/.ok
	.venv/bin/uv venv -q -p 3.12 .venv-tf
	.venv/bin/uv pip install -q -p $(PYTF) -r requirements-tf.txt
	touch $@

model:
	$(PYTF) sw/model/train_mnist.py --out $(MODEL)

build/model/golden.npz: $(MODEL) sw/model/export_ir.py
	$(PYTF) sw/model/export_ir.py $(MODEL) build/model --golden 1000 2>&1 | grep -v -i warn

ir: build/model/golden.npz

compile: ir
	cd sw && $(PY) -m npu.compiler ../build/model --out ../build/npu

pkg:
	cd sw && $(PY) -m npu.gen_pkg ../rtl/npu/npu_pkg.sv

lint:
	@for f in $(RTL_MODULES); do \
		$(TOOLS) verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-MULTITOP \
			$(RTL_PKGS) $(RTL_MODULES) --top-module $$(basename $$f .sv) >/dev/null || exit 1; \
		echo "lint ok: $$f"; \
	done

test: test-sw test-rtl

test-sw: ir
	cd sw && $(PY) -m pytest -q tests

test-rtl:
	$(TOOLS) cd tb && $(PY) -m pytest -q

fpga:
	$(MAKE) -C fpga/tangnano20k TOP=$(TOP)

clean:
	rm -rf build fpga/tangnano20k/build
