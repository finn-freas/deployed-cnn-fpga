# Top-level build for deployed-cnn-fpga
#   make env    - create Python environments
#   make help   - list targets

SHELL := /bin/bash
OSS   ?= $(HOME)/oss-cad-suite
PY    := .venv/bin/python
PYTF  := .venv-tf/bin/python

.PHONY: help env

help:
	@echo "env      create .venv (sim) and .venv-tf (TensorFlow)"

env: .venv/.ok .venv-tf/.ok

.venv/.ok: requirements-sim.txt
	python3 -m venv .venv
	.venv/bin/pip install -q -r requirements-sim.txt uv
	touch $@

.venv-tf/.ok: requirements-tf.txt .venv/.ok
	.venv/bin/uv venv -q -p 3.12 .venv-tf
	.venv/bin/uv pip install -q -p $(PYTF) -r requirements-tf.txt
	touch $@
