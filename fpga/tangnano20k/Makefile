# Tang Nano 20K build: yosys -> nextpnr-himbaechel -> gowin_pack -> openFPGALoader
#   make TOP=colorbars_top          build bitstream
#   make TOP=colorbars_top flash    program SRAM (volatile)
#   make TOP=colorbars_top prog     program flash (persistent)

SHELL  := /bin/bash
OSS    ?= $(HOME)/oss-cad-suite
ENV    := source $(OSS)/environment &&
ROOT   := ../..
TOP    ?= colorbars_top
TAG    ?=
BUILD  := build/$(TOP)$(TAG)
NEXTPNR_ARGS ?=                 # e.g. --placer sa, --threads 8
SYNTH_ARGS ?=                   # e.g. -nodsp
CST    := tangnano20k.cst
DEVICE := GW2AR-LV18QN88C8/I7
FAMILY := GW2A-18C
FREQ_MHZ ?= 25.2

# Vendor-primitive wrappers: read by Yosys's native frontend (understands Gowin cells).
# Everything else: read by yosys-slang (full SystemVerilog).
VENDOR_SRCS := \
	$(ROOT)/rtl/gowin/gw_clk_gen.sv \
	$(ROOT)/rtl/gowin/gw_dvi_phy.sv \
	$(ROOT)/rtl/gowin/gw_sdram_clk.sv 	$(ROOT)/rtl/gowin/gw_pll_sys.sv

# Per-top source lists
SRCS_colorbars_top := \
	$(ROOT)/rtl/common/reset_sync.sv \
	$(ROOT)/rtl/video/video_timing.sv \
	$(ROOT)/rtl/video/tmds_encoder.sv \
	$(ROOT)/rtl/video/dvi_tx.sv \
	$(ROOT)/rtl/video/test_pattern.sv \
	colorbars_top.sv

NPU_SRCS := \
	$(ROOT)/rtl/npu/npu_pkg.sv \
	$(ROOT)/rtl/common/sdpram.sv \
	$(ROOT)/rtl/npu/npu_pe.sv \
	$(ROOT)/rtl/npu/npu_array.sv \
	$(ROOT)/rtl/npu/npu_requant.sv \
	$(ROOT)/rtl/npu/npu_tile_engine.sv \
	$(ROOT)/rtl/npu/npu_pool.sv \
	$(ROOT)/rtl/npu/npu_core.sv

# Synthesis-only resource probe of the NPU core (not a functional board design)
SRCS_npu_probe_top := $(NPU_SRCS) npu_probe_top.sv
SRCS_dsp_probe_top := dsp_probe_top.sv
SRCS_dsp_mix_probe_top := dsp_mix_probe_top.sv
SRCS_mem_probe_top := $(ROOT)/rtl/common/sdpram.sv mem_probe_top.sv
SRCS_dp_probe_top := $(ROOT)/rtl/npu/npu_pkg.sv $(ROOT)/rtl/npu/npu_pe.sv $(ROOT)/rtl/npu/npu_array.sv $(ROOT)/rtl/npu/npu_requant.sv dp_probe_top.sv
SRCS_memdp_probe_top := $(SRCS_dp_probe_top) $(ROOT)/rtl/common/sdpram.sv memdp_probe_top.sv
SRCS_npu_probe_small_top := $(NPU_SRCS) npu_probe_small_top.sv

# The person detector: NPU + flash boot + UART + HDMI overlay
SRCS_npu_board_top := $(NPU_SRCS) 	$(ROOT)/rtl/common/reset_sync.sv 	$(ROOT)/rtl/common/spi_flash_reader.sv 	$(ROOT)/rtl/board/uart.sv 	$(ROOT)/rtl/board/sdram_ctrl.sv 	$(ROOT)/rtl/board/npu_system.sv 	$(ROOT)/rtl/board/status_cdc.sv 	$(ROOT)/rtl/common/sdpram_dc.sv 	$(ROOT)/rtl/board/bin2bcd.sv 	$(ROOT)/rtl/board/font_rom.sv 	$(ROOT)/rtl/board/npu_display.sv 	$(ROOT)/rtl/video/video_timing.sv 	$(ROOT)/rtl/video/tmds_encoder.sv 	$(ROOT)/rtl/video/dvi_tx.sv 	npu_board_top.sv
PACK_ARGS_npu_board_top := --mspi_as_gpio

SRCS := $(SRCS_$(TOP))

.PHONY: all flash prog clean

all: $(BUILD)/$(TOP).fs

$(BUILD)/$(TOP)-synth.json: $(SRCS) $(VENDOR_SRCS)
	@mkdir -p $(BUILD)
	$(ENV) yosys -m slang -q -l $(BUILD)/yosys.log -p "\
		read_verilog -sv $(VENDOR_SRCS); \
		read_slang --top $(TOP) $(SRCS); \
		synth_gowin -top $(TOP) -family gw2a $(SYNTH_ARGS) -json $@"

$(BUILD)/$(TOP)-pnr.json: $(BUILD)/$(TOP)-synth.json $(CST)
	$(ENV) nextpnr-himbaechel --json $< --write $@ --device $(DEVICE) \
		--vopt family=$(FAMILY) --vopt cst=$(CST) --report $(BUILD)/report.json \
		--freq $(FREQ_MHZ) $(NEXTPNR_ARGS) -l $(BUILD)/nextpnr.log --quiet

$(BUILD)/$(TOP).fs: $(BUILD)/$(TOP)-pnr.json
	$(ENV) gowin_pack -c -d $(FAMILY) $(PACK_ARGS_$(TOP)) -o $@ $< 2> >(grep -v "UserWarning\|warnings.warn" >&2)
	@echo "--- $(TOP): timing (target $(FREQ_MHZ) MHz) ---"
	@grep "Max frequency for clock" $(BUILD)/nextpnr.log | sed 's/^Info: */  /' | sort -u
	@echo "--- utilization ---"
	@grep -E "^Info:\s+(LUT4|ALU|DFF|BSRAM|MULT9X9|MULT18X18|RAM16SDP4|IOB):" $(BUILD)/nextpnr.log | sed 's/^Info:\s*/  /'

flash: $(BUILD)/$(TOP).fs
	$(ENV) openFPGALoader -b tangnano20k $<

prog: $(BUILD)/$(TOP).fs
	$(ENV) openFPGALoader -b tangnano20k -f $<

clean:
	rm -rf build
