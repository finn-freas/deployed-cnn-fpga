#!/bin/bash
set -e
export PATH="$HOME/oss-cad-suite/bin:$PATH"
cd "$(dirname "$0")/.."
MHZ=${1:-54}
B=${OUT:-build/board$MHZ}
mkdir -p $B
rm -f $B/synth.json $B/pnr.json $B/roastbot_top.fs
[ "$MHZ" = 54 ] || [ "$MHZ" = 63 ] || { echo "the 2x-SDRAM build is 54 or 63 MHz"; exit 1; }
CMHZ=${CMHZ:-$MHZ}
SDC=$B/clocks.sdc
python3 -c "p = 1000 / $CMHZ; print(f'create_clock -period {p:.3f} -name clk [get_nets clk]'); "\
"[print(f'create_clock -period {p / 2:.3f} -name {n} [get_nets {n}]') for n in ('clk_f', 'clk_p')]" > $SDC
VENDOR="rtl/board/gw_pll_2x.sv rtl/board/gw_pll_2x_63.sv rtl/board/gw_sdram_clk.sv"
SRCS="rtl/board/reset_sync.sv rtl/board/uart.sv rtl/board/sdram_ctrl.sv rtl/board/sdram_cdc.sv rtl/host_link.sv rtl/vmem.sv rtl/stream_rd.sv rtl/vec_unit.sv rtl/mv_unit.sv rtl/attn_unit.sv rtl/sampler.sv rtl/engine.sv rtl/board/roastbot_top.sv"
yosys -m slang -q -l $B/yosys.log -p "read_verilog -sv $VENDOR; read_slang --top roastbot_top -G MHZ=$MHZ $SRCS; \
  synth_gowin -top roastbot_top -family gw2a ${YOPT--nowidelut} -json $B/synth.json" 2>&1 | grep -v "^$" || true
nextpnr-himbaechel --json $B/synth.json --write $B/pnr.json --device GW2AR-LV18QN88C8/I7 \
  --vopt family=GW2A-18C --vopt cst=rtl/board/tangnano20k.cst --freq $CMHZ --sdc $SDC --ignore-rel-clk ${SEED:+--seed $SEED} -l $B/pnr.log --quiet
gowin_pack -d GW2A-18C -o $B/roastbot_top.fs $B/pnr.json 2>&1 | grep -v "UserWarning\|warnings.warn" || true
grep "Max frequency for clock" $B/pnr.log | sed 's/^Info: */  /' | sort -u
grep -E "^Info:\s+(LUT4|ALU|DFF|BSRAM|MULT18X18):" $B/pnr.log | sed 's/^Info:\s*/  /'
ls -la $B/roastbot_top.fs
