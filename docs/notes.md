# Design notes

Things I learned getting VWW onto the GW2AR-18, roughly in order. Cycle counts are for one
VWW inference with ideal memory unless I say otherwise.

## Getting the cycle count down

The first version that ran VWW bit-exact took **705,711 cycles**. Every step below was
checked bit-exact against TFLite and against the simulator's cycle model before I moved on.

| Step | Cycles |
|-|-:|
| first bit-exact VWW, drain overlapped with the next job | 705,711 |
| double-buffered transposer | 462,990 |
| weight prefetch for the next channel group | 419,791 |
| packed first layer (RGB pixels gathered into one word) | 376,047 |
| second requant pipeline in the drain | 335,657 |
| overlap the gather tail with the next job | 320,682 |
| depthwise mode in the array (below) | 264,161 |
| row-skew-only array (drain 38 -> 34 cycles per job) | 258,808 |
| parameters stream during the layer, packed-input words reused | **243,602** |

The depthwise layers were the biggest single chunk: 128k cycles before the depthwise mode,
71.5k after. A depthwise unit is 8 neighbouring output pixels of one row times one channel
block, and the PEs get their inputs straight from the activation memory reads.

Where the remaining time goes: the two requant pipelines need at least 34 cycles for a job's
64 outputs, which limits the small pointwise layers and the stride-1 depthwise layers.
Stride-2 depthwise is limited by memory reads instead.

## The placer that "hung"

At some point nextpnr just stopped making progress. With
`--placer-heap-cell-placement-timeout` it finally said "Unable to find legal placement for
cell ... MULT9X9": a multiplier in the pool unit's address math. The default timeout grows
with the square of the cell count, so it looks exactly like a hang. I rewrote all address
arithmetic as counters or shift-and-add, so the only DSPs left are the 64 PEs and the two
requant multipliers.

## The second wall: LUT positions

After that it still wouldn't place, and this one took a while to understand. On Gowin a
flip-flop's D input can only come from the LUT in its own slice. So any register fed by
another register (or by a RAM, or by a LUT that already drives a different register) gets a
pass-through LUT, and the utilisation report doesn't count those. Real demand is
LUT + ALU + those registers, and mine was **21.4k of 20.7k**. I confirmed it on a small probe
design where every flip-flop ended up behind its own LUT.

What got it down to 15.6k (75%):

- PE shadow registers latch the adder output on the job's last value, instead of a separate
  capture wave
- one requant datapath shared by single and double rounding
- 26-bit accumulators (enough for K <= 1024)
- parameters, transposer (diagonal byte RAMs) and row tables in LUT RAM
- average pooling lowered to a DWCONV with all-ones weights, so the pool unit only does max
- units read the sequencer's instruction directly instead of keeping copies
- depthwise mode without a line buffer: the array keeps only its row skew, so each PE row
  needs exactly the word that's arriving from memory, and every tap is issued with the read
  of its word (stride 2 reads even columns before odd ones)

The board design now sits at about 81.6% real demand, and from experience it stops placing
at around 83%. So every RTL change has to pay for itself. Removing register copies is
usually the cheapest win.

## Board system

`npu_board_top` is `npu_system` (flash boot loader, weights copied into the in-package
SDRAM, UART image upload, 48x48 thumbnail) plus the HDMI overlay. NPU, SDRAM, flash and
UART run at 54 MHz from a second PLL, and HDMI at 25.2 MHz. Status crosses between the two
clocks with a toggle handshake, and the thumbnail goes through a dual-clock BRAM.

Running weights straight from the 1-bit SPI flash would have taken 4.23M cycles, so the
board copies them into SDRAM at boot. From SDRAM, VWW takes 256,642 cycles (4.75 ms). The
SDRAM read latency is calibrated at boot because I couldn't be sure of the board's timing.

## Tried and dropped

- **4 requant pipelines:** the cycle model says -7.8% (238.7k), but it needs about 1.2k LUTs
  and 550 ALUs more than the chip has left. The array is already at its packing minimum
  (each PE's adder feeds two registers).
- **One-word parameters (27-bit bias):** fails on VWW. Some dead channels have biases around
  +-2^30 with tiny scales.
- **Cross-layer prefetch:** possible, but too much control logic for the space that's left.
