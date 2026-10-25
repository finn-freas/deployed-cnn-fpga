# Board bring-up: Tang Nano 20K

A checklist for the first power-up. Each step tests one thing, says what you should see,
and what to try if you don't. The tools and both bitstreams are tested in simulation
(`tb/test_bringup.py`, and `tb/test_board_check.py`, which runs the host script against the
simulated board), so a failure here points at the board, its pins or its timing.

Commands are for Windows PowerShell, from the repository:

```powershell
cd \\wsl$\Ubuntu\home\finnf\deployed-cnn-fpga
```

## What you need

- the Tang Nano 20K and a USB-C data cable (an HDMI monitor is optional)
- Python 3 with `numpy pyserial pillow opencv-python` (`pip install ...`)
- `openFPGALoader` from [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build)
  (here: `C:\oss-cad-suite`, on the PATH)
- the two bitstreams, built in WSL or downloaded from the CI run's artifacts
  (**Actions → latest run → npu_board_top-bitstream / bringup-bitstream**):

  ```bash
  make -C fpga/tangnano20k TOP=bringup_top FREQ_MHZ=54      # build/bringup_top/bringup_top.fs
  make -C fpga/tangnano20k TOP=npu_board_top FREQ_MHZ=54    # build/npu_board_top/npu_board_top.fs
  ```

## 0. Plug in

The board's USB chip shows up as a serial port (the UART) and a JTAG interface. Find the
port with `python -m serial.tools.list_ports` (or Device Manager → Ports), e.g. `COM5`.

`openFPGALoader -b tangnano20k --detect` should print a GW2AR-18. If it says `unable to open
ftdi device`, the JTAG interface needs the WinUSB driver: run [Zadig](https://zadig.akeo.ie),
**Options → List all devices**, pick the board's interface **0**, install WinUSB. Leave
interface 1 alone, since that one is the COM port.

## 1. Bring-up bitstream, into SRAM

Loading into SRAM is volatile and harmless: a power cycle restores whatever is in flash.

```powershell
openFPGALoader -b tangnano20k fpga\tangnano20k\build\bringup_top\bringup_top.fs
```

| LED | Expected | If not |
|-|-|-|
| 0 | blinks ~1.6 Hz (27 MHz crystal) | the bitstream did not load: check the openFPGALoader output |
| 1 | blinks ~3.2 Hz (54 MHz PLL) | the PLL did not lock (`rtl/gowin/gw_pll_sys.sv`) |
| 2 | on within a second (SDRAM ready) | SDRAM initialisation or calibration failed: step 3 says which |
| 3 | off (busy during a test) | |
| 4 | on after a passed test | |
| 5 | off; on = a test failed or SDRAM calibration failed | step 3 |

S1 resets (and re-runs SDRAM calibration); S2 runs all tests, so the LEDs report without a PC.

## 2. Write the model to flash

The model (weights, parameters, program and a demo image) goes into the flash at 4 MiB,
clear of the bitstream at 0:

```powershell
cd sw
python -m npu.board ..\models\ir\vww --image photo.jpg --program
cd ..
```

It prints the image's `header` and `sum` (the bring-up check compares them), then
openFPGALoader erases, writes and verifies the region. Writing the flash stops the FPGA's
design, so **reload the bring-up bitstream** (step 1) afterwards.

## 3. Check UART, SDRAM and flash

```powershell
python sw\board\board_check.py --port COM5 --model models\ir\vww
```

Expected (the same as in simulation, apart from the SDRAM range and latency):

```
[PASS] UART: ping -> 'pong'
[PASS] SDRAM calibration: SDRAM: ready, read latency CAS-1+1
[PASS] SDRAM test: mem: PASS, 0x20000 words x 2 patterns
[PASS] flash read: flash: image found, header b5e00d800000009c, 0x07bee words, sum ...
[PASS] flash holds models\ir\vww: 31726 words, sum ...
all checks passed
```

| Failure | Meaning | Try |
|-|-|-|
| UART: no reply | wrong port, bitstream not loaded, or UART pins | check LED 0; another COM port; pins 69 (TX) / 70 (RX) in `tangnano20k.cst` |
| UART: garbled | baud rate or clock | LED 1 (PLL) must blink |
| SDRAM calibration FAILED | no read latency (CAS−1 … CAS+2) returned the pattern | SDRAM pin names in `tangnano20k.cst`; the clock phase in `rtl/gowin/gw_sdram_clk.sv` |
| SDRAM test: errors on a few DQ lines | those data pins | the script names them (DQ0–31) |
| SDRAM test: every word wrong | read timing | the latency window or the clock phase |
| SDRAM test: some addresses only | address or bank pins | `O_sdram_addr` / `O_sdram_ba` |
| flash: no model image | nothing written, or the flash pins | redo step 2; pins 59–62, `--mspi_as_gpio` |
| flash holds another model | a different or corrupt write | redo step 2 (it verifies) |

You can also type the commands in any terminal at 1,000,000 baud: `p` ping, `s` SDRAM
status, `m` SDRAM test, `f` flash, `a` all.

## 4. The NPU, into SRAM

```powershell
openFPGALoader -b tangnano20k fpga\tangnano20k\build\npu_board_top\npu_board_top.fs
```

At power-up the board copies the model from flash into SDRAM and classifies the demo image.
The LEDs show: 0 person, 1 no person, 2 busy, 3 booted, 4 boot error (no valid model in
flash), and 5 a heartbeat. The HDMI output shows the thumbnail, the verdict, the score and
the latency.

```powershell
python sw\board\board_check.py --port COM5 --npu --model models\ir\vww
```

This sends random images and compares each result with the simulator, which is bit-exact
with the hardware. It prints the measured latency (expect about 4.75 ms).

## 5. Make it permanent

```powershell
openFPGALoader -b tangnano20k -f fpga\tangnano20k\build\npu_board_top\npu_board_top.fs
```

This writes the bitstream at the start of the flash (the model at 4 MiB stays). After a power
cycle the board boots into the person detector on its own.

## 6. Webcam demo

```powershell
python sw\board\demo.py --port COM5               # live, q quits
python sw\board\demo.py --port COM5 a.jpg b.jpg   # files
```

At 1 Mbaud a 96×96 RGB frame takes 0.28 s to send, so the demo runs at about 3 fps; the
NPU itself needs under 5 ms.

## Recovery

Nothing here can brick the board: the FPGA is always reachable over JTAG, SRAM loads vanish at
power-off, and `openFPGALoader -b tangnano20k -f <any .fs>` replaces whatever bitstream is in
flash.
