// bringup_top: board bring-up for the Tang Nano 20K (see docs/bringup.md).
//
// Same clocks, pins and memory controllers as npu_board_top, without the NPU: checks the
// UART, the SDRAM (initialisation, read-latency calibration, a write/read test of the NPU's
// region) and the configuration flash (the model image). Drive it with
// sw/board/board_check.py or any terminal at 1 Mbaud: p, m, f, a.
//
// LEDs (active-low), 0..5:
//   0  blinks ~1.6 Hz from the 27 MHz crystal: configured, oscillator running
//   1  blinks ~3.2 Hz from the 54 MHz PLL: PLL locked
//   2  SDRAM initialised and calibrated
//   3  a test is running
//   4  the last test passed
//   5  the last test failed, or SDRAM calibration failed
// S1 resets (and re-runs the SDRAM calibration); S2 runs all tests.
//
// Needs gowin_pack --mspi_as_gpio so the configuration flash pins stay usable as I/O.

`default_nettype none

module bringup_top (
    input  wire logic       clk_27,
    input  wire logic       btn_s1,
    input  wire logic       btn_s2,
    output      logic [5:0] led_n,
    input  wire logic       uart_rx,
    output      logic       uart_tx,
    output      logic       flash_cs_n,
    output      logic       flash_clk,
    output      logic       flash_mosi,
    input  wire logic       flash_miso,
    // in-package SDRAM
    output      logic        O_sdram_clk,
    output      logic        O_sdram_cke,
    output      logic        O_sdram_cs_n,
    output      logic        O_sdram_cas_n,
    output      logic        O_sdram_ras_n,
    output      logic        O_sdram_wen_n,
    output      logic [3:0]  O_sdram_dqm,
    output      logic [10:0] O_sdram_addr,
    output      logic [1:0]  O_sdram_ba,
    inout  wire logic [31:0] IO_sdram_dq
);
    localparam int CLK_HZ = 54_000_000;

    logic clk, lock, rst;
    gw_pll_sys u_pll (.clk_27, .clk_sys(clk), .locked(lock));
    reset_sync u_rst (.clk, .rst_async_in(!lock || btn_s1), .rst_out(rst));

    // S2: synchronise, debounce (~5 ms), one pulse per press
    logic [1:0]  s2_sync;
    logic [17:0] s2_cnt;
    logic        s2_state, run_all;
    always_ff @(posedge clk) begin
        s2_sync <= {s2_sync[0], btn_s2};
        run_all <= 1'b0;
        if (rst) begin
            s2_cnt   <= '0;
            s2_state <= 1'b0;
        end else if (s2_sync[1] == s2_state) begin
            s2_cnt <= '0;
        end else if (&s2_cnt) begin
            s2_state <= s2_sync[1];
            run_all  <= s2_sync[1];
        end else begin
            s2_cnt <= s2_cnt + 18'd1;
        end
    end

    logic [31:0] sd_dq_o;
    logic        sd_dq_oe;
    assign IO_sdram_dq = sd_dq_oe ? sd_dq_o : 'z;
    gw_sdram_clk u_sdclk (.clk, .sdram_clk(O_sdram_clk));

    logic [4:0] status;             // {fail, pass, cal_err, sdram ready, busy}
    bringup #(.CLK_HZ(CLK_HZ)) u_bu (
        .clk, .rst, .run_all, .uart_rx, .uart_tx, .status,
        .spi_cs_n(flash_cs_n), .spi_clk(flash_clk), .spi_mosi(flash_mosi), .spi_miso(flash_miso),
        .sd_cke(O_sdram_cke), .sd_cs_n(O_sdram_cs_n), .sd_ras_n(O_sdram_ras_n),
        .sd_cas_n(O_sdram_cas_n), .sd_we_n(O_sdram_wen_n), .sd_ba(O_sdram_ba),
        .sd_addr(O_sdram_addr), .sd_dqm(O_sdram_dqm), .sd_dq_o, .sd_dq_oe, .sd_dq_i(IO_sdram_dq));

    logic [23:0] hb27;
    logic [23:0] hb54;
    always_ff @(posedge clk_27) hb27 <= hb27 + 24'd1;
    always_ff @(posedge clk) hb54 <= hb54 + 24'd1;

    assign led_n = ~{status[4] | status[2], status[3], status[0], status[1], hb54[23], hb27[23]};

endmodule

`default_nettype wire
