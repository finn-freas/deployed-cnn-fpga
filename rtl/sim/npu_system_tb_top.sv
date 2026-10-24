// npu_system_tb_top: npu_system with a behavioural SPI flash, for simulation.
// The flash is preloaded through `load_*` (byte address relative to FLASH_BASE).
// UART runs at clk/8 to keep the simulation short. EXT_SDRAM adds the SDRAM model, clocked
// 180 degrees from the system clock as on the board.

`default_nettype none

module npu_system_tb_top #(
    parameter bit EXT_SDRAM = 1'b0
) (
    input  wire logic        clk,
    input  wire logic        rst,
    input  wire logic        run,
    input  wire logic        uart_rx,
    output      logic        uart_tx,
    output      logic        booted,
    output      logic        boot_err,
    output      logic        busy,
    output      logic        res_valid,
    output      logic [7:0]  res_class,
    output      logic signed [7:0] res_score,
    output      logic [31:0] res_cycles,
    output      logic [15:0] res_count,
    input  wire logic [11:0] thumb_raddr,
    output      logic [3:0]  thumb_rdata,
    input  wire logic        load_we,
    input  wire logic [19:0] load_addr,     // word address
    input  wire logic [63:0] load_data
);
    localparam logic [23:0] FLASH_BASE = 24'h40_0000;
    logic cs_n, sclk, mosi, miso;

    logic        sd_cke, sd_cs_n, sd_ras_n, sd_cas_n, sd_we_n, sd_dq_oe, m_oe;
    logic [1:0]  sd_ba;
    logic [10:0] sd_addr;
    logic [3:0]  sd_dqm;
    logic [31:0] sd_dq_o, m_dq;

    npu_system #(.CLK_HZ(8_000_000), .BAUD(1_000_000), .FLASH_BASE(FLASH_BASE),
                 .EXT_SDRAM(EXT_SDRAM)) u_sys (
        .clk, .rst, .run, .uart_rx, .uart_tx,
        .spi_cs_n(cs_n), .spi_clk(sclk), .spi_mosi(mosi), .spi_miso(miso),
        .booted, .boot_err, .busy, .res_valid, .res_class, .res_score, .res_cycles, .res_count,
        .thumb_raddr, .thumb_rdata,
        .sd_cke, .sd_cs_n, .sd_ras_n, .sd_cas_n, .sd_we_n, .sd_ba, .sd_addr, .sd_dqm,
        .sd_dq_o, .sd_dq_oe, .sd_dq_i(m_oe ? m_dq : 32'hDEAD_BEEF));

    /* verilator lint_off UNUSEDSIGNAL */
    logic unused_oe;
    assign unused_oe = sd_dq_oe;
    /* verilator lint_on UNUSEDSIGNAL */
    sdram_model u_sdram (
        .clk(!clk), .cke(sd_cke), .cs_n(sd_cs_n), .ras_n(sd_ras_n), .cas_n(sd_cas_n), .we_n(sd_we_n),
        .ba(sd_ba), .addr(sd_addr), .dqm(sd_dqm), .dq_i(sd_dq_o), .dq_o(m_dq), .dq_oe(m_oe));

    spi_flash_model u_flash (
        .cs_n, .sclk, .mosi, .miso, .load_clk(clk), .load_we,
        .load_addr(23'(FLASH_BASE + {load_addr, 3'b000})), .load_data);

endmodule

`default_nettype wire
