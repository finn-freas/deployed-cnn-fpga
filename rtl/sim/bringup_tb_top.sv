// bringup_tb_top: the bring-up tests with the behavioural SDRAM and SPI flash, for
// simulation. UART at clk/8; a smaller SDRAM test range; `inject` flips SDRAM data bit 5 on
// the way back (a stuck/broken DQ line) to exercise the failure report.

`default_nettype none

module bringup_tb_top #(
    parameter int MEM_WORDS = 4096
) (
    input  wire logic        clk,
    input  wire logic        rst,
    input  wire logic        run_all,
    input  wire logic        uart_rx,
    output      logic        uart_tx,
    output      logic [4:0]  status,
    input  wire logic        inject,
    input  wire logic        load_we,
    input  wire logic [19:0] load_addr,     // word address relative to FLASH_BASE
    input  wire logic [63:0] load_data
);
    localparam logic [23:0] FLASH_BASE = 24'h40_0000;
    logic cs_n, sclk, mosi, miso;
    logic        sd_cke, sd_cs_n, sd_ras_n, sd_cas_n, sd_we_n, sd_dq_oe, m_oe;
    logic [1:0]  sd_ba;
    logic [10:0] sd_addr;
    logic [3:0]  sd_dqm;
    logic [31:0] sd_dq_o, m_dq;

    bringup #(.CLK_HZ(8_000_000), .BAUD(1_000_000), .FLASH_BASE(FLASH_BASE),
              .MEM_WORDS(MEM_WORDS)) u_bu (
        .clk, .rst, .run_all, .uart_rx, .uart_tx, .status,
        .spi_cs_n(cs_n), .spi_clk(sclk), .spi_mosi(mosi), .spi_miso(miso),
        .sd_cke, .sd_cs_n, .sd_ras_n, .sd_cas_n, .sd_we_n, .sd_ba, .sd_addr, .sd_dqm,
        .sd_dq_o, .sd_dq_oe,
        .sd_dq_i(m_oe ? (m_dq ^ (inject ? 32'h20 : 32'h0)) : 32'hDEAD_BEEF));

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
