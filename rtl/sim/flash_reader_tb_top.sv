// flash_reader_tb_top: spi_flash_reader + spi_flash_model, for unit testing (simulation).

`default_nettype none

module flash_reader_tb_top #(
    parameter int CLK_DIV = 1
) (
    input  wire logic        clk,
    input  wire logic        rst,
    input  wire logic        req,
    input  wire logic [15:0] addr,
    input  wire logic [15:0] len,
    output      logic        ack,
    output      logic        rvalid,
    output      logic [63:0] rdata,
    input  wire logic        load_we,
    input  wire logic [22:0] load_addr,
    input  wire logic [63:0] load_data
);

    logic cs_n, sclk, mosi, miso;

    spi_flash_reader #(.AW(16), .FLASH_BASE(24'h40_0000), .CLK_DIV(CLK_DIV)) u_reader (
        .clk, .rst, .req, .addr, .len, .ack, .rvalid, .rdata,
        .spi_cs_n(cs_n), .spi_clk(sclk), .spi_mosi(mosi), .spi_miso(miso));

    spi_flash_model u_flash (
        .cs_n, .sclk, .mosi, .miso,
        .load_clk(clk), .load_we, .load_addr, .load_data);

endmodule

`default_nettype wire
