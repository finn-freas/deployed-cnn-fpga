`default_nettype none

module mv_tb_top (
    input  wire logic         clk,
    input  wire logic         rst,
    input  wire logic         start,
    input  wire logic [11:0]  rows,
    input  wire logic [11:0]  cols,
    output      logic         busy,
    input  wire logic         s_valid,
    input  wire logic [63:0]  s_data,
    output      logic         s_ready,
    output      logic         o_valid,
    output      logic [15:0]  o_data,
    input  wire logic         xw_en,
    input  wire logic [5:0]   xw_addr,
    input  wire logic [255:0] xw_data
);
    logic [255:0] xram [64];
    logic [5:0]   x_addr;
    logic [255:0] x_rdata;

    always_ff @(posedge clk) begin
        if (xw_en) xram[xw_addr] <= xw_data;
        x_rdata <= xram[x_addr];
    end

    mv_unit u_mv (.*, .tab8(1'b0));
endmodule

`default_nettype wire
