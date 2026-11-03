`default_nettype none

module gw_sdram_clk (
    input  wire logic clk,
    output      logic sdram_clk
);
    ODDR u_oddr (
        .D0 (1'b0),
        .D1 (1'b1),
        .TX (1'b0),
        .CLK(clk),
        .Q0 (sdram_clk),
        .Q1 ()
    );
endmodule

`default_nettype wire
