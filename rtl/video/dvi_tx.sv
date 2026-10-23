// dvi_tx: RGB + sync -> three TMDS 10-bit symbols (portable, no vendor primitives).
// Channel 0 = blue (carries hsync/vsync), 1 = green, 2 = red. Latency: 1 cycle.

`default_nettype none

module dvi_tx (
    input  wire logic       clk_pix,
    input  wire logic       rst,
    input  wire logic       de,
    input  wire logic       hsync,
    input  wire logic       vsync,
    input  wire logic [7:0] r,
    input  wire logic [7:0] g,
    input  wire logic [7:0] b,
    output      logic [9:0] tmds [3]
);

    tmds_encoder u_enc_b (.clk(clk_pix), .rst, .de, .d(b), .c0(hsync), .c1(vsync), .q(tmds[0]));
    tmds_encoder u_enc_g (.clk(clk_pix), .rst, .de, .d(g), .c0(1'b0),  .c1(1'b0),  .q(tmds[1]));
    tmds_encoder u_enc_r (.clk(clk_pix), .rst, .de, .d(r), .c0(1'b0),  .c1(1'b0),  .q(tmds[2]));

endmodule

`default_nettype wire
