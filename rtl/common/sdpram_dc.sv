`default_nettype none

module sdpram_dc #(
    parameter int W = 8,
    parameter int D = 1024,
    localparam int AW = $clog2(D)
) (
    input  wire logic          wclk,
    input  wire logic          we,
    input  wire logic [AW-1:0] waddr,
    input  wire logic [W-1:0]  wdata,
    input  wire logic          rclk,
    input  wire logic [AW-1:0] raddr,
    output      logic [W-1:0]  rdata
);

    logic [W-1:0] mem [D];

    always_ff @(posedge wclk) if (we) mem[waddr] <= wdata;
    always_ff @(posedge rclk) rdata <= mem[raddr];

endmodule

`default_nettype wire
