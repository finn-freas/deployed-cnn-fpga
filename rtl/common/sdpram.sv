// sdpram: simple dual-port RAM, one write port, one synchronous read port (1-cycle latency).
// Infers block RAM. Read-during-write to the same address returns the old data.

`default_nettype none

module sdpram #(
    parameter int W = 64,
    parameter int D = 1024,
    localparam int AW = $clog2(D)
) (
    input  wire logic          clk,
    input  wire logic          we,
    input  wire logic [AW-1:0] waddr,
    input  wire logic [W-1:0]  wdata,
    input  wire logic [AW-1:0] raddr,
    output      logic [W-1:0]  rdata
);

    logic [W-1:0] mem [D];

    always_ff @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        rdata <= mem[raddr];
    end

endmodule

`default_nettype wire
