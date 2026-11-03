`default_nettype none

module vmem #(
    parameter int ROWS = 256
) (
    input  wire logic                    clk,
    input  wire logic [$clog2(ROWS)-1:0] raddr,
    output      logic [255:0]            rdata,
    input  wire logic                    we,
    input  wire logic [$clog2(ROWS)-1:0] waddr,
    input  wire logic [15:0]             wmask,
    input  wire logic [255:0]            wdata
);
    for (genvar k = 0; k < 16; k++) begin : g_lane
        logic [15:0] mem [ROWS];
        always_ff @(posedge clk) begin
            if (we && wmask[k]) mem[waddr] <= wdata[16*k +: 16];
            rdata[16*k +: 16] <= mem[raddr];
        end
    end
endmodule

`default_nettype wire
