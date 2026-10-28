`default_nettype none

module ext_mem_model
    import npu_pkg::*;
#(
    parameter int LATENCY         = 6,
    parameter int CYCLES_PER_WORD = 2
) (
    input  wire logic              clk,
    input  wire logic              rst,
    input  wire logic              req,
    input  wire logic [EXT_AW-1:0] addr,
    input  wire logic [15:0]       len,
    output      logic              ack,
    output      logic              rvalid,
    output      logic [WORD_W-1:0] rdata,
    input  wire logic              we,
    input  wire logic [EXT_AW-1:0] waddr,
    input  wire logic [WORD_W-1:0] wdata
);

    logic [WORD_W-1:0] mem [EXT_DEPTH];
    logic              busy;
    logic [EXT_AW-1:0] ptr;
    logic [15:0]       left;
    logic [15:0]       wait_cnt;

    assign ack = req && !busy;

    always_ff @(posedge clk) begin
        rvalid <= 1'b0;
        if (we) mem[waddr] <= wdata;
        if (rst) begin
            busy <= 1'b0;
        end else if (ack) begin
            busy     <= (len != 0);
            ptr      <= addr;
            left     <= len;
            wait_cnt <= 16'(LATENCY);
        end else if (busy) begin
            if (wait_cnt != 0) begin
                wait_cnt <= wait_cnt - 16'd1;
            end else begin
                rvalid   <= 1'b1;
                rdata    <= mem[ptr];
                ptr      <= ptr + EXT_AW'(1);
                left     <= left - 16'd1;
                wait_cnt <= 16'(CYCLES_PER_WORD - 1);
                if (left == 16'd1) busy <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire
