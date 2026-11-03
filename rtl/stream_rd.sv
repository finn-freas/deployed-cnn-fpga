`default_nettype none

module stream_rd #(
    parameter int DEPTH = 1024,
    parameter int CHUNK = 512
) (
    input  wire logic        clk,
    input  wire logic        rst,
    input  wire logic        start,
    input  wire logic [19:0] addr,
    input  wire logic [19:0] len,

    output      logic        rd_req,
    output      logic [19:0] rd_addr,
    output      logic [15:0] rd_len,
    input  wire logic        rd_ack,
    input  wire logic        rvalid,
    input  wire logic [63:0] rdata,

    output      logic        s_valid,
    output      logic [63:0] s_data,
    input  wire logic        s_ready
);
    localparam int AW = $clog2(DEPTH);

    logic [63:0] mem [DEPTH];
    logic [AW:0] wp, rp;
    logic [AW:0] pending;
    logic [19:0] next, left;

    wire [AW:0] used = wp - rp;
    wire [AW+1:0] free = (AW+2)'(DEPTH) - (AW+2)'(used) - (AW+2)'(pending);
    wire [19:0] want = (left > 20'(CHUNK)) ? 20'(CHUNK) : left;

    logic        out_v;
    logic [63:0] out_d;
    wire  take   = out_v && s_ready;
    wire  refill = (used != 0) && (!out_v || take);
    assign s_valid = out_v;
    assign s_data  = out_d;

    always_ff @(posedge clk) begin
        if (rvalid) mem[wp[AW-1:0]] <= rdata;
        if (refill) out_d <= mem[rp[AW-1:0]];
    end

    always_ff @(posedge clk) begin
        if (rst || start) begin
            wp      <= '0;
            rp      <= '0;
            out_v   <= 1'b0;
            pending <= '0;
            rd_req  <= 1'b0;
            next    <= addr;
            left    <= start ? len : '0;
        end else begin
            if (rvalid) wp <= wp + 1'b1;
            if (refill) begin
                rp    <= rp + 1'b1;
                out_v <= 1'b1;
            end else if (take) begin
                out_v <= 1'b0;
            end
            if (rd_req && rd_ack) begin
                rd_req <= 1'b0;
            end else if (!rd_req && left != 0 && free >= (AW+2)'(want)) begin
                rd_req  <= 1'b1;
                rd_addr <= next;
                rd_len  <= 16'(want);
                next    <= next + want;
                left    <= left - want;
            end
            pending <= pending + ((!rd_req && left != 0 && free >= (AW+2)'(want)) ? (AW+1)'(want) : '0)
                               - (rvalid ? (AW+1)'(1) : '0);
        end
    end
endmodule

`default_nettype wire
