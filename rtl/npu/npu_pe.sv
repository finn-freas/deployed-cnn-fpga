`default_nettype none

module npu_pe
    import npu_pkg::*;
(
    input  wire logic                     clk,
    input  wire logic                     rst,
    input  wire logic                     v_in,
    input  wire logic                     l_in,
    input  wire logic signed [DATA_W-1:0] a_in,
    input  wire logic signed [DATA_W-1:0] b_in,
    output      logic signed [DATA_W-1:0] b_out,
    output      logic signed [ACC_W-1:0]  shadow
);

    logic signed [ACC_W-1:0] acc, prod, sum;

    assign prod = ACC_W'(16'(a_in) * 16'(b_in));
    assign sum  = acc + prod;

    always_ff @(posedge clk) begin
        if (rst)       acc <= '0;
        else if (v_in) acc <= l_in ? '0 : sum;
        if (v_in && l_in) shadow <= sum;
        b_out <= b_in;
    end

endmodule

`default_nettype wire
