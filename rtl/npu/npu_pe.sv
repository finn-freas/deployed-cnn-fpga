// npu_pe: output-stationary MAC processing element with a shadow accumulator.
// B moves down (registered); A, valid and last come from the row. With the last value of a
// group (v_in && l_in), the finished sum acc + a*b goes to `shadow` and the accumulator
// restarts from zero, so the array never stops to drain.
//
// The shadow takes the adder's output directly, so on Gowin it packs with the adder in one
// slice (a register fed by another register would need a pass-through LUT).

`default_nettype none

module npu_pe
    import npu_pkg::*;
(
    input  wire logic                     clk,
    input  wire logic                     rst,
    input  wire logic                     v_in,
    input  wire logic                     l_in,    // last value of a group (with v_in)
    input  wire logic signed [DATA_W-1:0] a_in,
    input  wire logic signed [DATA_W-1:0] b_in,
    output      logic signed [DATA_W-1:0] b_out,
    output      logic signed [ACC_W-1:0]  shadow   // result of the last finished group
);

    logic signed [ACC_W-1:0] acc, prod, sum;

    // Widen before multiplying: a cast operand is self-determined, so ACC_W'(a_in * b_in)
    // would compute an 8-bit product.
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
