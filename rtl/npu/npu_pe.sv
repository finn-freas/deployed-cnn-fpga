// npu_pe: output-stationary MAC processing element.
// A (and its valid) move right, B moves down, both registered; the int32 accumulator stays.

`default_nettype none

module npu_pe
    import npu_pkg::*;
(
    input  wire logic                     clk,
    input  wire logic                     clear,   // zero the accumulator (has priority)
    input  wire logic                     v_in,
    input  wire logic signed [DATA_W-1:0] a_in,
    input  wire logic signed [DATA_W-1:0] b_in,
    output      logic                     v_out,
    output      logic signed [DATA_W-1:0] a_out,
    output      logic signed [DATA_W-1:0] b_out,
    output      logic signed [ACC_W-1:0]  acc
);

    always_ff @(posedge clk) begin
        if (clear)     acc <= '0;
        // Widen before multiplying: a cast operand is self-determined, so ACC_W'(a_in * b_in)
        // would compute an 8-bit product.
        else if (v_in) acc <= acc + ACC_W'(16'(a_in) * 16'(b_in));
        a_out <= a_in;
        b_out <= b_in;
        v_out <= v_in;
    end

endmodule

`default_nettype wire
