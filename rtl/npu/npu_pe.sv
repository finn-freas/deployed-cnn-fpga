// npu_pe: output-stationary MAC processing element with a shadow accumulator.
// A, valid and first move right; B moves down; all registered. When the first value of
// a new group arrives (v_in && f_in), the finished sum moves to `shadow` and the
// accumulator restarts with the new product, so the array never stops to drain.

`default_nettype none

module npu_pe
    import npu_pkg::*;
(
    input  wire logic                     clk,
    input  wire logic                     v_in,
    input  wire logic                     f_in,    // first value of a group (with v_in)
    input  wire logic signed [DATA_W-1:0] a_in,
    input  wire logic signed [DATA_W-1:0] b_in,
    output      logic                     v_out,
    output      logic                     f_out,
    output      logic signed [DATA_W-1:0] a_out,
    output      logic signed [DATA_W-1:0] b_out,
    output      logic signed [ACC_W-1:0]  shadow   // result of the previous group
);

    logic signed [ACC_W-1:0] acc;
    logic signed [ACC_W-1:0] prod;

    // Widen before multiplying: a cast operand is self-determined, so ACC_W'(a_in * b_in)
    // would compute an 8-bit product.
    assign prod = ACC_W'(16'(a_in) * 16'(b_in));

    always_ff @(posedge clk) begin
        if (v_in) begin
            if (f_in) begin
                shadow <= acc;
                acc    <= prod;
            end else begin
                acc <= acc + prod;
            end
        end
        a_out <= a_in;
        b_out <= b_in;
        v_out <= v_in;
        f_out <= f_in;
    end

endmodule

`default_nettype wire
