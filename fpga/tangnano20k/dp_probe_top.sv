`default_nettype none

module dp_probe_top (
    input  wire logic       clk_27,
    input  wire logic       btn_s1,
    output      logic [5:0] led_n
);
    import npu_pkg::*;

    logic [63:0] lfsr = 64'h1;
    always_ff @(posedge clk_27) lfsr <= {lfsr[62:0], lfsr[63] ^ lfsr[62] ^ lfsr[60] ^ lfsr[59]};

    logic signed [DATA_W-1:0] a [N], b [N];
    logic signed [ACC_W-1:0]  sh [N][N];
    always_comb for (int i = 0; i < N; i++) begin
        a[i] = DATA_W'(lfsr >> (8 * i));
        b[i] = DATA_W'(lfsr >> ((8 * i + 3) % 56));
    end
    npu_array u_array (.clk(clk_27), .valid(lfsr[0]), .first(lfsr[1] & btn_s1), .a, .b, .shadow(sh));

    logic [5:0] sel;
    logic signed [ACC_W-1:0] s0, s1;
    always_ff @(posedge clk_27) begin
        sel <= sel + 6'd1;
        s0  <= sh[sel[5:3]][sel[2:0]];
        s1  <= sh[sel[5:3]][~sel[2:0]];
    end
    logic signed [7:0] o0, o1;
    npu_requant u_rq0 (.clk(clk_27), .acc(s0), .bias(32'(lfsr)), .mult(32'(lfsr >> 16)),
        .shift(SHIFT_W'(lfsr >> 40)), .rmode(lfsr[50]), .out_zp(8'(lfsr >> 20)),
        .act_min(-8'sd128), .act_max(8'sd127), .out(o0));
    npu_requant u_rq1 (.clk(clk_27), .acc(s1), .bias(32'(lfsr >> 8)), .mult(32'(lfsr >> 24)),
        .shift(SHIFT_W'(lfsr >> 44)), .rmode(lfsr[51]), .out_zp(8'(lfsr >> 28)),
        .act_min(-8'sd128), .act_max(8'sd127), .out(o1));

    assign led_n = ~{^o0, ^o1, 4'b0};
endmodule

`default_nettype wire
