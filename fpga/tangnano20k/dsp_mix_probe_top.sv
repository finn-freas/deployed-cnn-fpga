// dsp_probe_top: synthesis/PnR probe -- can the open flow place the NPU's DSP load?
// 64 registered signed 8x8 multiply-accumulates (like the PE array) + one 32x32 multiply
// (like requant), fed from an LFSR, reduced to LEDs.

`default_nettype none

module dsp_mix_probe_top (
    input  wire logic       clk_27,
    input  wire logic       btn_s1,
    output      logic [5:0] led_n
);
    logic [63:0] lfsr = 64'h1;
    always_ff @(posedge clk_27) lfsr <= {lfsr[62:0], lfsr[63] ^ lfsr[62] ^ lfsr[60] ^ lfsr[59]};

    logic signed [31:0] acc [64];
    logic [63:0] red;
    for (genvar i = 0; i < 64; i++) begin : g_mac
        logic signed [7:0] a, b;
        always_ff @(posedge clk_27) begin
            a <= 8'(lfsr >> (i % 56));
            b <= 8'(lfsr >> ((i * 7) % 56));
            if (btn_s1) acc[i] <= '0;
            else        acc[i] <= acc[i] + 32'(16'(a) * 16'(b));
        end
        assign red[i] = ^acc[i];
    end

    logic signed [31:0] x, m;
    logic signed [63:0] p;
    always_ff @(posedge clk_27) begin
        x <= 32'(lfsr);
        m <= 32'(lfsr >> 32);
        p <= 64'(x) * 64'(m);
    end

    // 11 unsigned 14x10 address-style multiplies (map to MULT18X18), like the NPU's address math
    logic [23:0] q [11];
    logic [10:0] qred;
    for (genvar j = 0; j < 11; j++) begin : g_addr
        logic [13:0] u;
        logic [9:0]  v;
        always_ff @(posedge clk_27) begin
            u    <= 14'(lfsr >> j);
            v    <= 10'(lfsr >> (j + 20));
            q[j] <= 24'(u) * 24'(v);
        end
        assign qred[j] = ^q[j];
    end

    assign led_n = ~{^red[63:32], ^red[31:0], ^p, ^qred, 2'b0};
endmodule

`default_nettype wire
