// mem_probe_top: place-and-route probe of the NPU's block RAMs alone (same shapes as the
// core: activation 7168x64, A_T 2048x64, weight buffer 1024x64, param buffer 256x70,
// instruction memory 320x32), driven from an LFSR.

`default_nettype none

module memdp_probe_top (
    input  wire logic       clk_27,
    input  wire logic       btn_s1,
    output      logic [5:0] led_n
);
    logic [63:0] lfsr = 64'h1;
    always_ff @(posedge clk_27) lfsr <= {lfsr[62:0], lfsr[63] ^ lfsr[62] ^ lfsr[60] ^ lfsr[59]};

    logic [63:0] q_act, q_at, q_wb;
    logic [69:0] q_pb;
    logic [31:0] q_im;

    sdpram #(.W(64), .D(7168)) u_act (.clk(clk_27), .we(lfsr[0] & btn_s1), .waddr(13'(lfsr >> 1)),
        .wdata(lfsr), .raddr(13'(lfsr >> 14)), .rdata(q_act));
    sdpram #(.W(64), .D(2048)) u_at  (.clk(clk_27), .we(lfsr[2]), .waddr(11'(lfsr >> 3)),
        .wdata({lfsr[31:0], lfsr[63:32]}), .raddr(11'(lfsr >> 17)), .rdata(q_at));
    sdpram #(.W(64), .D(1024)) u_wb  (.clk(clk_27), .we(lfsr[4]), .waddr(10'(lfsr >> 5)),
        .wdata(~lfsr), .raddr(10'(lfsr >> 19)), .rdata(q_wb));
    sdpram #(.W(70), .D(256))  u_pb  (.clk(clk_27), .we(lfsr[6]), .waddr(8'(lfsr >> 7)),
        .wdata({lfsr[5:0], lfsr}), .raddr(8'(lfsr >> 23)), .rdata(q_pb));
    sdpram #(.W(32), .D(320))  u_im  (.clk(clk_27), .we(lfsr[8]), .waddr(9'(lfsr >> 9)),
        .wdata(lfsr[40:9]), .raddr(9'(lfsr >> 29)), .rdata(q_im));

    logic [5:0] dp_led;
    dp_probe_top u_dp (.clk_27, .btn_s1, .led_n(dp_led));
    assign led_n = ~{^q_act, ^q_at, ^q_wb, ^q_pb, ^q_im, ^dp_led};
endmodule

`default_nettype wire
