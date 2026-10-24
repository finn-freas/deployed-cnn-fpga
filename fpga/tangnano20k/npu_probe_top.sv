// npu_probe_top: synthesis-only probe of the NPU core's resource use on the GW2AR-18.
// Not a working board design: the host/debug ports are driven from a tiny LFSR and the
// outputs are XOR-reduced to an LED, just enough to keep synthesis from removing logic.
// Memory depths are on-chip sizes (the weight store is a 2K-word stand-in for the
// streaming weight buffer that V3 adds).

`default_nettype none

module npu_probe_top (
    input  wire logic       clk_27,
    input  wire logic       btn_s1,
    output      logic [5:0] led_n
);
    import npu_pkg::*;

    logic [31:0] lfsr = 32'h1;
    always_ff @(posedge clk_27) lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};

    logic busy, paused, done;
    logic [IMEM_AW-1:0] pc;
    logic [7:0] cls;
    logic signed [7:0] score;
    logic [31:0] cycles, instr_cycles;
    logic [WORD_W-1:0] dbg_rdata;

    npu_core #(.ACT_MEM(8192), .W_MEM(2048), .P_MEM(256), .IN_MEM(32768)) u_core (
        .clk(clk_27), .rst(btn_s1),
        .start(lfsr[3]), .step_mode(1'b0), .resume(1'b0),
        .busy, .paused, .done, .pc, .result_class(cls), .result_score(score),
        .cycles, .instr_cycles,
        .host_we(lfsr[5]), .host_sel(lfsr[8:6]), .host_addr(lfsr[24:9]),
        .host_wdata({5{lfsr}}),
        .dbg_addr(lfsr[ACT_AW-1:0]), .dbg_rdata);

    assign led_n = ~{busy, paused, done, ^pc, ^{cls, score, cycles, instr_cycles}, ^dbg_rdata};

endmodule

`default_nettype wire
