// npu_probe_top: synthesis-only probe of the NPU core's resource use on the GW2AR-18.
// Not a working board design: host/debug ports and the external-store port are driven
// from an LFSR, and the outputs are XOR-reduced to LEDs, just enough to keep synthesis
// from removing logic. The real flash/SDRAM controller will sit on the external port.

`default_nettype none

module npu_probe_small_top (
    input  wire logic       clk_27,
    input  wire logic       btn_s1,
    output      logic [5:0] led_n
);
    import npu_pkg::*;

    logic [63:0] lfsr = 64'h1;
    always_ff @(posedge clk_27) lfsr <= {lfsr[62:0], lfsr[63] ^ lfsr[62] ^ lfsr[60] ^ lfsr[59]};

    logic busy, paused, done;
    logic [IMEM_AW-1:0] pc;
    logic [7:0] cls;
    logic signed [7:0] score;
    logic [31:0] cycles, instr_cycles;
    logic [WORD_W-1:0] dbg_rdata;
    logic ext_req;
    logic [EXT_AW-1:0] ext_addr;
    logic [15:0] ext_len;

    npu_core #(.ACT_MEM(3584)) u_core (
        .clk(clk_27), .rst(btn_s1),
        .start(lfsr[3]), .step_mode(1'b0), .resume(1'b0),
        .busy, .paused, .done, .pc, .result_class(cls), .result_score(score),
        .cycles, .instr_cycles,
        .host_we(lfsr[5]), .host_sel(lfsr[8:6]), .host_addr(lfsr[24:9]),
        .host_wdata(lfsr),
        .dbg_addr(lfsr[ACT_AW-1:0]), .dbg_rdata,
        .ext_req, .ext_addr, .ext_len, .ext_ack(ext_req & lfsr[40]),
        .ext_rvalid(lfsr[41]), .ext_rdata(lfsr));

    assign led_n = ~{busy, paused, done, ^pc, ^{cls, score, cycles, instr_cycles, ext_addr, ext_len},
                     ^dbg_rdata};

endmodule

`default_nettype wire
