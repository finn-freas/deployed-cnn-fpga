// npu_sim_top: NPU core + external store model, for simulation.
// Host port: sel 0 = instruction memory, 2 = activation memory, 3 = external store.

`default_nettype none

module npu_sim_top
    import npu_pkg::*;
#(
    parameter int EXT_LATENCY         = 6,
    parameter int EXT_CYCLES_PER_WORD = 2
) (
    input  wire logic                clk,
    input  wire logic                rst,
    input  wire logic                start,
    input  wire logic                step_mode,
    input  wire logic                resume,
    output      logic                busy,
    output      logic                paused,
    output      logic                done,
    output      logic [IMEM_AW-1:0]  pc,
    output      logic [7:0]          result_class,
    output      logic signed [7:0]   result_score,
    output      logic [31:0]         cycles,
    output      logic [31:0]         instr_cycles,
    input  wire logic                host_we,
    input  wire logic [2:0]          host_sel,
    input  wire logic [15:0]         host_addr,
    input  wire logic [INSTR_W-1:0]  host_wdata,
    input  wire logic [ACT_AW-1:0]   dbg_addr,
    output      logic [WORD_W-1:0]   dbg_rdata
);

    logic              ext_req, ext_ack, ext_rvalid;
    logic [EXT_AW-1:0] ext_addr;
    logic [15:0]       ext_len;
    logic [WORD_W-1:0] ext_rdata;

    npu_core u_core (
        .clk, .rst, .start, .step_mode, .resume, .busy, .paused, .done, .pc,
        .result_class, .result_score, .cycles, .instr_cycles,
        .host_we(host_we && host_sel != 3'd3), .host_sel, .host_addr, .host_wdata,
        .dbg_addr, .dbg_rdata,
        .ext_req, .ext_addr, .ext_len, .ext_ack, .ext_rvalid, .ext_rdata);

    ext_mem_model #(.LATENCY(EXT_LATENCY), .CYCLES_PER_WORD(EXT_CYCLES_PER_WORD)) u_ext (
        .clk, .rst, .req(ext_req), .addr(ext_addr), .len(ext_len), .ack(ext_ack),
        .rvalid(ext_rvalid), .rdata(ext_rdata),
        .we(host_we && host_sel == 3'd3), .waddr(EXT_AW'(host_addr)), .wdata(host_wdata[WORD_W-1:0]));

endmodule

`default_nettype wire
