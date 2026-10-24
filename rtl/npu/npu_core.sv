// npu_core: instruction sequencer, on-chip memories, execution units, host/debug access.
//
// On chip: instruction memory and activation memory (which also holds the packed input
// image). Weights and params live in an external store (flash/SDRAM) reached through a
// burst-read port; the tile engine buffers what it needs.
//
// Host port: writes instruction or activation memory while the core is idle or paused.
// Debug port: reads activation memory while idle or paused. With step_mode set, the core
// pauses after every instruction until `resume`, so a testbench can check each layer.

`default_nettype none

module npu_core
    import npu_pkg::*;
(
    input  wire logic                clk,
    input  wire logic                rst,
    // control
    input  wire logic                start,
    input  wire logic                step_mode,
    input  wire logic                resume,
    output      logic                busy,          // running (not idle, not paused)
    output      logic                paused,
    output      logic                done,          // one-cycle pulse after END
    output      logic [IMEM_AW-1:0]  pc,
    output      logic [7:0]          result_class,
    output      logic signed [7:0]   result_score,
    output      logic [31:0]         cycles,        // cycles since start (excluding pauses)
    output      logic [31:0]         instr_cycles,  // cycles taken by the last instruction
    // host write port
    input  wire logic                host_we,
    input  wire logic [2:0]          host_sel,      // 0 imem, 2 act (others: not in the core)
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire logic [15:0]         host_addr,     // 16 bits for the external store; core uses fewer
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire logic [INSTR_W-1:0]  host_wdata,
    // debug read port (activation memory, 1-cycle latency)
    input  wire logic [ACT_AW-1:0]   dbg_addr,
    output      logic [WORD_W-1:0]   dbg_rdata,
    // external store (weights + params): burst reads
    output      logic                ext_req,
    output      logic [EXT_AW-1:0]   ext_addr,
    output      logic [15:0]         ext_len,
    input  wire logic                ext_ack,
    input  wire logic                ext_rvalid,
    input  wire logic [WORD_W-1:0]   ext_rdata
);

    localparam logic [2:0] SEL_IMEM = 3'd0, SEL_ACT = 3'd2;

    typedef enum logic [2:0] {Q_IDLE, Q_FETCH, Q_DECODE, Q_WAIT, Q_PAUSE, Q_NEXT} qstate_e;
    typedef enum logic [1:0] {U_NONE, U_TILE, U_POOL, U_ARGMAX} unit_e;
    qstate_e qs;
    unit_e   unit;
    instr_t  ins;
    logic    host_ok;

    assign host_ok = (qs == Q_IDLE) || (qs == Q_PAUSE);
    assign busy    = !host_ok;
    assign paused  = (qs == Q_PAUSE);

    // ---------------------------------------------------------------- memories
    logic [INSTR_W-1:0] imem_rdata;
    sdpram #(.W(INSTR_W), .D(IMEM_DEPTH)) u_imem (
        .clk, .we(host_ok && host_we && host_sel == SEL_IMEM), .waddr(IMEM_AW'(host_addr)),
        .wdata(host_wdata), .raddr(pc), .rdata(imem_rdata));

    logic [ACT_AW-1:0] act_raddr, act_waddr;
    logic [WORD_W-1:0] act_rdata, act_wdata;
    logic              act_we;
    sdpram #(.W(WORD_W), .D(ACT_DEPTH)) u_act (
        .clk, .we(act_we), .waddr(act_waddr), .wdata(act_wdata),
        .raddr(act_raddr), .rdata(act_rdata));
    assign dbg_rdata = act_rdata;

    // ---------------------------------------------------------------- execution units
    logic tile_start, tile_done, pool_start, pool_done;
    logic [ACT_AW-1:0] t_act_raddr, t_act_waddr, p_act_raddr, p_act_waddr, a_act_raddr;
    logic [WORD_W-1:0] t_act_wdata, p_act_wdata;
    logic              t_act_we, p_act_we;

    npu_tile_engine u_tile (
        .clk, .rst, .start(tile_start), .ins, .done(tile_done),
        .act_raddr(t_act_raddr), .act_rdata, .act_we(t_act_we), .act_waddr(t_act_waddr),
        .act_wdata(t_act_wdata), .ext_req, .ext_addr, .ext_len, .ext_ack, .ext_rvalid, .ext_rdata);

    npu_pool u_pool (
        .clk, .rst, .start(pool_start), .ins, .done(pool_done),
        .act_raddr(p_act_raddr), .act_rdata, .act_we(p_act_we), .act_waddr(p_act_waddr),
        .act_wdata(p_act_wdata));

    // ARGMAX: one value per cycle, first maximum wins
    logic [9:0]  am_i;            // value index being read
    logic        am_run, am_dv, am_done;
    logic [9:0]  am_di;
    logic signed [7:0] am_best;
    assign a_act_raddr = ACT_AW'(ins.in_base + 16'(am_i >> 3));

    always_ff @(posedge clk) begin
        am_dv   <= 1'b0;
        am_done <= 1'b0;
        if (am_dv) begin
            logic signed [7:0] v;
            v = 8'(act_rdata >> (8 * am_di[2:0]));
            if (am_di == 0 || v > am_best) begin
                am_best      <= v;
                result_class <= 8'(am_di);
                result_score <= v;
            end
            if (am_di == 10'(ins.in_w) - 10'd1) am_done <= 1'b1;
        end
        if (rst) begin
            am_run <= 1'b0;
        end else if (qs == Q_DECODE && op_e'(imem_rdata[3:0]) == OP_ARGMAX) begin   // `ins` updates this cycle
            am_run <= 1'b1;
            am_i   <= '0;
        end else if (am_run) begin
            am_dv <= 1'b1;
            am_di <= am_i;
            if (am_i == 10'(ins.in_w) - 10'd1) am_run <= 1'b0;
            else                               am_i   <= am_i + 10'd1;
        end
    end

    // ---------------------------------------------------------------- memory port muxing
    always_comb begin
        unique case (unit)
            U_TILE:   act_raddr = t_act_raddr;
            U_POOL:   act_raddr = p_act_raddr;
            U_ARGMAX: act_raddr = a_act_raddr;
            default:  act_raddr = dbg_addr;
        endcase
        if (host_ok && host_we && host_sel == SEL_ACT) begin
            act_we    = 1'b1;
            act_waddr = ACT_AW'(host_addr);
            act_wdata = host_wdata[WORD_W-1:0];
        end else if (unit == U_POOL) begin
            act_we    = p_act_we;
            act_waddr = p_act_waddr;
            act_wdata = p_act_wdata;
        end else begin
            act_we    = t_act_we && (unit == U_TILE);
            act_waddr = t_act_waddr;
            act_wdata = t_act_wdata;
        end
    end

    // ---------------------------------------------------------------- sequencer
    logic [31:0] instr_start;
    logic        unit_done;
    assign unit_done = (unit == U_TILE && tile_done) || (unit == U_POOL && pool_done)
                    || (unit == U_ARGMAX && am_done) || (unit == U_NONE);

    always_ff @(posedge clk) begin
        tile_start <= 1'b0;
        pool_start <= 1'b0;
        done       <= 1'b0;
        if (qs != Q_IDLE && qs != Q_PAUSE) cycles <= cycles + 32'd1;

        if (rst) begin
            qs   <= Q_IDLE;
            unit <= U_NONE;
            pc   <= '0;
        end else begin
            unique case (qs)
                Q_IDLE: if (start) begin
                    pc     <= '0;
                    cycles <= '0;
                    qs     <= Q_FETCH;
                end

                Q_FETCH: begin
                    instr_start <= cycles;
                    qs <= Q_DECODE;           // imem_rdata valid next cycle
                end

                Q_DECODE: begin
                    ins <= instr_t'(imem_rdata);
                    qs  <= Q_WAIT;
                    unique case (op_e'(imem_rdata[3:0]))
                        OP_CONV, OP_DWCONV: begin
                            unit <= U_TILE;
                            tile_start <= 1'b1;
                        end
                        OP_MAXPOOL, OP_AVGPOOL: begin
                            unit <= U_POOL;
                            pool_start <= 1'b1;
                        end
                        OP_ARGMAX: unit <= U_ARGMAX;
                        OP_END: begin
                            unit <= U_NONE;
                            done <= 1'b1;
                            qs   <= Q_IDLE;
                        end
                        default: unit <= U_NONE;   // NOP
                    endcase
                end

                Q_WAIT: if (unit_done) begin
                    instr_cycles <= cycles - instr_start;
                    unit <= U_NONE;
                    qs   <= step_mode ? Q_PAUSE : Q_NEXT;
                end

                Q_PAUSE: if (resume) qs <= Q_NEXT;

                Q_NEXT: begin
                    pc <= pc + IMEM_AW'(1);
                    qs <= Q_FETCH;
                end

                default: qs <= Q_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
