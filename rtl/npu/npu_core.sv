`default_nettype none

module npu_core
    import npu_pkg::*;
#(
    parameter int ACT_MEM  = ACT_DEPTH,
    parameter bit HAS_POOL = 1'b1
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
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire logic [15:0]         host_addr,
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire logic [WORD_W-1:0]   host_wdata,
    input  wire logic [ACT_AW-1:0]   dbg_addr,
    output      logic [WORD_W-1:0]   dbg_rdata,
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

    localparam int IMEM_BEATS = INSTR_W / 32;
    localparam int IB_AW      = $clog2(IMEM_DEPTH * IMEM_BEATS);
    logic [31:0]        imem_rdata32;
    logic [IB_AW-1:0]   imem_raddr;
    logic [INSTR_W-1:0] imem_rdata;
    assign ins = instr_t'(imem_rdata);
    logic [3:0]         fb;
    logic [IB_AW-1:0]   ib_base;
    sdpram #(.W(32), .D(IMEM_DEPTH * IMEM_BEATS)) u_imem (
        .clk, .we(host_ok && host_we && host_sel == SEL_IMEM), .waddr(IB_AW'(host_addr)),
        .wdata(host_wdata[31:0]), .raddr(imem_raddr), .rdata(imem_rdata32));
    assign imem_raddr = ib_base + IB_AW'(fb);

    logic [ACT_AW-1:0] act_raddr, act_waddr;
    logic [WORD_W-1:0] act_rdata, act_wdata;
    logic              act_we;
    localparam int ACT_PA = $clog2(ACT_MEM);
    sdpram #(.W(WORD_W), .D(ACT_MEM)) u_act (
        .clk, .we(act_we), .waddr(ACT_PA'(act_waddr)), .wdata(act_wdata),
        .raddr(ACT_PA'(act_raddr)), .rdata(act_rdata));
    assign dbg_rdata = act_rdata;

    logic tile_start, tile_done, pool_start, pool_done;
    logic [ACT_AW-1:0] t_act_raddr, t_act_waddr, p_act_raddr, p_act_waddr, a_act_raddr;
    logic [WORD_W-1:0] t_act_wdata, p_act_wdata;
    logic              t_act_we, p_act_we;

    npu_tile_engine u_tile (
        .clk, .rst, .start(tile_start), .ins, .done(tile_done),
        .act_raddr(t_act_raddr), .act_rdata, .act_we(t_act_we), .act_waddr(t_act_waddr),
        .act_wdata(t_act_wdata), .ext_req, .ext_addr, .ext_len, .ext_ack, .ext_rvalid, .ext_rdata);

    if (HAS_POOL) begin : g_pool
        npu_pool u_pool (
            .clk, .rst, .start(pool_start), .ins, .done(pool_done),
            .act_raddr(p_act_raddr), .act_rdata, .act_we(p_act_we), .act_waddr(p_act_waddr),
            .act_wdata(p_act_wdata));
    end else begin : g_no_pool
        assign pool_done = 1'b1;
        /* verilator lint_off UNUSEDSIGNAL */
        logic unused_start;
        /* verilator lint_on UNUSEDSIGNAL */
        assign unused_start = pool_start;
        assign {p_act_raddr, p_act_waddr, p_act_wdata, p_act_we} = '0;
    end

    logic [9:0]  am_i;
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
        end else if (qs == Q_DECODE && op_e'(imem_rdata[3:0]) == OP_ARGMAX) begin
            am_run <= 1'b1;
            am_i   <= '0;
        end else if (am_run) begin
            am_dv <= 1'b1;
            am_di <= am_i;
            if (am_i == 10'(ins.in_w) - 10'd1) am_run <= 1'b0;
            else                               am_i   <= am_i + 10'd1;
        end
    end

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
                    pc      <= '0;
                    ib_base <= '0;
                    cycles <= '0;
                    fb     <= '0;
                    qs     <= Q_FETCH;
                end

                Q_FETCH: begin
                    if (fb != 0) imem_rdata[32 * 3'(fb - 4'd1) +: 32] <= imem_rdata32;
                    if (fb == 4'(IMEM_BEATS)) begin
                        instr_start <= cycles;
                        fb <= '0;
                        qs <= Q_DECODE;
                    end else begin
                        fb <= fb + 4'd1;
                    end
                end

                Q_DECODE: begin
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
                        default: unit <= U_NONE;
                    endcase
                end

                Q_WAIT: if (unit_done) begin
                    instr_cycles <= cycles - instr_start;
                    unit <= U_NONE;
                    qs   <= step_mode ? Q_PAUSE : Q_NEXT;
                end

                Q_PAUSE: if (resume) qs <= Q_NEXT;

                Q_NEXT: begin
                    pc      <= pc + IMEM_AW'(1);
                    ib_base <= ib_base + IB_AW'(IMEM_BEATS);
                    qs      <= Q_FETCH;
                end

                default: qs <= Q_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
