// npu_tile_engine: executes CONV and DWCONV instructions.
//
// Memory system (ISA v3):
//   * params for the whole layer are burst-read from the external store into a param
//     buffer at the start of the instruction;
//   * weights are burst-read into an on-chip weight buffer: the whole layer if it fits
//     ("cached" mode, tile-outer loop), otherwise one output group at a time ("group"
//     mode, group-outer loop, inputs re-gathered per group);
//   * im2col rows for a tile of N output pixels are gathered into a transposed buffer
//     A_T, where word k holds byte k of all N rows, so streaming reads one word per k.
//     Word-mode gathers (channel-blocked input) pass through a double-buffered N x N
//     byte transposer at one word per cycle; packed-mode gathers (uint8 HWC input in
//     activation memory) assemble each A_T word from N bytes, one per cycle.
//
// Compute:
//   * O3, overlapped drain: a "job" is one output group of one tile, streamed as K values
//     with the first one flagged. The flagged value moves every PE's finished sum into its
//     shadow register as it passes, so jobs stream back to back while a separate drain
//     unit reads the previous job's shadows through requant into activation memory.
//     Injections are spaced at least PERIOD_MIN cycles apart so a shadow is never
//     overwritten before the drain has read it.
//   * Double-buffered A_T: a producer FSM gathers the next "unit" (a tile; for DWCONV a
//     tile's channel block) into one half while a consumer FSM streams the jobs of the
//     other half. The next job's first value captures the previous one; a zero-valued
//     capture wave is only injected after the instruction's last job.
// DWCONV runs as a CONV over one channel block with block-diagonal weights expanded on
// the fly from the compact depthwise weights.

`default_nettype none

module npu_tile_engine
    import npu_pkg::*;
(
    input  wire logic                clk,
    input  wire logic                rst,
    input  wire logic                start,
    input  wire instr_t              ins,
    output      logic                done,
    // activation memory
    output      logic [ACT_AW-1:0]   act_raddr,
    input  wire logic [WORD_W-1:0]   act_rdata,
    output      logic                act_we,
    output      logic [ACT_AW-1:0]   act_waddr,
    output      logic [WORD_W-1:0]   act_wdata,
    // external store: burst reads
    output      logic                ext_req,
    output      logic [EXT_AW-1:0]   ext_addr,
    output      logic [15:0]         ext_len,
    input  wire logic                ext_ack,       // request accepted
    input  wire logic                ext_rvalid,
    input  wire logic [WORD_W-1:0]   ext_rdata
);

    localparam int KA_AW       = $clog2(K_MAX);      // address within one A_T half
    localparam int RQ_LAT      = 4;                  // npu_requant latency
    // Timing (injection issued in cycle i reaches PE(r,c) in cycle i+1+r+c):
    //   the drain has two requant pipelines and reads a column pair per cycle, starting in
    //   cycle i+DRAIN_DELAY+1: PE(r,c) is read (N/2)*r + c/2 cycles later. That must be after
    //   its capture in cycle i+1+r+c, i.e. DRAIN_DELAY > ceil(c/2) - (N/2-1)*r for all r,c:
    //   DRAIN_DELAY >= N/2 + 1 (the worst case is row 0, column N-1);
    //   a later injection in cycle i2 overwrites PE(r,c)'s shadow in cycle i2+1+r+c, which
    //   must follow that read; and since that injection also hands the next job to the
    //   drain, this drain's N*N/2 issues must be finished: i2 >= i + DRAIN_DELAY + N*N/2 + 1
    //   (this bound also covers the overwrite condition).
    localparam int DRAIN_DELAY = N / 2 + 1;
    localparam int PERIOD_MIN  = DRAIN_DELAY + N * N / 2 + 1;
    localparam int DQ          = RQ_LAT + 1;         // drain issue -> requant output

    // ---------------------------------------------------------------- main FSM
    typedef enum logic [3:0] {
        S_IDLE, S_INIT, S_PLOAD_REQ, S_PLOAD, S_WLOAD_REQ, S_WLOAD, S_RUN, S_FINISH, S_DONE
    } state_e;
    state_e state;

    /* verilator lint_off UNUSEDSIGNAL */
    instr_t I;                      // each unit uses a subset of the fields
    /* verilator lint_on UNUSEDSIGNAL */
    logic        is_dw, pk_mode, cached;
    logic [5:0]  taps;              // ksize^2
    logic [10:0] K;                 // reduction length
    logic [7:0]  KW;                // A words per pixel in word mode (K / N)
    logic [13:0] P;                 // output pixels
    logic [16:0] layer_words;       // weight words of the whole layer
    logic        run_start;         // pulse: producer and consumer begin

    // ---------------------------------------------------------------- external loads
    logic [16:0] ld_cnt, ld_len;
    logic [WORD_W-1:0] p_word0;
    logic        w_loading;         // a weight burst is landing in the weight buffer

    // ---------------------------------------------------------------- buffers
    logic                 wb_we;
    logic [WBUF_AW-1:0]   wb_waddr, wb_raddr;
    logic [WORD_W-1:0]    wb_rdata;
    sdpram #(.W(WORD_W), .D(WBUF_DEPTH)) u_wbuf (
        .clk, .we(wb_we), .waddr(wb_waddr), .wdata(ext_rdata), .raddr(wb_raddr), .rdata(wb_rdata));

    logic                 pb_we;
    logic [PBUF_AW-1:0]   pb_waddr, pb_raddr;
    logic [PARAM_W-1:0]   pb_wdata, pb_rdata;
    sdpram #(.W(PARAM_W), .D(PBUF_DEPTH)) u_pbuf (
        .clk, .we(pb_we), .waddr(pb_waddr), .wdata(pb_wdata), .raddr(pb_raddr), .rdata(pb_rdata));

    logic                 at_we;
    logic [KA_AW:0]       at_waddr, at_raddr;     // {half, k}
    logic [WORD_W-1:0]    at_wdata, at_rdata;
    sdpram #(.W(WORD_W), .D(2 * K_MAX)) u_at (
        .clk, .we(at_we), .waddr(at_waddr), .wdata(at_wdata), .raddr(at_raddr), .rdata(at_rdata));

    logic [1:0]  full;              // A_T half holds a gathered unit
    logic [13:0] unit_p0 [2];       // per half: first output pixel of the unit's tile
    logic [5:0]  unit_g  [2];       // per half: channel block (DWCONV) / group (group mode)
    logic        unit_last [2];     // per half: last unit of the instruction

    function automatic logic [6:0] step_y(input logic [6:0] y, input logic [6:0] x, input logic [6:0] w);
        return (x == w - 7'd1) ? y + 7'd1 : y;
    endfunction
    function automatic logic [6:0] step_x(input logic [6:0] x, input logic [6:0] w);
        return (x == w - 7'd1) ? 7'd0 : x + 7'd1;
    endfunction

    // ---------------------------------------------------------------- producer (gather)
    typedef enum logic [2:0] {P_IDLE, P_WAIT_BUF, P_SETUP, P_GATHER, P_TAIL} pstate_e;
    pstate_e pstate;
    logic        gbuf;              // A_T half being filled
    logic [13:0] p_p0;              // unit's tile: first output pixel
    logic [6:0]  p_oy, p_ox;        // unit's tile: first pixel coords
    logic [6:0]  p_noy, p_nox;      // next tile's first pixel coords
    logic [5:0]  p_g;               // unit's channel block / group
    logic        p_last_tile, p_last_g, p_new_tile;
    assign p_last_tile = (p_p0 + 14'(N)) >= P;
    assign p_last_g    = (p_g == I.out_g - 6'd1);

    logic [6:0]  row_py [N];
    logic [6:0]  row_px [N];
    logic [N-1:0] row_valid;
    logic [6:0]  cy, cx;            // coordinate walker (tile setup)
    logic [3:0]  sr;                // setup row counter

    // Gather order: k-word outer, rows inner. Word mode reads one channel-blocked word per
    // (k-word, row). Packed mode (O2): each k-word is a contiguous 8-byte window of one input
    // row (the compiler pads each kernel row to KR), read as two aligned words (phase 0/1).
    logic [2:0]  gr;                // row being read
    logic [10:0] gk;                // k-word
    logic        ph;                // packed mode: which of the two aligned words
    logic [3:0]  ky;
    logic [2:0]  kx;                // word mode: kernel column
    logic [5:0]  kc;                // word mode: channel block
    logic [3:0]  kj;                // packed mode: window within the kernel row (0..KRW-1)
    logic [3:0]  krw;               // packed mode: KR / N
    logic [8:0]  ksc;               // packed mode: ksize * C (real bytes per kernel row)

    logic signed [9:0]  iy, ix, ix0;
    logic               inb, row_ok;
    logic [23:0]        pix_lin;
    logic signed [22:0] win_addr;   // packed: byte address of the window (may be < 0 at the edge)
    logic signed [9:0]  lo_m, hi_m; // packed: window bytes m with lo_m <= m < hi_m are in bounds
    always_comb begin
        iy  = 10'(signed'({3'b0, row_py[gr]}) * signed'({8'b0, I.stride})) + 10'(ky) - 10'(I.pad);
        ix0 = 10'(signed'({3'b0, row_px[gr]}) * signed'({8'b0, I.stride})) - 10'(I.pad);
        ix  = ix0 + 10'(kx);
        row_ok  = row_valid[gr] && iy >= 0 && iy < 10'(I.in_h);
        inb     = row_ok && ix >= 0 && ix < 10'(I.in_w);
        pix_lin = 24'(unsigned'(iy)) * 24'(I.in_w) + 24'(unsigned'(ix));
        win_addr = 23'(signed'({1'b0, I.in_base, 3'b000}))
                 + 23'((23'(signed'(iy)) * 23'(signed'({1'b0, I.in_w})) + 23'(ix0)) * 23'(signed'({1'b0, I.in_cb})))
                 + 23'(signed'({1'b0, kj, 3'b000}));
        lo_m = (ix0 < 0) ? 10'(-ix0) * 10'(I.in_cb) : 10'sd0;
        hi_m = 10'(10'(signed'({1'b0, I.in_w})) - ix0) * 10'(I.in_cb);
        if (pk_mode) act_raddr = ACT_AW'((win_addr >>> 3) + 23'(ph));
        else         act_raddr = ACT_AW'(24'(I.in_base) + pix_lin * 24'(I.in_cb) + 24'(is_dw ? p_g : kc));
    end

    // read-data pipeline (act memory has 1-cycle latency)
    logic        d_v, d_pad, d_ph;
    logic [2:0]  d_r, d_sh;
    logic [7:0]  d_k;               // k-word
    logic        d_rowok;
    logic [3:0]  d_kj;
    logic signed [9:0] d_lo, d_hi;
    logic [WORD_W-1:0] w0;          // packed: first aligned word of the window

    // double-buffered transposer (both modes). Packed array on purpose: it is read N ways at
    // once (one column of bytes per cycle), so as an unpacked array synthesis would infer a
    // multi-port LUT RAM and replicate it per read port; as a vector it is plain flip-flops.
    logic [1:0][N-1:0][WORD_W-1:0] tb;
    logic        tb_sel;            // buffer being filled
    logic        wo_active, wo_buf;
    logic [2:0]  wo_cnt;
    logic [7:0]  wo_kw;

    // packed mode: assemble a row's window from the two aligned words, applying padding
    logic [WORD_W-1:0] pk_row;
    always_comb begin
        logic [WORD_W-1:0]   win;
        logic signed [9:0]   m;
        win = WORD_W'({act_rdata, w0} >> (8 * d_sh));
        for (int b = 0; b < N; b++) begin
            m = 10'(signed'({1'b0, d_kj, 3'b000})) + 10'(b);
            if (m >= 10'(signed'({1'b0, ksc})))   pk_row[8 * b +: 8] = 8'd0;        // past the kernel row
            else if (!d_rowok || m < d_lo || m >= d_hi) pk_row[8 * b +: 8] = I.in_zp;   // padding
            else                                       pk_row[8 * b +: 8] = 8'(win >> (8 * b)) + I.in_zp;
        end
    end

    always_comb begin
        at_we    = wo_active;
        at_waddr = {gbuf, KA_AW'({wo_kw, wo_cnt})};
        at_wdata = '0;
        for (int r = 0; r < N; r++) at_wdata[8 * r +: 8] = 8'(tb[wo_buf][r] >> (8 * wo_cnt));
    end

    logic gather_last;
    assign gather_last = (gr == 3'(N - 1)) && (gk == 11'(KW) - 11'd1) && (!pk_mode || ph);
    always_comb KW = 8'(K >> 3);

    // ---------------------------------------------------------------- consumer (stream)
    typedef enum logic [3:0] {
        C_IDLE, C_WAIT_FULL, C_INJ_WAIT, C_STREAM, C_NEXT, C_CAP_WAIT, C_CAPTURE, C_DONE
    } cstate_e;
    cstate_e cstate;
    logic        sbuf;              // A_T half being streamed
    logic [13:0] c_p0;              // current unit's tile
    logic [5:0]  c_g;               // current job's output group
    logic        c_last_unit;
    logic        c_last_job;
    assign c_last_job = !(cached && !is_dw) || (c_g == I.out_g - 6'd1);

    // ---------------------------------------------------------------- weight loader (group mode)
    // Loads output groups' weights in order while the consumer streams. With K <= half the
    // buffer (wdbl), group g goes to half g%2 and group g+1 is prefetched during group g;
    // otherwise one group at a time into the whole buffer.
    typedef enum logic [1:0] {L_IDLE, L_REQ, L_LOAD} lstate_e;
    lstate_e lstate;
    logic       wdbl;
    logic [6:0] lg;                 // next group to load
    logic [6:0] loaded_cnt;         // groups whose weights have fully landed
    logic [6:0] cons_done;          // groups the consumer has finished
    logic       l_ok;               // the target half is free
    assign wdbl = (K <= 11'(WBUF_DEPTH / 2));
    assign l_ok = (lg < 7'(I.out_g)) && (lg < cons_done + (wdbl ? 7'd2 : 7'd1));

    logic [10:0] sk;
    logic        st_valid, st_first, st_zero;   // registered: aligned with buffer read data
    logic [2:0]  st_sel;
    logic [7:0]  since_inj;         // cycles since the last injection was issued (saturating)
    logic        inj_now;           // an injection (job's first value or capture wave) issues
    logic signed [DATA_W-1:0] arr_a [N];
    logic signed [DATA_W-1:0] arr_b [N];
    logic signed [ACC_W-1:0]  arr_shadow [N][N];

    always_comb begin
        for (int r = 0; r < N; r++) arr_a[r] = st_zero ? '0 : DATA_W'(at_rdata >> (8 * r));
        for (int c = 0; c < N; c++) begin
            if (st_zero)    arr_b[c] = '0;
            else if (is_dw) arr_b[c] = (3'(c) == st_sel) ? DATA_W'(wb_rdata >> (8 * c)) : '0;
            else            arr_b[c] = DATA_W'(wb_rdata >> (8 * c));
        end
    end

    npu_array u_array (.clk, .valid(st_valid), .first(st_first), .a(arr_a), .b(arr_b),
                       .shadow(arr_shadow));

    // ---------------------------------------------------------------- drain (concurrent)
    logic        pend_valid;        // the job streamed last, captured by the next injection
    logic [13:0] pend_p0;
    logic [5:0]  pend_g;
    logic        dr_busy, dr_run;
    logic [2:0]  dr_wait;
    logic [5:0]  dr_idx;            // column-pair index: row dr_idx[4:2], pair dr_idx[1:0]
    logic [13:0] dr_p0;
    logic [5:0]  dr_g;
    // per-channel params: prefetched for the pending job while it streams (nx_*), moved to
    // the drain's set (dp_*) at handover, so the drain never waits on the param buffer
    logic signed [31:0]        nx_bias [N], nx_mult [N], dp_bias [N], dp_mult [N];
    logic signed [SHIFT_W-1:0] nx_shift [N], dp_shift [N];
    logic [3:0]  pf_j;              // prefetch index (N = done)
    logic        pf_v;
    logic [2:0]  pf_jd;
    // two requant pipelines: even and odd column of each pair
    logic signed [ACC_W-1:0]   rq_acc   [2];
    logic signed [31:0]        rq_bias  [2], rq_mult [2];
    logic signed [SHIFT_W-1:0] rq_shift [2];
    logic signed [7:0]         rq_out   [2];
    logic [DQ-1:0]           tq_v;
    logic [1:0]              tq_cp  [DQ];
    logic                    tq_wr  [DQ];           // last pair of a valid pixel
    logic [ACT_AW-1:0]       tq_adr [DQ];
    logic [WORD_W-17:0]      row_word;              // bytes 0..N-3 of the pixel being drained

    for (genvar q = 0; q < 2; q++) begin : g_rq
        npu_requant u_rq (
            .clk, .acc(rq_acc[q]), .bias(rq_bias[q]), .mult(rq_mult[q]), .shift(rq_shift[q]),
            .rmode(I.rmode), .out_zp(I.out_zp), .act_min(I.act_min), .act_max(I.act_max),
            .out(rq_out[q]));
    end

    always_comb begin
        act_we    = tq_v[DQ-1] && tq_wr[DQ-1];
        act_waddr = tq_adr[DQ-1];
        act_wdata = {rq_out[1], rq_out[0], row_word};
    end

    // ---------------------------------------------------------------- buffer addressing
    always_comb begin
        if (!cached)    wb_raddr = wdbl ? {c_g[0], (WBUF_AW-1)'(sk)} : WBUF_AW'(sk);   // group mode
        else if (is_dw) wb_raddr = WBUF_AW'(24'(c_g) * 24'(taps) + 24'(sk >> 3));
        else            wb_raddr = WBUF_AW'(24'(c_g) * 24'(K) + 24'(sk));
        at_raddr = {sbuf, KA_AW'(sk)};
        pb_raddr = PBUF_AW'(16'(pend_g) * 16'(N) + 16'(pf_j[2:0]));               // param prefetch
        // external loads land in the buffers
        wb_we    = ext_rvalid && w_loading;
        wb_waddr = (!cached && wdbl) ? {lg[0], (WBUF_AW-1)'(ld_cnt)} : WBUF_AW'(ld_cnt);
        pb_we    = ext_rvalid && (state == S_PLOAD) && ld_cnt[0];
        pb_waddr = PBUF_AW'(ld_cnt >> 1);
        pb_wdata = {SHIFT_W'(ext_rdata), p_word0[63:32], p_word0[31:0]};
    end
    assign w_loading = (state == S_WLOAD) || (lstate == L_LOAD);
    assign inj_now   = (cstate == C_STREAM && sk == '0) || cstate == C_CAPTURE;

    // ---------------------------------------------------------------- sequential logic
    always_ff @(posedge clk) begin
        done      <= 1'b0;
        run_start <= 1'b0;
        d_v       <= 1'b0;
        st_valid  <= 1'b0;
        st_first  <= 1'b0;
        st_zero   <= 1'b0;
        if (since_inj != 8'hFF) since_inj <= since_inj + 8'd1;

        // ---- gather data landing: one row word per (k-word, row), then the transposer
        if (d_v && pk_mode && !d_ph) w0 <= act_rdata;
        if (d_v && (!pk_mode || d_ph)) begin
            tb[tb_sel][d_r] <= pk_mode ? pk_row : (d_pad ? {N{I.in_zp}} : act_rdata);
            if (d_r == 3'(N - 1)) begin
                tb_sel    <= !tb_sel;
                wo_active <= 1'b1;
                wo_buf    <= tb_sel;
                wo_cnt    <= '0;
                wo_kw     <= d_k;
            end
        end
        if (wo_active && !(d_v && (!pk_mode || d_ph) && d_r == 3'(N - 1))) begin
            wo_cnt <= wo_cnt + 3'd1;
            if (wo_cnt == 3'(N - 1)) wo_active <= 1'b0;
        end

        // ---- external load data
        if (ext_rvalid) begin
            ld_cnt <= ld_cnt + 17'd1;
            if (!ld_cnt[0]) p_word0 <= ext_rdata;
        end

        // ---- param prefetch for the pending job (N reads, one cycle latency)
        pf_v <= 1'b0;
        if (pf_j != 4'(N)) begin
            pf_v  <= 1'b1;
            pf_jd <= pf_j[2:0];
            pf_j  <= pf_j + 4'd1;
        end
        if (pf_v) begin
            nx_bias[pf_jd]  <= 32'(pb_rdata);
            nx_mult[pf_jd]  <= 32'(pb_rdata >> 32);
            nx_shift[pf_jd] <= SHIFT_W'(pb_rdata >> 64);
        end

        // ---- injection bookkeeping: hand the pending job to the drain
        if (inj_now) begin
            since_inj <= '0;
            if (pend_valid) begin
                dr_busy  <= 1'b1;
                dr_wait  <= 3'(DRAIN_DELAY - 1);
                dr_run   <= 1'b0;
                dr_p0    <= pend_p0;
                dr_g     <= pend_g;
                dp_bias  <= nx_bias;
                dp_mult  <= nx_mult;
                dp_shift <= nx_shift;
            end
            pend_valid <= (cstate == C_STREAM);
            pend_p0    <= c_p0;
            pend_g     <= c_g;
            if (cstate == C_STREAM) pf_j <= '0;    // start fetching this job's params
        end

        // ---- drain: wait DRAIN_DELAY, then issue N*N/2 column-pair reads (row-major)
        tq_v <= {tq_v[DQ-2:0], 1'b0};
        for (int i = DQ - 1; i > 0; i--) begin
            tq_cp[i]  <= tq_cp[i-1];
            tq_wr[i]  <= tq_wr[i-1];
            tq_adr[i] <= tq_adr[i-1];
        end
        if (dr_busy && !inj_now) begin
            if (!dr_run) begin
                if (dr_wait == 0) begin
                    dr_run <= 1'b1;
                    dr_idx <= '0;
                end else begin
                    dr_wait <= dr_wait - 3'd1;
                end
            end else begin
                logic [13:0] p;
                logic [2:0]  r, c0;
                r  = dr_idx[4:2];
                c0 = {dr_idx[1:0], 1'b0};
                p  = dr_p0 + 14'(r);
                for (int q = 0; q < 2; q++) begin
                    rq_acc[q]   <= arr_shadow[r][c0 + 3'(q)];
                    rq_bias[q]  <= dp_bias[c0 + 3'(q)];
                    rq_mult[q]  <= dp_mult[c0 + 3'(q)];
                    rq_shift[q] <= dp_shift[c0 + 3'(q)];
                end
                tq_v[0]   <= 1'b1;
                tq_cp[0]  <= dr_idx[1:0];
                tq_wr[0]  <= (dr_idx[1:0] == 2'(N / 2 - 1)) && (p < P);
                tq_adr[0] <= ACT_AW'(24'(I.out_base) + 24'(p) * 24'(I.out_g) + 24'(dr_g));
                dr_idx    <= dr_idx + 6'd1;
                if (dr_idx == 6'(N * N / 2 - 1)) begin
                    dr_busy <= 1'b0;
                    dr_run  <= 1'b0;
                end
            end
        end
        if (tq_v[DQ-1] && tq_cp[DQ-1] != 2'(N / 2 - 1))
            row_word[16 * tq_cp[DQ-1] +: 16] <= {rq_out[1], rq_out[0]};

        if (rst) begin
            state      <= S_IDLE;
            pstate     <= P_IDLE;
            cstate     <= C_IDLE;
            lstate     <= L_IDLE;
            wo_active  <= 1'b0;
            ext_req    <= 1'b0;
            dr_busy    <= 1'b0;
            dr_run     <= 1'b0;
            pend_valid <= 1'b0;
            tq_v       <= '0;
            since_inj  <= 8'hFF;
            pf_j       <= 4'(N);
            full       <= '0;
        end else begin
            // ======================================================== main
            unique case (state)
                S_IDLE: if (start) begin
                    I <= ins;
                    state <= S_INIT;
                end

                S_INIT: begin
                    is_dw   <= (I.op == OP_DWCONV);
                    pk_mode <= (I.op == OP_CONV) && I.in_src;
                    taps    <= 6'(I.ksize) * 6'(I.ksize);
                    if (I.op == OP_DWCONV) begin
                        K <= 11'(I.ksize) * 11'(I.ksize) * 11'(N);
                        layer_words <= 17'(I.out_g) * 17'(I.ksize) * 17'(I.ksize);
                    end else if (I.in_src) begin
                        // packed: K = ksize * KR, KR = ksize*C rounded up to a multiple of N
                        logic [8:0]  sc;
                        logic [3:0]  rw;
                        logic [10:0] kp;
                        sc = 9'(I.ksize) * 9'(I.in_cb);
                        rw = 4'((sc + 9'(N - 1)) >> 3);
                        kp = 11'(I.ksize) * 11'(rw) * 11'(N);
                        ksc <= sc;
                        krw <= rw;
                        K   <= kp;
                        layer_words <= 17'(I.out_g) * 17'(kp);
                    end else begin
                        K <= 11'(I.ksize) * 11'(I.ksize) * 11'(I.in_cb) * 11'(N);
                        layer_words <= 17'(I.out_g) * 17'(I.ksize) * 17'(I.ksize) * 17'(I.in_cb) * 17'(N);
                    end
                    P          <= 14'(I.out_h) * 14'(I.out_w);
                    pend_valid <= 1'b0;
                    full       <= '0;
                    state      <= S_PLOAD_REQ;
                end

                S_PLOAD_REQ: begin
                    cached   <= (layer_words <= 17'(WBUF_DEPTH));
                    ext_req  <= 1'b1;
                    ext_addr <= EXT_AW'(I.p_base);
                    ext_len  <= 16'(I.out_g) * 16'(N * PARAM_WORDS);
                    ld_len   <= 17'(I.out_g) * 17'(N * PARAM_WORDS);
                    ld_cnt   <= '0;
                    if (ext_req && ext_ack) begin
                        ext_req <= 1'b0;
                        state   <= S_PLOAD;
                    end
                end

                S_PLOAD: if (ld_cnt == ld_len) begin
                    if (cached) begin
                        state <= S_WLOAD_REQ;
                    end else begin
                        run_start <= 1'b1;      // group mode: the consumer loads weights per group
                        state     <= S_RUN;
                    end
                end

                S_WLOAD_REQ: begin
                    ext_req  <= 1'b1;
                    ext_addr <= EXT_AW'(I.w_base);
                    ext_len  <= 16'(layer_words);
                    ld_len   <= layer_words;
                    ld_cnt   <= '0;
                    if (ext_req && ext_ack) begin
                        ext_req <= 1'b0;
                        state   <= S_WLOAD;
                    end
                end

                S_WLOAD: if (ld_cnt == ld_len) begin
                    run_start <= 1'b1;
                    state     <= S_RUN;
                end

                S_RUN: if (cstate == C_DONE) state <= S_FINISH;

                S_FINISH: if (!dr_busy && !(|tq_v)) state <= S_DONE;

                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase

            // ======================================================== producer
            unique case (pstate)
                P_IDLE: if (run_start) begin
                    gbuf   <= 1'b0;
                    p_p0   <= '0;
                    p_oy   <= '0;
                    p_ox   <= '0;
                    p_g    <= '0;
                    p_new_tile <= 1'b1;
                    pstate <= P_WAIT_BUF;
                end

                P_WAIT_BUF: if (!full[gbuf]) begin
                    if (p_new_tile) begin
                        pstate <= P_SETUP;
                        sr <= '0;
                        cy <= p_oy;
                        cx <= p_ox;
                    end else begin
                        pstate <= P_GATHER;
                        gr <= '0; gk <= '0; ky <= '0; kx <= '0; kc <= '0; kj <= '0; ph <= 1'b0;
                    end
                end

                P_SETUP: begin
                    row_py[sr[2:0]]    <= cy;
                    row_px[sr[2:0]]    <= cx;
                    row_valid[sr[2:0]] <= (p_p0 + 14'(sr)) < P;
                    cy <= step_y(cy, cx, I.out_w);
                    cx <= step_x(cx, I.out_w);
                    if (sr == 4'(N - 1)) begin
                        p_noy  <= step_y(cy, cx, I.out_w);
                        p_nox  <= step_x(cx, I.out_w);
                        pstate <= P_GATHER;
                        gr <= '0; gk <= '0; ky <= '0; kx <= '0; kc <= '0; kj <= '0; ph <= 1'b0;
                    end
                    sr <= sr + 4'd1;
                end

                P_GATHER: begin
                    d_v     <= 1'b1;
                    d_r     <= gr;
                    d_k     <= 8'(gk);
                    d_pad   <= !inb;
                    d_ph    <= ph;
                    d_sh    <= win_addr[2:0];
                    d_kj    <= kj;
                    d_rowok <= row_ok;
                    d_lo    <= lo_m;
                    d_hi    <= hi_m;
                    if (gather_last) begin
                        pstate <= P_TAIL;
                    end else if (pk_mode && !ph) begin
                        ph <= 1'b1;                     // second aligned word of this window
                    end else if (gr != 3'(N - 1)) begin
                        ph <= 1'b0;
                        gr <= gr + 3'd1;
                    end else if (pk_mode) begin
                        // next window: within the kernel row, then the next kernel row
                        ph <= 1'b0;
                        gr <= '0;
                        gk <= gk + 11'd1;
                        if (kj == krw - 4'd1) begin
                            kj <= '0;
                            ky <= ky + 4'd1;
                        end else begin
                            kj <= kj + 4'd1;
                        end
                    end else begin
                        gr <= '0;
                        gk <= gk + 11'd1;
                        if (is_dw || kc == I.in_cb - 6'd1) begin
                            kc <= '0;
                            if (kx == I.ksize - 3'd1) begin
                                kx <= '0;
                                if (ky != 4'hF) ky <= ky + 4'd1;
                            end else begin
                                kx <= kx + 3'd1;
                            end
                        end else begin
                            kc <= kc + 6'd1;
                        end
                    end
                end

                // last data landed and the transposer has written it: publish the unit
                P_TAIL: if (!d_v && !wo_active) begin
                    logic last_unit;
                    full[gbuf]      <= 1'b1;
                    unit_p0[gbuf]   <= p_p0;
                    unit_g[gbuf]    <= p_g;
                    gbuf            <= !gbuf;
                    // advance: DWCONV channel blocks inside a tile; group mode tiles inside
                    // a group; cached CONV tiles only
                    if (is_dw) begin
                        last_unit = p_last_g && p_last_tile;
                        if (!p_last_g) begin
                            p_g <= p_g + 6'd1;
                            p_new_tile <= 1'b0;
                        end else begin
                            p_g <= '0;
                            p_p0 <= p_p0 + 14'(N);
                            p_oy <= p_noy;
                            p_ox <= p_nox;
                            p_new_tile <= 1'b1;
                        end
                    end else if (!cached) begin
                        last_unit = p_last_g && p_last_tile;
                        p_new_tile <= 1'b1;
                        if (!p_last_tile) begin
                            p_p0 <= p_p0 + 14'(N);
                            p_oy <= p_noy;
                            p_ox <= p_nox;
                        end else begin
                            p_g  <= p_g + 6'd1;
                            p_p0 <= '0;
                            p_oy <= '0;
                            p_ox <= '0;
                        end
                    end else begin
                        last_unit = p_last_tile;
                        p_p0 <= p_p0 + 14'(N);
                        p_oy <= p_noy;
                        p_ox <= p_nox;
                        p_new_tile <= 1'b1;
                    end
                    unit_last[gbuf] <= last_unit;
                    pstate <= last_unit ? P_IDLE : P_WAIT_BUF;
                end

                default: pstate <= P_IDLE;
            endcase

            // ======================================================== consumer
            unique case (cstate)
                C_IDLE: if (run_start) begin
                    sbuf      <= 1'b0;
                    cons_done <= '0;
                    cstate    <= C_WAIT_FULL;
                end

                // a gathered unit, and (group mode) its group's weights
                C_WAIT_FULL: if (full[sbuf] && (cached || loaded_cnt > 7'(unit_g[sbuf]))) begin
                    c_p0        <= unit_p0[sbuf];
                    c_g         <= (cached && !is_dw) ? 6'd0 : unit_g[sbuf];
                    c_last_unit <= unit_last[sbuf];
                    cstate      <= C_INJ_WAIT;
                end

                C_INJ_WAIT: if (since_inj >= 8'(PERIOD_MIN - 1)) begin
                    sk     <= '0;
                    cstate <= C_STREAM;
                end

                C_STREAM: begin
                    st_valid <= 1'b1;
                    st_first <= (sk == '0);
                    st_sel   <= sk[2:0];
                    if (sk == K - 11'd1) cstate <= C_NEXT;
                    else                 sk <= sk + 11'd1;
                end

                C_NEXT: begin
                    if (!c_last_job) begin
                        c_g    <= c_g + 6'd1;            // next group of the same tile
                        cstate <= C_INJ_WAIT;
                    end else begin
                        full[sbuf] <= 1'b0;              // A_T half free for the producer
                        sbuf       <= !sbuf;
                        cstate     <= c_last_unit ? C_CAP_WAIT : C_WAIT_FULL;
                        // group mode: the group's last tile frees its weight-buffer half
                        if (!cached && (c_p0 + 14'(N)) >= P) cons_done <= cons_done + 7'd1;
                    end
                end

                // after the instruction's last job: zero-valued capture wave
                C_CAP_WAIT: if (since_inj >= 8'(PERIOD_MIN - 1)) cstate <= C_CAPTURE;

                C_CAPTURE: begin
                    st_valid <= 1'b1;
                    st_first <= 1'b1;
                    st_zero  <= 1'b1;
                    cstate   <= C_DONE;
                end

                C_DONE: if (state == S_FINISH) cstate <= C_IDLE;

                default: cstate <= C_IDLE;
            endcase

            // ======================================================== weight loader
            unique case (lstate)
                L_IDLE: begin
                    if (run_start) begin
                        lg         <= '0;
                        loaded_cnt <= '0;
                    end else if (state == S_RUN && !cached && l_ok) begin
                        lstate <= L_REQ;
                    end
                end

                L_REQ: begin
                    ext_req  <= 1'b1;
                    ext_addr <= EXT_AW'(24'(I.w_base) + 24'(lg) * 24'(K));
                    ext_len  <= 16'(K);
                    ld_len   <= 17'(K);
                    ld_cnt   <= '0;
                    if (ext_req && ext_ack) begin
                        ext_req <= 1'b0;
                        lstate  <= L_LOAD;
                    end
                end

                L_LOAD: if (ld_cnt == ld_len) begin
                    loaded_cnt <= lg + 7'd1;
                    lg         <= lg + 7'd1;
                    lstate     <= L_IDLE;
                end

                default: lstate <= L_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
