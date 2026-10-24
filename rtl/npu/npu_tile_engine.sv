// npu_tile_engine: executes CONV and DWCONV instructions.
//
// Memory system (ISA v3):
//   * params for the whole layer are burst-read from the external store into a param
//     buffer at the start of the instruction;
//   * weights are burst-read into an on-chip weight buffer: the whole layer if it fits
//     ("cached" mode, tile-outer loop), otherwise one output group at a time ("group"
//     mode, group-outer loop, inputs re-gathered per group);
//   * im2col rows for a tile of N output pixels are gathered into one transposed buffer
//     A_T, where word k holds byte k of all N rows, so streaming reads one word per k.
//     Word-mode gathers (channel-blocked input) pass through a double-buffered N x N
//     byte transposer at one word per cycle; packed-mode gathers (uint8 HWC input in
//     activation memory) assemble each A_T word from N bytes, one per cycle.
//
// Compute (O3, overlapped drain): a "job" is one output group of one tile, streamed as
// K values with the first one flagged. The flagged value moves every PE's finished sum
// into its shadow register as it passes, so jobs stream back to back while a separate
// drain unit reads the previous job's shadows through requant into activation memory.
// When the next thing to do is a gather (new tile, or a depthwise channel block), a
// one-cycle zero-valued capture wave is injected instead, so the drain overlaps the
// gather. Injections are spaced at least PERIOD_MIN cycles apart so a shadow is never
// overwritten before the drain has read it. DWCONV runs as a CONV over one channel
// block with block-diagonal weights expanded on the fly from the compact depthwise
// weights, re-gathering per channel block.

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

    localparam int KA_AW       = $clog2(K_MAX);      // A_T address
    localparam int RQ_LAT      = 4;                  // npu_requant latency
    // Timing (injection issued in cycle i reaches PE(r,c) in cycle i+1+r+c):
    //   the drain's first read is in cycle i+DRAIN_DELAY+1 and it reads PE(r,c) N*r+c cycles
    //   later, which is after that PE captured as long as DRAIN_DELAY >= 1;
    //   a later injection in cycle i2 overwrites PE(r,c)'s shadow in cycle i2+1+r+c, so it
    //   must wait until i2 > i+DRAIN_DELAY+(N-1)*r for every r; and since that injection
    //   also hands the next job to the drain, this drain's N*N reads (cycles i+DRAIN_DELAY+1
    //   .. i+DRAIN_DELAY+N*N) must be finished: i2 >= i + DRAIN_DELAY + N*N + 1.
    localparam int DRAIN_DELAY = 3;
    localparam int PERIOD_MIN  = DRAIN_DELAY + N * N + 1;
    localparam int DQ          = RQ_LAT + 1;         // drain issue -> requant output

    typedef enum logic [4:0] {
        S_IDLE, S_INIT, S_PLOAD_REQ, S_PLOAD, S_WLOAD_REQ, S_WLOAD, S_TILE_SETUP,
        S_GATHER, S_GATHER_TAIL, S_INJ_WAIT, S_STREAM, S_NEXT, S_CAP_WAIT, S_CAPTURE,
        S_AFTER_CAP, S_FINISH, S_DONE
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

    // tile / group
    logic [13:0] tile_p0;
    logic [6:0]  tile_oy, tile_ox, next_oy, next_ox;
    logic [5:0]  g;
    logic        last_tile, last_group;
    assign last_tile  = (tile_p0 + 14'(N)) >= P;
    assign last_group = (g == I.out_g - 6'd1);

    // ---------------------------------------------------------------- external loads
    logic [16:0] ld_cnt, ld_len;
    logic [WORD_W-1:0] p_word0;

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
    logic [KA_AW-1:0]     at_waddr, at_raddr;
    logic [WORD_W-1:0]    at_wdata, at_rdata;
    sdpram #(.W(WORD_W), .D(K_MAX)) u_at (
        .clk, .we(at_we), .waddr(at_waddr), .wdata(at_wdata), .raddr(at_raddr), .rdata(at_rdata));

    // ---------------------------------------------------------------- per-row pixel coords
    logic [6:0]  row_py [N];
    logic [6:0]  row_px [N];
    logic [N-1:0] row_valid;
    logic [6:0]  cy, cx;            // coordinate walker (tile setup)
    logic [3:0]  sr;                // setup row counter

    function automatic logic [6:0] step_y(input logic [6:0] y, input logic [6:0] x, input logic [6:0] w);
        return (x == w - 7'd1) ? y + 7'd1 : y;
    endfunction
    function automatic logic [6:0] step_x(input logic [6:0] x, input logic [6:0] w);
        return (x == w - 7'd1) ? 7'd0 : x + 7'd1;
    endfunction

    // ---------------------------------------------------------------- gather
    logic [2:0]  gr;                // row being read
    logic [10:0] gk;                // k-word (word mode) or byte k (packed mode)
    logic [3:0]  ky;                // one bit wider than ksize: packed mode counts past the kernel
    logic [2:0]  kx;
    logic [5:0]  kc;                // channel block (word mode) or channel (packed mode)

    logic signed [9:0] iy, ix;
    logic              real_tap, inb;
    logic [23:0]       pix_lin;
    logic [26:0]       byte_addr;
    always_comb begin
        iy = 10'(signed'({3'b0, row_py[gr]}) * signed'({8'b0, I.stride})) + 10'(ky) - 10'(I.pad);
        ix = 10'(signed'({3'b0, row_px[gr]}) * signed'({8'b0, I.stride})) + 10'(kx) - 10'(I.pad);
        real_tap  = ky < 4'(I.ksize);
        inb = row_valid[gr] && real_tap && iy >= 0 && iy < 10'(I.in_h) && ix >= 0 && ix < 10'(I.in_w);
        pix_lin   = 24'(unsigned'(iy)) * 24'(I.in_w) + 24'(unsigned'(ix));
        byte_addr = 27'({I.in_base, 3'b000}) + 27'(pix_lin * 24'(I.in_cb)) + 27'(kc);
        if (pk_mode) act_raddr = ACT_AW'(byte_addr >> 3);
        else         act_raddr = ACT_AW'(24'(I.in_base) + pix_lin * 24'(I.in_cb) + 24'(is_dw ? g : kc));
    end

    // read-data pipeline (act memory has 1-cycle latency)
    logic        d_v, d_pad, d_real;
    logic [2:0]  d_r, d_lane;
    logic [KA_AW-1:0] d_k;

    // word mode: double-buffered transposer. Packed array on purpose: it is read N ways at
    // once (one column of bytes per cycle), so as an unpacked array synthesis would infer a
    // multi-port LUT RAM and replicate it per read port; as a vector it is plain flip-flops.
    logic [1:0][N-1:0][WORD_W-1:0] tb;
    logic        tb_sel;            // buffer being filled
    logic        wo_active, wo_buf;
    logic [2:0]  wo_cnt;
    logic [7:0]  wo_kw;
    // packed mode: word under assembly (byte r = row r)
    logic [WORD_W-1:0] pk_word;

    always_comb begin
        logic [WORD_W-1:0] w;
        logic [7:0] b;
        at_we    = 1'b0;
        at_waddr = '0;
        at_wdata = '0;
        w = pk_word;
        b = '0;
        if (pk_mode) begin
            if (d_v) begin
                b = !d_real ? 8'd0 : (d_pad ? I.in_zp : 8'(act_rdata >> (8 * d_lane)) + I.in_zp);
                w[8 * d_r +: 8] = b;
                at_we    = (d_r == 3'(N - 1));
                at_waddr = d_k;
                at_wdata = w;
            end
        end else if (wo_active) begin
            at_we    = 1'b1;
            at_waddr = KA_AW'({wo_kw, wo_cnt});
            for (int r = 0; r < N; r++) at_wdata[8 * r +: 8] = 8'(tb[wo_buf][r] >> (8 * wo_cnt));
        end
    end

    // ---------------------------------------------------------------- injection / array
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
    // pending: the job streamed last, whose result the next injection captures
    logic        pend_valid;
    logic [13:0] pend_p0;
    logic [5:0]  pend_g;
    // the job being drained
    logic        dr_busy;
    logic [1:0]  dr_wait;
    logic        dr_run;
    logic [6:0]  dr_idx;
    logic [13:0] dr_p0;
    logic [5:0]  dr_g;
    // issue stage -> requant pipeline tags
    logic signed [ACC_W-1:0] dr_sh;                 // selected shadow, registered
    logic [DQ-1:0]           tq_v;
    logic [2:0]              tq_c   [DQ];
    logic                    tq_wr  [DQ];           // last byte of a valid pixel
    logic [ACT_AW-1:0]       tq_adr [DQ];
    logic signed [7:0]       rq_out;
    logic [WORD_W-9:0]       row_word;              // bytes 0..N-2 of the pixel being drained

    npu_requant u_rq (
        .clk, .acc(dr_sh), .bias(32'(pb_rdata)), .mult(32'(pb_rdata >> 32)),
        .shift(SHIFT_W'(pb_rdata >> 64)), .rmode(I.rmode),
        .out_zp(I.out_zp), .act_min(I.act_min), .act_max(I.act_max), .out(rq_out));

    always_comb begin
        act_we    = tq_v[DQ-1] && tq_wr[DQ-1];
        act_waddr = tq_adr[DQ-1];
        act_wdata = {rq_out, row_word};
    end

    // ---------------------------------------------------------------- buffer addressing
    always_comb begin
        if (!cached)    wb_raddr = WBUF_AW'(sk);                                  // group mode (CONV)
        else if (is_dw) wb_raddr = WBUF_AW'(24'(g) * 24'(taps) + 24'(sk >> 3));
        else            wb_raddr = WBUF_AW'(24'(g) * 24'(K) + 24'(sk));
        at_raddr = KA_AW'(sk);
        pb_raddr = PBUF_AW'(16'(dr_g) * 16'(N) + 16'(dr_idx[2:0]));               // drain
        // external loads land in the buffers
        wb_we    = ext_rvalid && (state == S_WLOAD);
        wb_waddr = WBUF_AW'(ld_cnt);
        pb_we    = ext_rvalid && (state == S_PLOAD) && ld_cnt[0];
        pb_waddr = PBUF_AW'(ld_cnt >> 1);
        pb_wdata = {SHIFT_W'(ext_rdata), p_word0[63:32], p_word0[31:0]};
    end

    // ---------------------------------------------------------------- control
    logic gather_last;
    assign gather_last = (gr == 3'(N - 1)) && (gk == (pk_mode ? K - 11'd1 : 11'(KW) - 11'd1));
    always_comb KW = 8'(K >> 3);
    assign inj_now = (state == S_STREAM && sk == '0) || state == S_CAPTURE;

    // what follows a capture wave
    typedef enum logic [1:0] {A_GATHER_SAME_TILE, A_NEXT_TILE, A_NEXT_GROUP_LOAD, A_FINISH} after_e;
    after_e after_cap;

    always_ff @(posedge clk) begin
        done     <= 1'b0;
        d_v      <= 1'b0;
        st_valid <= 1'b0;
        st_first <= 1'b0;
        st_zero  <= 1'b0;
        if (since_inj != 8'hFF) since_inj <= since_inj + 8'd1;

        // ---- gather data landing
        if (d_v) begin
            if (pk_mode) begin
                pk_word <= at_wdata;
            end else begin
                tb[tb_sel][d_r] <= d_pad ? {N{I.in_zp}} : act_rdata;
                if (d_r == 3'(N - 1)) begin
                    tb_sel    <= !tb_sel;
                    wo_active <= 1'b1;
                    wo_buf    <= tb_sel;
                    wo_cnt    <= '0;
                    wo_kw     <= 8'(d_k);
                end
            end
        end
        if (wo_active && !(d_v && !pk_mode && d_r == 3'(N - 1))) begin
            wo_cnt <= wo_cnt + 3'd1;
            if (wo_cnt == 3'(N - 1)) wo_active <= 1'b0;
        end

        // ---- external load data
        if (ext_rvalid) begin
            ld_cnt <= ld_cnt + 17'd1;
            if (!ld_cnt[0]) p_word0 <= ext_rdata;
        end

        // ---- injection bookkeeping: hand the pending job to the drain
        if (inj_now) begin
            since_inj <= '0;
            if (pend_valid) begin
                dr_busy <= 1'b1;
                dr_wait <= 2'(DRAIN_DELAY - 1);
                dr_run  <= 1'b0;
                dr_p0   <= pend_p0;
                dr_g    <= pend_g;
            end
            pend_valid <= (state == S_STREAM);
            pend_p0    <= tile_p0;
            pend_g     <= g;
        end

        // ---- drain: wait DRAIN_DELAY, then issue N*N shadow reads (row-major)
        tq_v <= {tq_v[DQ-2:0], 1'b0};
        for (int i = DQ - 1; i > 0; i--) begin
            tq_c[i]   <= tq_c[i-1];
            tq_wr[i]  <= tq_wr[i-1];
            tq_adr[i] <= tq_adr[i-1];
        end
        if (dr_busy && !inj_now) begin
            if (!dr_run) begin
                if (dr_wait == 0) begin
                    dr_run <= 1'b1;
                    dr_idx <= '0;
                end else begin
                    dr_wait <= dr_wait - 2'd1;
                end
            end else begin
                logic [13:0] p;
                p = dr_p0 + 14'(dr_idx[5:3]);
                dr_sh     <= arr_shadow[dr_idx[5:3]][dr_idx[2:0]];
                tq_v[0]   <= 1'b1;
                tq_c[0]   <= dr_idx[2:0];
                tq_wr[0]  <= (dr_idx[2:0] == 3'(N - 1)) && (p < P);
                tq_adr[0] <= ACT_AW'(24'(I.out_base) + 24'(p) * 24'(I.out_g) + 24'(dr_g));
                dr_idx    <= dr_idx + 7'd1;
                if (dr_idx == 7'(N * N - 1)) begin
                    dr_busy <= 1'b0;
                    dr_run  <= 1'b0;
                end
            end
        end
        if (tq_v[DQ-1] && tq_c[DQ-1] != 3'(N - 1)) row_word[8 * tq_c[DQ-1] +: 8] <= rq_out;

        if (rst) begin
            state      <= S_IDLE;
            wo_active  <= 1'b0;
            ext_req    <= 1'b0;
            dr_busy    <= 1'b0;
            dr_run     <= 1'b0;
            pend_valid <= 1'b0;
            tq_v       <= '0;
            since_inj  <= 8'hFF;
        end else begin
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
                        logic [10:0] kp;
                        kp = ((11'(I.ksize) * 11'(I.ksize) * 11'(I.in_cb) + 11'(N - 1)) >> 3) << 3;
                        K <= kp;
                        layer_words <= 17'(I.out_g) * 17'(kp);
                    end else begin
                        K <= 11'(I.ksize) * 11'(I.ksize) * 11'(I.in_cb) * 11'(N);
                        layer_words <= 17'(I.out_g) * 17'(I.ksize) * 17'(I.ksize) * 17'(I.in_cb) * 17'(N);
                    end
                    P          <= 14'(I.out_h) * 14'(I.out_w);
                    tile_p0    <= '0;
                    tile_oy    <= '0;
                    tile_ox    <= '0;
                    g          <= '0;
                    pend_valid <= 1'b0;
                    state      <= S_PLOAD_REQ;
                end

                // ---------------- load the layer's params
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

                S_PLOAD: if (ld_cnt == ld_len) state <= S_WLOAD_REQ;

                // ---------------- load weights: whole layer (cached) or group g
                S_WLOAD_REQ: begin
                    ext_req  <= 1'b1;
                    ext_addr <= cached ? EXT_AW'(I.w_base) : EXT_AW'(24'(I.w_base) + 24'(g) * 24'(K));
                    ext_len  <= cached ? 16'(layer_words) : 16'(K);
                    ld_len   <= cached ? layer_words : 17'(K);
                    ld_cnt   <= '0;
                    if (ext_req && ext_ack) begin
                        ext_req <= 1'b0;
                        state   <= S_WLOAD;
                    end
                end

                S_WLOAD: if (ld_cnt == ld_len) begin
                    state <= S_TILE_SETUP;
                    sr    <= '0;
                    cy    <= tile_oy;
                    cx    <= tile_ox;
                end

                // ---------------- per-tile row coordinates
                S_TILE_SETUP: begin
                    row_py[sr[2:0]]    <= cy;
                    row_px[sr[2:0]]    <= cx;
                    row_valid[sr[2:0]] <= (tile_p0 + 14'(sr)) < P;
                    cy <= step_y(cy, cx, I.out_w);
                    cx <= step_x(cx, I.out_w);
                    if (sr == 4'(N - 1)) begin
                        next_oy <= step_y(cy, cx, I.out_w);
                        next_ox <= step_x(cx, I.out_w);
                        state <= S_GATHER;
                        gr <= '0; gk <= '0; ky <= '0; kx <= '0; kc <= '0;
                    end
                    sr <= sr + 4'd1;
                end

                // ---------------- gather: one read per cycle, rows innermost
                S_GATHER: begin
                    d_v    <= 1'b1;
                    d_r    <= gr;
                    d_k    <= KA_AW'(gk);
                    d_pad  <= !inb;
                    d_real <= real_tap;
                    d_lane <= byte_addr[2:0];
                    if (gather_last) begin
                        state <= S_GATHER_TAIL;
                    end else if (gr != 3'(N - 1)) begin
                        gr <= gr + 3'd1;
                    end else begin
                        gr <= '0;
                        gk <= gk + 11'd1;
                        // next tap: innermost channel (block), then kx, then ky
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

                S_GATHER_TAIL: if (!d_v && !wo_active) state <= S_INJ_WAIT;

                // ---------------- stream one job; its first value is an injection
                S_INJ_WAIT: if (since_inj >= 8'(PERIOD_MIN - 1)) begin
                    sk    <= '0;
                    state <= S_STREAM;
                end

                S_STREAM: begin
                    st_valid <= 1'b1;
                    st_first <= (sk == '0);
                    st_sel   <= sk[2:0];
                    if (sk == K - 11'd1) state <= S_NEXT;
                    else                 sk <= sk + 11'd1;
                end

                S_NEXT: begin
                    if (cached && !is_dw && !last_group) begin
                        // next group of the same tile reuses A: its first value captures this job
                        g     <= g + 6'd1;
                        state <= S_INJ_WAIT;
                    end else begin
                        state <= S_CAP_WAIT;
                        if (is_dw && !last_group)  after_cap <= A_GATHER_SAME_TILE;
                        else if (!last_tile)       after_cap <= A_NEXT_TILE;
                        else if (!cached && !last_group) after_cap <= A_NEXT_GROUP_LOAD;
                        else                       after_cap <= A_FINISH;
                    end
                end

                // ---------------- zero-valued capture wave, so the drain overlaps what follows
                S_CAP_WAIT: if (since_inj >= 8'(PERIOD_MIN - 1)) state <= S_CAPTURE;

                S_CAPTURE: begin
                    st_valid <= 1'b1;
                    st_first <= 1'b1;
                    st_zero  <= 1'b1;
                    state    <= S_AFTER_CAP;
                end

                S_AFTER_CAP: begin
                    unique case (after_cap)
                        A_GATHER_SAME_TILE: begin
                            g     <= g + 6'd1;
                            state <= S_GATHER;
                            gr <= '0; gk <= '0; ky <= '0; kx <= '0; kc <= '0;
                        end
                        A_NEXT_TILE: begin
                            tile_p0 <= tile_p0 + 14'(N);
                            tile_oy <= next_oy;
                            tile_ox <= next_ox;
                            if (cached) g <= '0;
                            state <= S_TILE_SETUP;
                            sr    <= '0;
                            cy    <= next_oy;
                            cx    <= next_ox;
                        end
                        A_NEXT_GROUP_LOAD: begin
                            g       <= g + 6'd1;
                            tile_p0 <= '0;
                            tile_oy <= '0;
                            tile_ox <= '0;
                            state   <= S_WLOAD_REQ;
                        end
                        default: state <= S_FINISH;
                    endcase
                end

                S_FINISH: if (!dr_busy && !(|tq_v)) state <= S_DONE;

                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
