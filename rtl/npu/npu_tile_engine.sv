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
// Per output group: PARAMS -> CLEAR -> STREAM k=0..K-1 -> FLUSH -> DRAIN (requant).
// DWCONV runs as a CONV over one channel block with block-diagonal weights expanded on
// the fly from the compact depthwise weights, re-gathering per channel block.

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

    localparam int KA_AW   = $clog2(K_MAX);              // A_T address
    localparam int FLUSH_N = 3 * (N - 1) + 3;            // array latency + margin
    localparam int RQ_LAT  = 4;                          // npu_requant latency

    typedef enum logic [4:0] {
        S_IDLE, S_INIT, S_PLOAD_REQ, S_PLOAD, S_WLOAD_REQ, S_WLOAD, S_TILE_SETUP,
        S_GATHER, S_GATHER_TAIL, S_PARAMS, S_CLEAR, S_STREAM, S_FLUSH, S_DRAIN,
        S_DRAIN_TAIL, S_NEXT_GROUP, S_NEXT_TILE, S_DONE
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

    // word mode: double-buffered transposer
    logic [WORD_W-1:0] tb [2][N];
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
                at_waddr = KA_AW'(d_k);
                at_wdata = w;
            end
        end else if (wo_active) begin
            at_we    = 1'b1;
            at_waddr = KA_AW'({wo_kw, wo_cnt});
            for (int r = 0; r < N; r++) at_wdata[8 * r +: 8] = 8'(tb[wo_buf][r] >> (8 * wo_cnt));
        end
    end

    // ---------------------------------------------------------------- params
    logic signed [31:0]        prm_bias  [N];
    logic signed [31:0]        prm_mult  [N];
    logic signed [SHIFT_W-1:0] prm_shift [N];
    logic [3:0] pj;
    logic       pj_valid;
    logic [2:0] pj_d;

    // ---------------------------------------------------------------- stream / array
    logic [10:0] sk;
    logic        st_valid;
    logic [2:0]  st_sel;
    logic        arr_clear;
    logic signed [DATA_W-1:0] arr_a [N];
    logic signed [DATA_W-1:0] arr_b [N];
    logic signed [ACC_W-1:0]  arr_acc [N][N];
    logic [5:0]  fcnt;

    always_comb begin
        for (int r = 0; r < N; r++) arr_a[r] = DATA_W'(at_rdata >> (8 * r));
        for (int c = 0; c < N; c++) begin
            if (is_dw) arr_b[c] = (3'(c) == st_sel) ? DATA_W'(wb_rdata >> (8 * c)) : '0;
            else       arr_b[c] = DATA_W'(wb_rdata >> (8 * c));
        end
    end

    npu_array u_array (.clk, .clear(arr_clear), .valid(st_valid), .a(arr_a), .b(arr_b), .acc(arr_acc));

    // ---------------------------------------------------------------- drain / requant
    logic [6:0]  didx;
    logic signed [ACC_W-1:0]   rq_acc;
    logic signed [31:0]        rq_bias, rq_mult;
    logic signed [SHIFT_W-1:0] rq_shift;
    logic signed [7:0]         rq_out;
    logic [RQ_LAT:0] tag_v;
    logic [5:0]      tag_i [RQ_LAT+1];
    logic [WORD_W-9:0] row_word;    // bytes 0..N-2 of the pixel being drained
    logic [5:0] d_ti;

    npu_requant u_rq (
        .clk, .acc(rq_acc), .bias(rq_bias), .mult(rq_mult), .shift(rq_shift), .rmode(I.rmode),
        .out_zp(I.out_zp), .act_min(I.act_min), .act_max(I.act_max), .out(rq_out));

    assign d_ti = tag_i[RQ_LAT];

    always_comb begin
        logic [13:0] p;
        p = tile_p0 + 14'(d_ti[5:3]);
        act_we    = tag_v[RQ_LAT] && (d_ti[2:0] == 3'(N - 1)) && (p < P);
        act_waddr = ACT_AW'(24'(I.out_base) + 24'(p) * 24'(I.out_g) + 24'(g));
        act_wdata = {rq_out, row_word};
    end

    // ---------------------------------------------------------------- buffer addressing
    always_comb begin
        if (!cached)    wb_raddr = WBUF_AW'(sk);                                  // group mode (CONV)
        else if (is_dw) wb_raddr = WBUF_AW'(24'(g) * 24'(taps) + 24'(sk >> 3));
        else            wb_raddr = WBUF_AW'(24'(g) * 24'(K) + 24'(sk));
        at_raddr = KA_AW'(sk);
        pb_raddr = PBUF_AW'(16'(g) * 16'(N) + 16'(pj[2:0]));
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

    always_ff @(posedge clk) begin
        done      <= 1'b0;
        arr_clear <= 1'b0;
        d_v       <= 1'b0;
        pj_valid  <= 1'b0;
        st_valid  <= 1'b0;
        tag_v     <= {tag_v[RQ_LAT-1:0], 1'b0};
        for (int i = RQ_LAT; i > 0; i--) tag_i[i] <= tag_i[i-1];

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

        // ---- params landing
        if (pj_valid) begin
            prm_bias[pj_d]  <= 32'(pb_rdata);
            prm_mult[pj_d]  <= 32'(pb_rdata >> 32);
            prm_shift[pj_d] <= SHIFT_W'(pb_rdata >> 64);
        end
        if (tag_v[RQ_LAT] && d_ti[2:0] != 3'(N - 1)) row_word[8 * d_ti[2:0] +: 8] <= rq_out;

        if (rst) begin
            state     <= S_IDLE;
            tag_v     <= '0;
            wo_active <= 1'b0;
            ext_req   <= 1'b0;
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
                    P       <= 14'(I.out_h) * 14'(I.out_w);
                    tile_p0 <= '0;
                    tile_oy <= '0;
                    tile_ox <= '0;
                    g       <= '0;
                    state   <= S_PLOAD_REQ;
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

                S_GATHER_TAIL: if (!d_v && !wo_active) begin
                    pj    <= '0;
                    state <= S_PARAMS;
                end

                // ---------------- params of group g from the param buffer
                S_PARAMS: begin
                    if (pj == 4'(N)) begin
                        state <= S_CLEAR;
                    end else begin
                        pj_valid <= 1'b1;
                        pj_d     <= pj[2:0];
                        pj       <= pj + 4'd1;
                    end
                end

                S_CLEAR: begin
                    arr_clear <= 1'b1;
                    sk        <= '0;
                    state     <= S_STREAM;
                end

                S_STREAM: begin
                    st_valid <= 1'b1;
                    st_sel   <= sk[2:0];
                    if (sk == K - 11'd1) begin
                        fcnt  <= '0;
                        state <= S_FLUSH;
                    end else begin
                        sk <= sk + 11'd1;
                    end
                end

                S_FLUSH: begin
                    fcnt <= fcnt + 6'd1;
                    if (fcnt == 6'(FLUSH_N)) begin
                        didx  <= '0;
                        state <= S_DRAIN;
                    end
                end

                S_DRAIN: begin
                    rq_acc   <= arr_acc[didx[5:3]][didx[2:0]];
                    rq_bias  <= prm_bias[didx[2:0]];
                    rq_mult  <= prm_mult[didx[2:0]];
                    rq_shift <= prm_shift[didx[2:0]];
                    tag_v[0] <= 1'b1;
                    tag_i[0] <= didx[5:0];
                    if (didx == 7'(N * N - 1)) begin
                        fcnt  <= '0;
                        state <= S_DRAIN_TAIL;
                    end
                    didx <= didx + 7'd1;
                end

                S_DRAIN_TAIL: begin
                    fcnt <= fcnt + 6'd1;
                    if (fcnt == 6'(RQ_LAT + 2)) state <= cached ? S_NEXT_GROUP : S_NEXT_TILE;
                end

                // cached mode: groups inside the tile loop
                S_NEXT_GROUP: begin
                    if (g == I.out_g - 6'd1) begin
                        state <= S_NEXT_TILE;
                    end else begin
                        g <= g + 6'd1;
                        if (is_dw) begin
                            // depthwise: A depends on the channel block, so gather again
                            state <= S_GATHER;
                            gr <= '0; gk <= '0; ky <= '0; kx <= '0; kc <= '0;
                        end else begin
                            pj    <= '0;
                            state <= S_PARAMS;
                        end
                    end
                end

                S_NEXT_TILE: begin
                    if (tile_p0 + 14'(N) >= P) begin
                        if (cached || g == I.out_g - 6'd1) begin
                            state <= S_DONE;
                        end else begin
                            // group mode: next group, back to the first tile
                            g       <= g + 6'd1;
                            tile_p0 <= '0;
                            tile_oy <= '0;
                            tile_ox <= '0;
                            state   <= S_WLOAD_REQ;
                        end
                    end else begin
                        tile_p0 <= tile_p0 + 14'(N);
                        tile_oy <= next_oy;
                        tile_ox <= next_ox;
                        if (cached) g <= '0;
                        state <= S_TILE_SETUP;
                        sr    <= '0;
                        cy    <= next_oy;
                        cx    <= next_ox;
                    end
                end

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
