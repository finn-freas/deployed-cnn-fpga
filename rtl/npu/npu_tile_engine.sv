// npu_tile_engine: executes CONV and DWCONV instructions (v1 micro-architecture).
//
// Per tile of N output pixels (raster order):
//   GATHER  im2col rows for the N pixels into N A-banks (one 64-bit word = 8 k values).
//           Word mode reads channel-blocked activation words; packed mode (CONV in_src=1)
//           assembles words byte by byte from the packed uint8 input buffer.
//   per output group g (N output channels):
//     PARAMS  load the group's N (bias, mult, shift) entries
//     CLEAR / STREAM  feed k = 0..K-1 into the systolic array
//     FLUSH   wait for the array pipeline
//     DRAIN   N*N accumulators -> requant -> one activation word per pixel
// CONV gathers once per tile and reuses A for every group. DWCONV runs as a CONV over a
// single channel block with block-diagonal weights (expanded on the fly from the compact
// depthwise weights), so it re-gathers for each channel block. Everything is sequential;
// overlapping the phases is a later optimization.

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
    // packed input buffer
    output      logic [IN_AW-1:0]    in_raddr,
    input  wire logic [7:0]          in_rdata,
    // weight / param stores
    output      logic [W_AW-1:0]     w_raddr,
    input  wire logic [WORD_W-1:0]   w_rdata,
    output      logic [P_AW-1:0]     p_raddr,
    input  wire logic [PARAM_W-1:0]  p_rdata
);

    localparam int KW_AW   = $clog2(K_MAX / N);          // A-bank word address
    localparam int FLUSH_N = 3 * (N - 1) + 3;            // array latency + margin
    localparam int RQ_LAT  = 4;                          // npu_requant latency

    typedef enum logic [3:0] {
        S_IDLE, S_INIT, S_GATHER, S_GATHER_TAIL, S_PARAMS, S_CLEAR, S_STREAM, S_FLUSH,
        S_DRAIN, S_DRAIN_TAIL, S_NEXT_GROUP, S_NEXT_TILE, S_DONE
    } state_e;
    state_e state;

    /* verilator lint_off UNUSEDSIGNAL */
    instr_t I;                      // each unit uses a subset of the fields
    /* verilator lint_on UNUSEDSIGNAL */
    logic        is_dw, pk_mode;
    logic [5:0]  taps;              // ksize^2
    logic [10:0] K;                 // reduction length
    logic [7:0]  KW;                // A words per pixel (K / 8)
    logic [13:0] P;                 // output pixels

    // tile / group
    logic [13:0] tile_p0;
    logic [6:0]  tile_oy, tile_ox, next_oy, next_ox;
    logic [5:0]  g;

    // ---------------------------------------------------------------- gather counters
    logic [2:0]  gr;                // row (pixel within tile)
    logic [6:0]  py, px;            // output coords of pixel gr
    logic [10:0] gk;                // word index (word mode) or byte index (packed mode)
    logic [3:0]  ky;                // one bit wider than ksize: packed mode counts past the kernel
    logic [2:0]  kx;
    logic [5:0]  kc;                // channel block (word mode) or channel (packed mode)
    logic        last_issue;

    function automatic logic [6:0] step_y(input logic [6:0] y, input logic [6:0] x, input logic [6:0] w);
        return (x == w - 7'd1) ? y + 7'd1 : y;
    endfunction
    function automatic logic [6:0] step_x(input logic [6:0] x, input logic [6:0] w);
        return (x == w - 7'd1) ? 7'd0 : x + 7'd1;
    endfunction

    // input coordinate of the current tap, and its bounds check
    logic signed [9:0] iy, ix;
    logic              pix_valid, real_tap, inb;
    always_comb begin
        iy = 10'(signed'({3'b0, py}) * signed'({8'b0, I.stride})) + 10'(ky) - 10'(I.pad);
        ix = 10'(signed'({3'b0, px}) * signed'({8'b0, I.stride})) + 10'(kx) - 10'(I.pad);
        pix_valid = (tile_p0 + 14'(gr)) < P;
        real_tap  = ky < 4'(I.ksize);                   // packed mode runs past the real taps
        inb = pix_valid && real_tap && iy >= 0 && iy < 10'(I.in_h) && ix >= 0 && ix < 10'(I.in_w);
    end

    // read addresses (combinational; the RAMs register them)
    logic [23:0] pix_lin;
    always_comb begin
        pix_lin   = 24'(unsigned'(iy)) * 24'(I.in_w) + 24'(unsigned'(ix));
        act_raddr = ACT_AW'(24'(I.in_base) + pix_lin * 24'(I.in_cb) + 24'(is_dw ? g : kc));
        in_raddr  = IN_AW'(24'(I.in_base) + pix_lin * 24'(I.in_cb) + 24'(kc));
    end

    // gather write pipeline (data arrives one cycle after the address)
    logic        gw_valid, gw_pad, gw_real;
    logic [2:0]  gw_bank;
    logic [10:0] gw_k;
    logic [WORD_W-1:0] asm_word;    // packed-mode word under assembly

    // ---------------------------------------------------------------- A banks
    logic [N-1:0]       ab_we;
    logic [KW_AW-1:0]   ab_waddr, ab_raddr;
    logic [WORD_W-1:0]  ab_wdata;
    logic [WORD_W-1:0]  ab_rdata [N];

    for (genvar r = 0; r < N; r++) begin : g_abank
        sdpram #(.W(WORD_W), .D(K_MAX / N)) u_bank (
            .clk, .we(ab_we[r]), .waddr(ab_waddr), .wdata(ab_wdata),
            .raddr(ab_raddr), .rdata(ab_rdata[r]));
    end

    always_comb begin
        logic [WORD_W-1:0] w;
        logic [7:0] byte_v;
        w        = asm_word;
        byte_v   = '0;
        ab_we    = '0;
        ab_waddr = KW_AW'(pk_mode ? (gw_k >> 3) : gw_k);
        ab_wdata = '0;
        if (gw_valid) begin
            if (!pk_mode) begin
                ab_wdata = gw_pad ? {N{I.in_zp}} : act_rdata;
                ab_we[gw_bank] = 1'b1;
            end else begin
                // uint8 -> int8 by adding in_zp; out-of-bounds taps read as u8 0; taps past
                // the real kernel are zero (their weights are zero too).
                byte_v = !gw_real ? 8'd0 : (gw_pad ? I.in_zp : in_rdata + I.in_zp);
                w[8 * gw_k[2:0] +: 8] = byte_v;
                ab_wdata = w;
                ab_we[gw_bank] = (gw_k[2:0] == 3'd7);
            end
        end
    end

    // ---------------------------------------------------------------- params
    logic signed [31:0]        prm_bias  [N];
    logic signed [31:0]        prm_mult  [N];
    logic signed [SHIFT_W-1:0] prm_shift [N];
    logic [3:0] pj;                 // param load index (0..N)
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
        for (int r = 0; r < N; r++) arr_a[r] = DATA_W'(ab_rdata[r] >> (8 * st_sel));
        for (int c = 0; c < N; c++) begin
            if (is_dw) arr_b[c] = (3'(c) == st_sel) ? DATA_W'(w_rdata >> (8 * c)) : '0;
            else       arr_b[c] = DATA_W'(w_rdata >> (8 * c));
        end
    end

    npu_array u_array (.clk, .clear(arr_clear), .valid(st_valid), .a(arr_a), .b(arr_b), .acc(arr_acc));

    // ---------------------------------------------------------------- drain / requant
    logic [6:0]  didx;              // 0..N*N
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

    // ---------------------------------------------------------------- weight / param addresses
    always_comb begin
        if (is_dw) w_raddr = W_AW'(24'(I.w_base) + 24'(g) * 24'(taps) + 24'(sk >> 3));
        else       w_raddr = W_AW'(24'(I.w_base) + 24'(g) * 24'(K) + 24'(sk));
        p_raddr  = P_AW'(16'(I.p_base) + 16'(g) * 16'(N) + 16'(pj[2:0]));
        ab_raddr = KW_AW'(sk >> 3);
    end

    // ---------------------------------------------------------------- control
    assign last_issue = (gr == 3'(N - 1)) && (gk == (pk_mode ? K - 11'd1 : 11'(KW) - 11'd1));

    always_ff @(posedge clk) begin
        done      <= 1'b0;
        arr_clear <= 1'b0;
        gw_valid  <= 1'b0;
        pj_valid  <= 1'b0;
        st_valid  <= 1'b0;
        tag_v     <= {tag_v[RQ_LAT-1:0], 1'b0};
        for (int i = RQ_LAT; i > 0; i--) tag_i[i] <= tag_i[i-1];

        // land gathered packed bytes into the word under assembly
        if (gw_valid && pk_mode) asm_word <= ab_wdata;
        // land loaded params
        if (pj_valid) begin
            prm_bias[pj_d]  <= 32'(p_rdata);
            prm_mult[pj_d]  <= 32'(p_rdata >> 32);
            prm_shift[pj_d] <= SHIFT_W'(p_rdata >> 64);
        end
        // collect requantized bytes of the current output pixel
        if (tag_v[RQ_LAT] && d_ti[2:0] != 3'(N - 1)) row_word[8 * d_ti[2:0] +: 8] <= rq_out;

        if (rst) begin
            state <= S_IDLE;
            tag_v <= '0;
        end else begin
            unique case (state)
                S_IDLE: if (start) begin
                    I <= ins;
                    state <= S_INIT;
                end

                S_INIT: begin
                    is_dw  <= (I.op == OP_DWCONV);
                    pk_mode <= (I.op == OP_CONV) && I.in_src;
                    taps   <= 6'(I.ksize) * 6'(I.ksize);
                    if (I.op == OP_DWCONV)
                        K <= 11'(I.ksize) * 11'(I.ksize) * 11'(N);
                    else if (I.in_src)
                        K <= ((11'(I.ksize) * 11'(I.ksize) * 11'(I.in_cb) + 11'(N - 1)) >> 3) << 3;
                    else
                        K <= 11'(I.ksize) * 11'(I.ksize) * 11'(I.in_cb) * 11'(N);
                    P       <= 14'(I.out_h) * 14'(I.out_w);
                    tile_p0 <= '0;
                    tile_oy <= '0;
                    tile_ox <= '0;
                    g       <= '0;
                    state   <= S_GATHER;
                    gr <= '0; py <= '0; px <= '0; gk <= '0; ky <= '0; kx <= '0; kc <= '0;
                end

                // ---------------- gather: one read per cycle
                S_GATHER: begin
                    gw_valid <= 1'b1;
                    gw_bank  <= gr;
                    gw_k     <= gk;
                    gw_pad   <= !inb;
                    gw_real  <= real_tap;
                    if (last_issue) begin
                        state   <= S_GATHER_TAIL;
                        next_oy <= step_y(py, px, I.out_w);
                        next_ox <= step_x(px, I.out_w);
                    end else if (gk == (pk_mode ? K - 11'd1 : 11'(KW) - 11'd1)) begin
                        // next pixel
                        gr <= gr + 3'd1;
                        gk <= '0; ky <= '0; kx <= '0; kc <= '0;
                        py <= step_y(py, px, I.out_w);
                        px <= step_x(px, I.out_w);
                    end else begin
                        gk <= gk + 11'd1;
                        // advance taps: innermost channel (block), then kx, then ky
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

                S_GATHER_TAIL: begin
                    pj    <= '0;
                    state <= S_PARAMS;
                end

                // ---------------- params: N reads
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

                // ---------------- stream k into the array
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

                // ---------------- drain N*N accumulators through requant
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
                    if (fcnt == 6'(RQ_LAT + 2)) state <= S_NEXT_GROUP;
                end

                S_NEXT_GROUP: begin
                    if (g == I.out_g - 6'd1) begin
                        state <= S_NEXT_TILE;
                    end else begin
                        g <= g + 6'd1;
                        if (is_dw) begin
                            // depthwise: A depends on the channel block, so gather again
                            state <= S_GATHER;
                            gr <= '0; gk <= '0; ky <= '0; kx <= '0; kc <= '0;
                            py <= tile_oy; px <= tile_ox;
                        end else begin
                            pj    <= '0;
                            state <= S_PARAMS;
                        end
                    end
                end

                S_NEXT_TILE: begin
                    if (tile_p0 + 14'(N) >= P) begin
                        state <= S_DONE;
                    end else begin
                        tile_p0 <= tile_p0 + 14'(N);
                        tile_oy <= next_oy;
                        tile_ox <= next_ox;
                        g       <= '0;
                        state   <= S_GATHER;
                        gr <= '0; gk <= '0; ky <= '0; kx <= '0; kc <= '0;
                        py <= next_oy; px <= next_ox;
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

    always_comb KW = 8'(K >> 3);

endmodule

`default_nettype wire
