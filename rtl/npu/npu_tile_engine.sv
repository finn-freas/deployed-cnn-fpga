// npu_tile_engine: executes CONV and DWCONV instructions.
//
// Memory system (ISA v3):
//   * params are burst-read from the external store into a param buffer while the layer
//     runs, and each job (or depthwise unit) waits only for its own output group's: cached
//     and depthwise layers load them after the weights, group mode loads each group's after
//     its weights;
//   * weights are burst-read into an on-chip weight buffer: the whole layer if it fits
//     ("cached" mode, tile-outer loop), otherwise one output group at a time ("group"
//     mode, group-outer loop, inputs re-gathered per group);
//   * CONV: im2col rows for a tile of N output pixels are gathered into a transposed
//     buffer A_T, where word k holds byte k of all N rows, so streaming reads one word per
//     k. Word-mode gathers (channel-blocked input) pass through a double-buffered N x N
//     byte transposer at one word per cycle; packed-mode gathers (uint8 HWC input in
//     activation memory) assemble each A_T word from N bytes, one per cycle.
//
// CONV compute:
//   * O3, overlapped drain: a "job" is one output group of one tile, streamed as K values
//     with the first one flagged. The flagged value moves every PE's finished sum into its
//     shadow register as it passes, so jobs stream back to back while a separate drain
//     unit reads the previous job's shadows through requant into activation memory.
//     Injections are spaced at least PERIOD_MIN cycles apart so a shadow is never
//     overwritten before the drain has read it.
//   * Double-buffered A_T: a producer FSM gathers the next "unit" (a tile; in group mode a
//     tile of one group) into one half while a consumer FSM streams the jobs of the other
//     half. The next job's first value captures the previous one; a zero-valued capture
//     wave is only injected after the instruction's last job.
//
// DWCONV (the array in depthwise mode, PE(r,c) = output pixel r x channel lane c): a unit
// is N horizontally adjacent output pixels of one output row and one channel block. Its
// input rows stream through a shift register ("line") one word per cycle, SEG = 7*stride+3
// words per kernel row; when a kernel row has fully arrived the line holds the N pixels'
// windows at positions r*stride, and the next kernel-row's words shifting in move them
// one column per cycle, so the row's taps issue on consecutive cycles with no extra reads.
//
// Address arithmetic is incremental (no multipliers): the few products of instruction
// fields are formed once by shift-and-add at the start of the instruction.

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
    // Timing (a value issued in cycle i reaches row r of the array in cycle i+1+r): a job's
    //   last value, issued in cycle h, moves row r's sums into the shadows in cycle h+1+r;
    //   the drain (two requant pipelines, a column pair per cycle) takes the job in cycle h
    //   and reads row r in cycles h+DRAIN_DELAY+1+(N/2)*r .. +N/2-1, after that capture for
    //   DRAIN_DELAY >= 1. The next job's capture of row r (h2+1+r) must follow the read, and
    //   the drain must have finished its N*N/2 reads when it takes the next job:
    //   h2 >= h + DRAIN_DELAY + N*N/2 + 1 = h + PERIOD_MIN (which covers the overwrite).
    //   Jobs of an instruction have equal lengths, so injections (first values) are spaced
    //   the same way.
    localparam int DRAIN_DELAY    = 1;
    localparam int PERIOD_MIN     = DRAIN_DELAY + N * N / 2 + 1;
    localparam int PERIOD_DW      = PERIOD_MIN;
    localparam int DQ             = RQ_LAT + 1;      // drain issue -> requant output

    // ---------------------------------------------------------------- main FSM
    typedef enum logic [3:0] {
        S_IDLE, S_INIT, S_CONST, S_WLOAD_REQ, S_WLOAD, S_RUN, S_FINISH, S_DONE
    } state_e;
    state_e state;

    /* verilator lint_off UNUSEDSIGNAL */
    instr_t I;                      // the instruction, held by the sequencer while we run;
    assign I = ins;                 // each unit uses a subset of the fields
    /* verilator lint_on UNUSEDSIGNAL */
    logic        is_dw, pk_mode, cached;
    logic [5:0]  taps;              // ksize^2
    logic [10:0] K;                 // CONV reduction length
    logic [7:0]  KW;                // A words per pixel in word mode (K / N)
    logic [13:0] P;                 // output pixels
    logic [16:0] layer_words;       // weight words of the whole layer
    logic        run_start;         // pulse: producer/consumer or depthwise streamer begin
    // instruction-derived constants (in units of the input layout: words, or bytes if packed)
    logic [12:0] RS;                // input row stride: in_w * in_cb
    logic [12:0] OWG;               // output row stride: out_w * out_g
    logic [7:0]  SC;                // stride * in_cb
    logic [14:0] SRS;               // stride * RS
    logic [8:0]  PADC;              // pad * in_cb
    logic [15:0] PADR;              // pad * RS
    logic [8:0]  NOG;               // N * out_g: output words per tile

    // a * b for a small (3-bit) b, as shifts and adds
    function automatic logic [17:0] mul3(input logic [14:0] a, input logic [2:0] b);
        return (b[0] ? 18'(a) : 18'd0) + (b[1] ? 18'(a) << 1 : 18'd0) + (b[2] ? 18'(a) << 2 : 18'd0);
    endfunction

    // shift-and-add multipliers for the larger products (7 steps, in S_CONST)
    logic [3:0]  m_cnt;             // 0 idle, 1..7 multiply steps, 8 derived constants
    logic [16:0] m_a1;              // layer_words = out_g * (K or taps)
    logic [13:0] m_a2;              // P = out_h * out_w
    logic [12:0] m_a3, m_a4;        // RS = in_cb * in_w, OWG = out_g * out_w
    logic [6:0]  m_b1, m_b2, m_b3, m_b4;
    assign cached = (layer_words <= 17'(WBUF_DEPTH));

    // ---------------------------------------------------------------- external loads
    logic [16:0] ld_cnt, ld_len;
    logic [WORD_W-1:0] p_word0;
    logic        w_loading;         // a weight burst is landing in the weight buffer
    logic        p_loading;         // a param burst is landing in the param buffer
    // param loader for cached and depthwise layers (group mode: the weight loader)
    typedef enum logic [1:0] {PL_IDLE, PL_REQ, PL_LOAD, PL_DONE} plstate_e;
    plstate_e    plstate;

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
    logic [12:0] unit_ob [2];       // per half: output word address of that pixel
    logic [5:0]  unit_g  [2];       // per half: group (group mode)
    logic        unit_last [2];     // per half: last unit of the instruction

    // ---------------------------------------------------------------- producer (gather)
    typedef enum logic [1:0] {P_IDLE, P_WAIT_BUF, P_SETUP, P_GATHER} pstate_e;
    // A unit is published (its half marked full) when the transposer writes its last block;
    // meanwhile the producer already gathers the next unit. pub_* hold the published unit.
    logic        pub_half, pub_last;
    logic [13:0] pub_p0;
    logic [12:0] pub_ob;
    logic [5:0]  pub_g;
    pstate_e pstate;
    logic        gbuf;              // A_T half being filled
    logic [13:0] p_p0;              // unit's tile: first output pixel
    logic [12:0] p_ob;              // ... its output word address
    logic [5:0]  p_g;               // unit's group
    logic        p_last_tile, p_last_g, p_new_tile;
    assign p_last_tile = (p_p0 + 14'(N)) >= P;
    assign p_last_g    = (p_g == I.out_g - 6'd1);

    // Pixel walker: output column x and the input coordinates/offset of the window corner,
    // wy = y*stride - pad, wx = x*stride - pad, wyb = wy*RS, wxb = wx*in_cb. p_* hold the
    // tile's first pixel, p_n* the next tile's.
    typedef struct packed {
        logic [6:0]         x;
        logic signed [9:0]  wy, wx;
        logic signed [19:0] wyb, wxb;
    } walk_t;
    walk_t       pw, pnw, cw;       // tile start, next tile start, setup cursor
    walk_t       w_origin;          // pixel 0
    function automatic walk_t walk_step(input walk_t w, input logic [6:0] out_w,
                                        input logic [1:0] stride, input logic [2:0] pad,
                                        input logic [7:0] sc, input logic [14:0] srs,
                                        input logic [8:0] padc);
        walk_t n = w;
        if (w.x == out_w - 7'd1) begin
            n.x   = '0;
            n.wx  = -10'(pad);
            n.wxb = -20'(padc);
            n.wy  = w.wy + 10'(stride);
            n.wyb = w.wyb + 20'(srs);
        end else begin
            n.x   = w.x + 7'd1;
            n.wx  = w.wx + 10'(stride);
            n.wxb = w.wxb + 20'(sc);
        end
        return n;
    endfunction

    logic signed [9:0]  row_iy0 [N];    // per tile row: window corner
    logic signed [9:0]  row_ix0 [N];
    logic signed [19:0] row_base [N];   // wy*RS + wx*in_cb
    logic signed [19:0] row_cxb [N];    // wx*in_cb
    logic [N-1:0] row_valid;
    logic [3:0]  sr;                // setup row counter

    // Gather order: k-word outer, rows inner. Word mode reads one channel-blocked word per
    // (k-word, row). Packed mode (O2): each k-word is a contiguous 8-byte window of one input
    // row (the compiler pads each kernel row to KR), spanning two aligned words (phase 0/1).
    // Consecutive windows of a kernel row are 8 bytes apart, so a window's first word is the
    // previous window's second: only a kernel row's first window reads phase 0.
    logic [2:0]  gr;                // row being read
    logic [10:0] gk;                // k-word
    logic        ph;                // packed mode: which of the two aligned words
    logic [3:0]  ky;
    logic [2:0]  kx;                // word mode: kernel column
    logic [5:0]  kc;                // word mode: channel block
    logic [3:0]  kj;                // packed mode: window within the kernel row (0..KRW-1)
    logic [3:0]  krw;               // packed mode: KR / N
    logic [8:0]  ksc;               // packed mode: ksize * C (real bytes per kernel row)
    logic [15:0] koff, krow;        // window offset from the corner: ky*RS + (kx*in_cb + kc | kj*N)

    logic signed [9:0]  iy, ix;
    logic               inb, row_ok;
    logic signed [19:0] g_addr;     // word address (word mode) / byte address (packed)
    logic signed [15:0] lo_m, hi_m; // packed: window bytes m with lo_m <= m < hi_m are in bounds
    logic [3:0]  pk_s, pk_e, pk_p;  // packed: bytes b of this window with pk_s <= b < pk_e are
                                    // in bounds; bytes b >= pk_p are past the kernel row
    // clamp(v - m0, 0, N) for the window starting at byte m0 = kj*N
    function automatic logic [3:0] edge8(input logic signed [15:0] v, input logic [3:0] win);
        logic signed [15:0] d;
        d = v - 16'(signed'({1'b0, win, 3'b000}));
        return (d < 0) ? 4'd0 : (d > 16'sd8) ? 4'd8 : 4'(d);
    endfunction
    always_comb begin
        iy     = row_iy0[gr] + 10'(ky);
        ix     = row_ix0[gr] + 10'(kx);
        row_ok = row_valid[gr] && iy >= 0 && iy < 10'(I.in_h);
        inb    = row_ok && ix >= 0 && ix < 10'(I.in_w);
        g_addr = (pk_mode ? 20'({I.in_base, 3'b000}) : 20'(I.in_base)) + row_base[gr] + 20'(koff);
        lo_m   = (row_ix0[gr] < 0) ? -16'(row_cxb[gr]) : 16'sd0;
        hi_m   = 16'(signed'({3'b0, RS})) - 16'(row_cxb[gr]);
        pk_s   = edge8(lo_m, kj);
        pk_e   = edge8(hi_m, kj);
        pk_p   = edge8(16'(signed'({7'b0, ksc})), kj);
    end

    // read-data pipeline (act memory has 1-cycle latency)
    logic        d_v, d_pad, d_ph;
    logic [2:0]  d_r, d_sh;
    logic [7:0]  d_k;               // k-word
    logic        d_rowok;
    logic        d_half, d_lastu;    // A_T half of the unit; last read of the unit
    logic [3:0]  d_s, d_e, d_p;
    // packed: the last word read for each tile row (a window's first word: its own phase 0,
    // or the previous window's phase 1)
    logic [WORD_W-1:0] w_hist [N];

    // double-buffered N x N byte transposer (both modes): rows are written one per cycle,
    // columns read one per cycle. Byte (row r, column c) lives in byte RAM (r + c) mod N at
    // {buffer, r}, so a row write and a column read each touch every RAM exactly once
    // (rotated by r, resp. c): eight small LUT RAMs instead of 2*N*N bytes of registers.
    logic        tb_we;
    logic [WORD_W-1:0] tb_wdata;    // row d_r of the block being filled
    logic [7:0]  tb_q [N];          // RAM k's byte of the column being written out
    logic        tb_sel;            // buffer being filled
    logic        wo_active, wo_buf;
    logic        wo_half, wo_lastu;  // A_T half of the block being written; last block of its unit
    logic [2:0]  wo_cnt;
    logic [7:0]  wo_kw;

    // packed mode: assemble a row's window from the two aligned words, applying padding
    logic [WORD_W-1:0] pk_row;
    always_comb begin
        logic [WORD_W-1:0]   win;
        win = WORD_W'({act_rdata, w_hist[d_r]} >> (8 * d_sh));
        for (int b = 0; b < N; b++) begin
            if (4'(b) >= d_p)                                pk_row[8 * b +: 8] = 8'd0;   // past the kernel row
            else if (!d_rowok || 4'(b) < d_s || 4'(b) >= d_e) pk_row[8 * b +: 8] = I.in_zp; // padding
            else                                              pk_row[8 * b +: 8] = 8'(win >> (8 * b)) + I.in_zp;
        end
    end

    always_comb begin
        tb_we    = d_v && (!pk_mode || d_ph);
        tb_wdata = pk_mode ? pk_row : (d_pad ? {N{I.in_zp}} : act_rdata);
    end
    for (genvar k = 0; k < N; k++) begin : g_tb
        logic [7:0] ram [2 * N];
        always_ff @(posedge clk)
            if (tb_we) ram[{tb_sel, d_r}] <= 8'(tb_wdata >> (8 * 3'(3'(k) - d_r)));
        assign tb_q[k] = ram[{wo_buf, 3'(3'(k) - wo_cnt)}];
    end

    always_comb begin
        at_we    = wo_active;
        at_waddr = {wo_half, KA_AW'({wo_kw, wo_cnt})};
        for (int r = 0; r < N; r++) at_wdata[8 * r +: 8] = tb_q[3'(3'(r) + wo_cnt)];
    end

    logic gather_last, krow_end;
    assign gather_last = (gr == 3'(N - 1)) && (gk == 11'(KW) - 11'd1) && (!pk_mode || ph);
    assign krow_end    = pk_mode ? (kj == krw - 4'd1) : (kc == I.in_cb - 6'd1 && kx == I.ksize - 3'd1);
    always_comb KW = 8'(K >> 3);

    // ---------------------------------------------------------------- consumer (stream)
    typedef enum logic [3:0] {
        C_IDLE, C_WAIT_FULL, C_INJ_WAIT, C_STREAM, C_NEXT, C_DONE
    } cstate_e;
    cstate_e cstate;
    logic        sbuf;              // A_T half being streamed
    logic [13:0] c_p0;              // current unit's tile
    logic [12:0] c_ob;              // ... its output word address
    logic [5:0]  c_g;               // current job's output group
    logic [9:0]  c_wb;              // cached mode: weight buffer address of group c_g
    logic        c_last_unit;
    logic        c_last_job;
    assign c_last_job = !cached || (c_g == I.out_g - 6'd1);
    // cached / depthwise: group g's params have landed (16 words per group of N entries)
    function automatic logic p_ready(input logic [5:0] g);
        return (plstate == PL_DONE) || (plstate == PL_LOAD && !I.uniform && ld_cnt > 17'({g, 4'hF}));
    endfunction

    // ---------------------------------------------------------------- weight loader (group mode)
    // Loads output groups' weights in order while the consumer streams. With K <= half the
    // buffer (wdbl), group g goes to half g%2 and group g+1 is prefetched during group g;
    // otherwise one group at a time into the whole buffer.
    typedef enum logic [2:0] {L_IDLE, L_REQ, L_LOAD, L_PREQ, L_PLOAD} lstate_e;
    lstate_e lstate;
    logic       wdbl;
    logic [6:0] lg;                 // next group to load
    logic [15:0] l_waddr;           // its external address
    logic [15:0] l_paddr;           // its params' external address
    logic [6:0] loaded_cnt;         // groups whose weights and params have fully landed
    logic [6:0] cons_done;          // groups the consumer has finished
    logic       l_ok;               // the target half is free
    assign wdbl = (K <= 11'(WBUF_DEPTH / 2));
    assign l_ok = (lg < 7'(I.out_g)) && (lg < cons_done + (wdbl ? 7'd2 : 7'd1));

    logic [10:0] sk;
    logic        st_valid, st_last;  // registered: aligned with buffer read data
    logic        c_set;             // param set of the consumer's job
    logic [7:0]  since_inj;         // cycles since the last injection was issued (saturating)
    logic        inj_first;         // a job's first value issues (an "injection")
    logic        hand;              // a job's last value issues: the drain takes the job
    logic signed [DATA_W-1:0] arr_a [N];
    logic signed [DATA_W-1:0] arr_b [N];
    logic signed [ACC_W-1:0]  arr_shadow [N][N];

    always_comb begin
        for (int r = 0; r < N; r++) arr_a[r] = DATA_W'(at_rdata >> (8 * r));
        for (int c = 0; c < N; c++) arr_b[c] = DATA_W'(wb_rdata >> (8 * c));
    end

    // ---------------------------------------------------------------- depthwise streamer
    typedef enum logic [1:0] {D_IDLE, D_RUN, D_TAIL, D_DONE} dstate_e;
    dstate_e dstate;
    logic        dw_s2;             // stride 2 (else 1)
    logic [4:0]  seg_m1;            // SEG - 1
    // unit loop: channel block (outer), output row, N-pixel segment of the row (inner)
    logic [5:0]  dw_cb;
    logic [6:0]  dw_y, dw_x0;
    logic        dw_more;           // units left to read
    logic signed [9:0]  u_iy0, u_ix0;               // window corner of the segment's pixel 0
    logic signed [19:0] u_base, y_base, cb_base;    // its word address (may be out of range)
    logic [12:0] u_ob, y_ob, cb_ob;                 // output word address of pixel 0
    logic [9:0]  cb_wb;                             // weight buffer address of the block's taps
    logic [3:0]  u_nval;                            // pixels of the segment inside the row
    // read cursor within the unit: kernel row rd_ky, column rd_j of SEG
    logic [2:0]  rd_ky;
    logic [4:0]  rd_j;
    logic signed [9:0]  rd_iy, rd_ix;
    logic signed [19:0] rd_addr, rd_row;
    logic        ustart;            // the cursor is at a unit's first word
    logic signed [9:0]  e_iy, e_ix; // effective cursor (a unit's first word comes from u_*)
    logic signed [19:0] e_addr, e_row;
    logic        rd_inb, rd_go, dd_pad;
    logic        u_set;             // param set of the unit being read
    logic [9:0]  w_row, e_w;        // weight buffer address of kernel row rd_ky's taps
    // taps: the read of word rd_j of a kernel row issues tap kx (if any) in the same cycle
    logic        dw_tap, dw_tap_first, dw_tap_last;
    logic [1:0]  dw_kx;
    logic [WORD_W-1:0] dw_a;
    always_comb begin
        ustart = (rd_j == 0) && (rd_ky == 0);
        e_iy   = ustart ? u_iy0 : rd_iy;
        e_ix   = ustart ? u_ix0 : rd_ix;
        e_addr = ustart ? u_base : rd_addr;
        e_row  = ustart ? u_base : rd_row;
        e_w    = ustart ? cb_wb : w_row;
        rd_inb = e_iy >= 0 && e_iy < 10'(I.in_h) && e_ix >= 0 && e_ix < 10'(I.in_w);
        // a unit's first read is also its first tap (injection): PERIOD_DW after the last
        rd_go  = !ustart || (dw_more && since_inj >= 8'(PERIOD_DW - 1) && p_ready(dw_cb));
        // stride 1 reads columns 0..9 in order: tap kx = word kx. Stride 2 reads the even
        // columns 0,2..16 then the odd ones 1..15: kx 0 = word 0, kx 2 = word 1, kx 1 = word 9.
        if (!dw_s2) begin
            dw_tap = (rd_j < 5'(I.ksize));
            dw_kx  = 2'(rd_j);
            dw_tap_last = (rd_j == 5'(I.ksize - 3'd1));
        end else begin
            dw_tap = (rd_j == 0) || (rd_j == 1 && I.ksize == 3'd3) || (rd_j == 9 && I.ksize >= 3'd2);
            dw_kx  = (rd_j == 0) ? 2'd0 : (rd_j == 1) ? 2'd2 : 2'd1;
            dw_tap_last = (I.ksize == 3'd1) ? (rd_j == 0) : (rd_j == 9);
        end
        dw_tap       = (dstate == D_RUN) && rd_go && dw_tap;
        dw_tap_last  = dw_tap && dw_tap_last && (rd_ky == 3'(I.ksize - 3'd1));
        dw_tap_first = dw_tap && ustart;
        u_nval = (I.out_w - dw_x0 >= 7'(N)) ? 4'(N) : 4'(I.out_w - dw_x0);
        // the word read last cycle: row r of the array uses it r cycles after row 0
        dw_a   = dd_pad ? {N{I.in_zp}} : act_rdata;
    end

    npu_array u_array (.clk, .rst, .valid(st_valid), .last(st_last), .a(arr_a), .b(arr_b),
                       .dw(is_dw), .dw_a, .shadow(arr_shadow));

    // ---------------------------------------------------------------- drain (concurrent)
    logic        dr_busy, dr_run;
    logic [2:0]  dr_wait;
    logic [5:0]  dr_idx;            // column-pair index: row dr_idx[4:2], pair dr_idx[1:0]
    logic [ACT_AW-1:0] job_ob;      // output word address of the drained job's first pixel
    logic [3:0]  dr_nval;           // rows that are real pixels
    // per-channel params: prefetched when a job starts (CONV: its first value; DWCONV: its
    // first read) into one of two sets of a small LUT RAM, alternating, which the drain reads
    // once it takes the job, so it never waits on the param buffer. Even and odd channels are
    // separate RAMs (one read port each, for the two requant pipelines), addressed
    // {set, channel pair}.
    logic        wset, rset;        // set the prefetch writes / the drain reads
    logic [PARAM_W-1:0] prm_rd [2];
    logic [3:0]  pf_j;              // prefetch index (N = done)
    logic [5:0]  pf_g;              // output group being prefetched
    logic        pf_v;
    logic [2:0]  pf_jd;
    for (genvar q = 0; q < 2; q++) begin : g_prm
        logic [PARAM_W-1:0] ram [N];        // {set, channel pair}
        always_ff @(posedge clk) if (pf_v && pf_jd[0] == 1'(q)) ram[{wset, pf_jd[2:1]}] <= pb_rdata;
        assign prm_rd[q] = ram[{rset, dr_idx[1:0]}];
    end
    // two requant pipelines: even and odd column of each pair
    logic signed [ACC_W-1:0]   rq_acc   [2];
    logic signed [31:0]        rq_bias  [2], rq_mult [2];
    logic signed [SHIFT_W-1:0] rq_shift [2];
    logic signed [7:0]         rq_out   [2];
    logic [DQ-1:0]           tq_v;
    logic [1:0]              tq_cp  [DQ];
    logic                    tq_wr  [DQ];           // last pair of a valid pixel
    logic                    tq_first [DQ];         // a job's first pair
    logic [ACT_AW-1:0]       wa;                    // output side: the pixel's word address
    logic [WORD_W-17:0]      row_word;              // bytes 0..N-3 of the pixel being drained

    for (genvar q = 0; q < 2; q++) begin : g_rq
        npu_requant u_rq (
            .clk, .acc(rq_acc[q]), .bias(rq_bias[q]), .mult(rq_mult[q]), .shift(rq_shift[q]),
            .rmode(I.rmode), .out_zp(I.out_zp), .act_min(I.act_min), .act_max(I.act_max),
            .out(rq_out[q]));
    end

    always_comb begin
        act_we    = tq_v[DQ-1] && tq_wr[DQ-1];
        act_waddr = wa;
        act_wdata = {rq_out[1], rq_out[0], row_word};
    end

    // ---------------------------------------------------------------- buffer addressing
    always_comb begin
        if (is_dw)        act_raddr = ACT_AW'(e_addr);
        else if (pk_mode) act_raddr = ACT_AW'((g_addr >>> 3) + 20'(ph));
        else              act_raddr = ACT_AW'(g_addr);
        if (is_dw)        wb_raddr = e_w + WBUF_AW'(dw_kx);
        else if (!cached) wb_raddr = wdbl ? {c_g[0], (WBUF_AW-1)'(sk)} : WBUF_AW'(sk);   // group mode
        else              wb_raddr = c_wb + WBUF_AW'(sk);
        at_raddr = {sbuf, KA_AW'(sk)};
        pb_raddr = PBUF_AW'({pf_g, I.uniform ? 3'd0 : pf_j[2:0]});                         // param prefetch
        // external loads land in the buffers
        wb_we    = ext_rvalid && w_loading;
        wb_waddr = (!cached && wdbl) ? {lg[0], (WBUF_AW-1)'(ld_cnt)} : WBUF_AW'(ld_cnt);
        pb_we    = ext_rvalid && p_loading && ld_cnt[0];
        pb_waddr = (lstate == L_PLOAD) ? PBUF_AW'({lg[5:0], 3'b000}) + PBUF_AW'(ld_cnt >> 1)
                                       : PBUF_AW'(ld_cnt >> 1);
        pb_wdata = {SHIFT_W'(ext_rdata), p_word0[63:32], p_word0[31:0]};
    end
    assign w_loading = (state == S_WLOAD) || (lstate == L_LOAD);
    assign p_loading = (plstate == PL_LOAD) || (lstate == L_PLOAD);
    assign inj_first = (cstate == C_STREAM && sk == '0) || dw_tap_first;
    assign hand      = (cstate == C_STREAM && sk == K - 11'd1) || dw_tap_last;

    // ---------------------------------------------------------------- sequential logic
    always_ff @(posedge clk) begin
        done      <= 1'b0;
        run_start <= 1'b0;
        d_v       <= 1'b0;
        st_valid  <= 1'b0;
        st_last   <= 1'b0;
        if (since_inj != 8'hFF) since_inj <= since_inj + 8'd1;

        // ---- instruction constants: shift-and-add steps, then the derived ones
        if (m_cnt != 0) begin
            m_cnt <= (m_cnt == 4'd8) ? 4'd0 : m_cnt + 4'd1;
            if (m_cnt != 4'd8) begin
                if (m_b1[0]) layer_words <= layer_words + m_a1;
                if (m_b2[0]) P           <= P + m_a2;
                if (m_b3[0]) RS          <= RS + m_a3;
                if (m_b4[0]) OWG         <= OWG + m_a4;
                m_a1 <= m_a1 << 1; m_a2 <= m_a2 << 1; m_a3 <= m_a3 << 1; m_a4 <= m_a4 << 1;
                m_b1 <= m_b1 >> 1; m_b2 <= m_b2 >> 1; m_b3 <= m_b3 >> 1; m_b4 <= m_b4 >> 1;
            end else begin
                SRS  <= 15'(mul3(15'(RS), {1'b0, I.stride}));
                PADR <= 16'(mul3(15'(RS), I.pad));
            end
        end

        // ---- gather data landing: one row word per (k-word, row), then the transposer
        if (d_v && pk_mode) w_hist[d_r] <= act_rdata;
        if (tb_we) begin
            if (d_r == 3'(N - 1)) begin
                tb_sel    <= !tb_sel;
                wo_active <= 1'b1;
                wo_buf    <= tb_sel;
                wo_cnt    <= '0;
                wo_kw     <= d_k;
                wo_half   <= d_half;
                wo_lastu  <= d_lastu;
            end
        end
        if (wo_active && wo_lastu && wo_cnt == 3'(N - 1)) begin
            // the unit's last block is written: publish it to the consumer
            full[pub_half]      <= 1'b1;
            unit_p0[pub_half]   <= pub_p0;
            unit_ob[pub_half]   <= pub_ob;
            unit_g[pub_half]    <= pub_g;
            unit_last[pub_half] <= pub_last;
        end
        if (wo_active && !(d_v && (!pk_mode || d_ph) && d_r == 3'(N - 1))) begin
            wo_cnt <= wo_cnt + 3'd1;
            if (wo_cnt == 3'(N - 1)) wo_active <= 1'b0;
        end

        // ---- depthwise taps: issued with the read whose word they use (it reaches the
        // array with the weight, one cycle later)
        if (dw_tap) begin
            st_valid <= 1'b1;
            st_last  <= dw_tap_last;
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

        // ---- injections, param prefetch starts, and handing finished jobs to the drain
        if (inj_first) since_inj <= '0;
        if (!is_dw && inj_first) begin             // CONV: fetch this job's params
            pf_j  <= '0;
            pf_g  <= c_g;
            wset  <= !wset;
            c_set <= !wset;
        end
        if (hand) begin
            dr_busy <= 1'b1;
            dr_wait <= 3'(DRAIN_DELAY - 1);
            dr_run  <= 1'b0;
            if (is_dw) begin
                job_ob  <= ACT_AW'(u_ob);
                dr_nval <= u_nval;
                rset    <= u_set;
            end else begin
                job_ob  <= ACT_AW'(c_ob + 13'(c_g));
                dr_nval <= (P - c_p0 >= 14'(N)) ? 4'(N) : 4'(P - c_p0);
                rset    <= c_set;
            end
        end

        // ---- drain: wait DRAIN_DELAY, then issue N*N/2 column-pair reads (row-major)
        tq_v <= {tq_v[DQ-2:0], 1'b0};
        for (int i = DQ - 1; i > 0; i--) begin
            tq_cp[i]  <= tq_cp[i-1];
            tq_wr[i]  <= tq_wr[i-1];
            tq_first[i] <= tq_first[i-1];
        end
        if (dr_busy && !hand) begin
            if (!dr_run) begin
                if (dr_wait == 0) begin
                    dr_run <= 1'b1;
                    dr_idx <= '0;
                end else begin
                    dr_wait <= dr_wait - 3'd1;
                end
            end else begin
                logic [2:0]  r, c0;
                r  = dr_idx[4:2];
                c0 = {dr_idx[1:0], 1'b0};
                for (int q = 0; q < 2; q++) begin
                    rq_acc[q]   <= arr_shadow[r][c0 + 3'(q)];
                    rq_bias[q]  <= 32'(prm_rd[q]);
                    rq_mult[q]  <= 32'(prm_rd[q] >> 32);
                    rq_shift[q] <= SHIFT_W'(prm_rd[q] >> 64);
                end
                tq_v[0]   <= 1'b1;
                tq_cp[0]  <= dr_idx[1:0];
                tq_wr[0]  <= (dr_idx[1:0] == 2'(N / 2 - 1)) && (4'(r) < dr_nval);
                tq_first[0] <= (dr_idx == '0);
                dr_idx    <= dr_idx + 6'd1;
                if (dr_idx == 6'(N * N / 2 - 1)) begin
                    dr_busy <= 1'b0;
                    dr_run  <= 1'b0;
                end
            end
        end
        if (tq_v[DQ-1] && tq_cp[DQ-1] != 2'(N / 2 - 1))
            row_word[16 * tq_cp[DQ-1] +: 16] <= {rq_out[1], rq_out[0]};
        // output-side address (the job's base is held from the hand-over until its first pair
        // comes out, well before the next hand-over): row r's pixel goes to base + r*out_g
        if (tq_v[DQ-1]) begin
            if (tq_first[DQ-1])                    wa <= job_ob;
            else if (tq_cp[DQ-1] == 2'(N / 2 - 1)) wa <= wa + ACT_AW'(I.out_g);
        end

        if (rst) begin
            state      <= S_IDLE;
            pstate     <= P_IDLE;
            cstate     <= C_IDLE;
            lstate     <= L_IDLE;
            dstate     <= D_IDLE;
            wo_active  <= 1'b0;
            ext_req    <= 1'b0;
            plstate    <= PL_IDLE;
            dr_busy    <= 1'b0;
            dr_run     <= 1'b0;
            tq_v       <= '0;
            since_inj  <= 8'hFF;
            pf_j       <= 4'(N);
            full       <= '0;
            m_cnt      <= '0;
            dd_pad     <= 1'b0;
            is_dw      <= 1'b0;
            wset       <= 1'b0;
        end else begin
            // ======================================================== main
            unique case (state)
                S_IDLE: if (start) state <= S_INIT;

                S_INIT: begin
                    logic [8:0]  sc;
                    logic [3:0]  rw;
                    logic [5:0]  t;
                    logic [10:0] k;
                    t  = 6'(mul3(15'(I.ksize), I.ksize));
                    sc = 9'(mul3(15'(I.in_cb), I.ksize));
                    rw = 4'((sc + 9'(N - 1)) >> 3);
                    // packed: K = ksize * KR, KR = ksize*C rounded up to a multiple of N;
                    // channel-blocked: K = ksize^2 * in_cb * N
                    k  = I.in_src ? 11'(mul3(15'(rw), I.ksize)) << 3 : 11'(mul3(15'(sc), I.ksize)) << 3;
                    is_dw   <= (I.op == OP_DWCONV);
                    pk_mode <= (I.op == OP_CONV) && I.in_src;
                    taps    <= t;
                    ksc     <= sc;
                    krw     <= rw;
                    K       <= k;
                    SC      <= 8'(mul3(15'(I.in_cb), {1'b0, I.stride}));
                    PADC    <= 9'(mul3(15'(I.in_cb), I.pad));
                    NOG     <= {I.out_g, 3'b000};
                    dw_s2   <= (I.stride == 2'd2);
                    seg_m1  <= (I.stride == 2'd2) ? 5'(7 * 2 + 2) : 5'(7 + 2);
                    // multiplies: layer_words, P, RS, OWG
                    layer_words <= '0; P <= '0; RS <= '0; OWG <= '0;
                    m_a1 <= 17'((I.op == OP_DWCONV) ? 11'(t) : k);  m_b1 <= I.uniform ? 7'd1 : 7'(I.out_g);
                    m_a2 <= 14'(I.out_w);                          m_b2 <= I.out_h;
                    m_a3 <= 13'(I.in_w);                           m_b3 <= 7'(I.in_cb);
                    m_a4 <= 13'(I.out_w);                          m_b4 <= 7'(I.out_g);
                    m_cnt <= 4'd1;
                    full       <= '0;
                    state      <= S_CONST;
                end

                // the constants (layer_words decides the mode)
                S_CONST: if (m_cnt == 0) begin
                    if (cached) begin
                        state <= S_WLOAD_REQ;
                    end else begin
                        run_start <= 1'b1;      // group mode: weights and params per group
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

                S_RUN: if (is_dw ? (dstate == D_DONE) : (cstate == C_DONE)) state <= S_FINISH;

                S_FINISH: if (!dr_busy && !(|tq_v)) state <= S_DONE;

                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase

            // ======================================================== producer
            unique case (pstate)
                P_IDLE: if (run_start && !is_dw) begin
                    gbuf   <= 1'b0;
                    p_p0   <= '0;
                    p_ob   <= I.out_base[12:0];
                    pw     <= w_origin;
                    p_g    <= '0;
                    p_new_tile <= 1'b1;
                    pstate <= P_WAIT_BUF;
                end

                P_WAIT_BUF: if (!full[gbuf]) begin
                    koff <= '0;
                    krow <= '0;
                    if (p_new_tile) begin
                        pstate <= P_SETUP;
                        sr <= '0;
                        cw <= pw;
                    end else begin
                        pstate <= P_GATHER;
                        gr <= '0; gk <= '0; ky <= '0; kx <= '0; kc <= '0; kj <= '0; ph <= 1'b0;
                    end
                end

                P_SETUP: begin
                    row_iy0[sr[2:0]]   <= cw.wy;
                    row_ix0[sr[2:0]]   <= cw.wx;
                    row_base[sr[2:0]]  <= cw.wyb + cw.wxb;
                    row_cxb[sr[2:0]]   <= cw.wxb;
                    row_valid[sr[2:0]] <= (p_p0 + 14'(sr)) < P;
                    cw <= walk_step(cw, I.out_w, I.stride, I.pad, SC, SRS, PADC);
                    if (sr == 4'(N - 1)) begin
                        pnw    <= walk_step(cw, I.out_w, I.stride, I.pad, SC, SRS, PADC);
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
                    d_sh    <= g_addr[2:0];
                    d_s     <= pk_s;
                    d_e     <= pk_e;
                    d_p     <= pk_p;
                    d_rowok <= row_ok;
                    d_half  <= gbuf;
                    d_lastu <= gather_last;
                    if (gather_last) begin
                        // last read issued: latch the unit for publishing, advance to the next
                        // unit (group mode: tiles inside a group; cached: tiles only) and start
                        // on it right away
                        logic last_unit;
                        pub_half <= gbuf;
                        pub_p0   <= p_p0;
                        pub_ob   <= p_ob;
                        pub_g    <= p_g;
                        gbuf     <= !gbuf;
                        p_new_tile <= 1'b1;
                        if (!cached) begin
                            last_unit = p_last_g && p_last_tile;
                            if (!p_last_tile) begin
                                p_p0 <= p_p0 + 14'(N);
                                p_ob <= p_ob + 13'(NOG);
                                pw   <= pnw;
                            end else begin
                                p_g  <= p_g + 6'd1;
                                p_p0 <= '0;
                                p_ob <= I.out_base[12:0];
                                pw   <= w_origin;
                            end
                        end else begin
                            last_unit = p_last_tile;
                            p_p0 <= p_p0 + 14'(N);
                            p_ob <= p_ob + 13'(NOG);
                            pw   <= pnw;
                        end
                        pub_last <= last_unit;
                        pstate   <= last_unit ? P_IDLE : P_WAIT_BUF;
                    end else if (pk_mode && !ph) begin
                        ph <= 1'b1;                     // second aligned word of this window
                    end else if (gr != 3'(N - 1)) begin
                        ph <= (kj != 0);
                        gr <= gr + 3'd1;
                    end else begin
                        // next k-word: offsets run contiguously along a kernel row (word mode:
                        // kx*in_cb + kc; packed: kj*N), then jump to the next kernel row
                        ph <= pk_mode && !krow_end;     // next window: phase 1 only unless it starts a kernel row
                        gr <= '0;
                        gk <= gk + 11'd1;
                        if (krow_end) begin
                            krow <= krow + 16'(RS);
                            koff <= krow + 16'(RS);
                        end else begin
                            koff <= koff + (pk_mode ? 16'(N) : 16'd1);
                        end
                        if (pk_mode) begin
                            if (kj == krw - 4'd1) begin
                                kj <= '0;
                                ky <= ky + 4'd1;
                            end else begin
                                kj <= kj + 4'd1;
                            end
                        end else if (kc == I.in_cb - 6'd1) begin
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

                default: pstate <= P_IDLE;
            endcase

            // ======================================================== consumer
            unique case (cstate)
                C_IDLE: if (run_start && !is_dw) begin
                    sbuf      <= 1'b0;
                    cons_done <= '0;
                    cstate    <= C_WAIT_FULL;
                end

                // a gathered unit, and (group mode) its group's weights
                C_WAIT_FULL: if (full[sbuf] && (cached || loaded_cnt > 7'(unit_g[sbuf]))) begin
                    c_p0        <= unit_p0[sbuf];
                    c_ob        <= unit_ob[sbuf];
                    c_g         <= cached ? 6'd0 : unit_g[sbuf];
                    c_wb        <= '0;
                    c_last_unit <= unit_last[sbuf];
                    cstate      <= C_INJ_WAIT;
                end

                C_INJ_WAIT: if (since_inj >= 8'(PERIOD_MIN - 1) && (!cached || p_ready(c_g))) begin
                    sk     <= '0;
                    cstate <= C_STREAM;
                end

                C_STREAM: begin
                    st_valid <= 1'b1;
                    st_last  <= (sk == K - 11'd1);
                    if (sk == K - 11'd1) cstate <= C_NEXT;
                    else                 sk <= sk + 11'd1;
                end

                C_NEXT: begin
                    if (!c_last_job) begin
                        c_g    <= c_g + 6'd1;            // next group of the same tile
                        c_wb   <= c_wb + 10'(K);
                        cstate <= C_INJ_WAIT;
                    end else begin
                        full[sbuf] <= 1'b0;              // A_T half free for the producer
                        sbuf       <= !sbuf;
                        cstate     <= c_last_unit ? C_DONE : C_WAIT_FULL;
                        // group mode: the group's last tile frees its weight-buffer half
                        if (!cached && (c_p0 + 14'(N)) >= P) cons_done <= cons_done + 7'd1;
                    end
                end

                C_DONE: if (state == S_FINISH) cstate <= C_IDLE;

                default: cstate <= C_IDLE;
            endcase

            // ======================================================== weight loader
            unique case (lstate)
                L_IDLE: begin
                    if (run_start) begin
                        lg         <= '0;
                        l_waddr    <= I.w_base;
                        l_paddr    <= I.p_base;
                        loaded_cnt <= '0;
                    end else if (state == S_RUN && !cached && l_ok) begin
                        lstate <= L_REQ;
                    end
                end

                L_REQ: begin
                    ext_req  <= 1'b1;
                    ext_addr <= EXT_AW'(l_waddr);
                    ext_len  <= 16'(K);
                    ld_len   <= 17'(K);
                    ld_cnt   <= '0;
                    if (ext_req && ext_ack) begin
                        ext_req <= 1'b0;
                        lstate  <= L_LOAD;
                    end
                end

                L_LOAD: if (ld_cnt == ld_len) begin
                    l_waddr    <= l_waddr + 16'(K);
                    lstate     <= L_PREQ;
                end

                // the group's N param entries
                L_PREQ: begin
                    ext_req  <= 1'b1;
                    ext_addr <= EXT_AW'(l_paddr);
                    ext_len  <= 16'(N * PARAM_WORDS);
                    ld_len   <= 17'(N * PARAM_WORDS);
                    ld_cnt   <= '0;
                    if (ext_req && ext_ack) begin
                        ext_req <= 1'b0;
                        lstate  <= L_PLOAD;
                    end
                end

                L_PLOAD: if (ld_cnt == ld_len) begin
                    loaded_cnt <= lg + 7'd1;
                    lg         <= lg + 7'd1;
                    l_paddr    <= l_paddr + 16'(N * PARAM_WORDS);
                    lstate     <= L_IDLE;
                end

                default: lstate <= L_IDLE;
            endcase

            // ======================================================== param loader
            // (cached and depthwise layers: the weights have landed, the port is free)
            unique case (plstate)
                PL_IDLE: if (run_start && cached) plstate <= PL_REQ;

                PL_REQ: begin
                    ext_req  <= 1'b1;
                    ext_addr <= EXT_AW'(I.p_base);
                    ext_len  <= I.uniform ? 16'(PARAM_WORDS) : 16'({I.out_g, 4'b0000});   // entries * PARAM_WORDS
                    ld_len   <= I.uniform ? 17'(PARAM_WORDS) : 17'({I.out_g, 4'b0000});
                    ld_cnt   <= '0;
                    if (ext_req && ext_ack) begin
                        ext_req <= 1'b0;
                        plstate <= PL_LOAD;
                    end
                end

                PL_LOAD: if (ld_cnt == ld_len) plstate <= PL_DONE;

                PL_DONE: if (state == S_DONE) plstate <= PL_IDLE;

                default: plstate <= PL_IDLE;
            endcase

            // ======================================================== depthwise streamer
            unique case (dstate)
                D_IDLE: if (run_start && is_dw) begin
                    dw_cb   <= '0;
                    dw_y    <= '0;
                    dw_x0   <= '0;
                    dw_more <= 1'b1;
                    u_iy0   <= -10'(I.pad);
                    u_ix0   <= -10'(I.pad);
                    cb_base <= 20'(I.in_base) - 20'(PADR) - 20'(PADC);
                    y_base  <= 20'(I.in_base) - 20'(PADR) - 20'(PADC);
                    u_base  <= 20'(I.in_base) - 20'(PADR) - 20'(PADC);
                    cb_ob   <= I.out_base[12:0];
                    y_ob    <= I.out_base[12:0];
                    u_ob    <= I.out_base[12:0];
                    cb_wb   <= '0;
                    rd_ky   <= '0;
                    rd_j    <= '0;
                    dstate  <= D_RUN;
                end

                D_RUN: begin
                    if (ustart && !dw_more) dstate <= D_TAIL;
                    if (rd_go && ustart) begin
                        // a unit's first read: fetch its params
                        pf_j  <= '0;
                        pf_g  <= I.uniform ? 6'd0 : dw_cb;
                        wset  <= !wset;
                        u_set <= !wset;
                    end
                    if (rd_go) begin
                        dd_pad <= !rd_inb;
                        if (rd_j == seg_m1) begin
                            // kernel row complete
                            rd_j  <= '0;
                            rd_ix <= u_ix0;
                            w_row <= e_w + 10'(I.ksize);
                            if (rd_ky == 3'(I.ksize - 3'd1)) begin
                                rd_ky <= '0;
                                // next unit: segment, then row, then channel block
                                if (8'(dw_x0) + 8'(N) < 8'(I.out_w)) begin
                                    dw_x0  <= dw_x0 + 7'(N);
                                    u_ix0  <= u_ix0 + (dw_s2 ? 10'(2 * N) : 10'(N));
                                    u_base <= u_base + 20'({SC, 3'b000});
                                    u_ob   <= u_ob + 13'(NOG);
                                end else begin
                                    dw_x0 <= '0;
                                    u_ix0 <= -10'(I.pad);
                                    if (dw_y != I.out_h - 7'd1) begin
                                        dw_y   <= dw_y + 7'd1;
                                        u_iy0  <= u_iy0 + 10'(I.stride);
                                        y_base <= y_base + 20'(SRS);
                                        u_base <= y_base + 20'(SRS);
                                        y_ob   <= y_ob + OWG;
                                        u_ob   <= y_ob + OWG;
                                    end else begin
                                        dw_y    <= '0;
                                        u_iy0   <= -10'(I.pad);
                                        dw_cb   <= dw_cb + 6'd1;
                                        cb_base <= cb_base + 20'd1;
                                        y_base  <= cb_base + 20'd1;
                                        u_base  <= cb_base + 20'd1;
                                        cb_ob   <= cb_ob + 13'd1;
                                        y_ob    <= cb_ob + 13'd1;
                                        u_ob    <= cb_ob + 13'd1;
                                        cb_wb   <= I.uniform ? cb_wb : cb_wb + 10'(taps);
                                        if (dw_cb == I.out_g - 6'd1) dw_more <= 1'b0;
                                    end
                                end
                            end else begin
                                rd_ky   <= rd_ky + 3'd1;
                                rd_iy   <= e_iy + 10'sd1;
                                rd_row  <= e_row + 20'(RS);
                                rd_addr <= e_row + 20'(RS);
                            end
                        end else begin
                            // next column: stride 1 in order; stride 2 evens, then odds
                            rd_j <= rd_j + 5'd1;
                            if (!dw_s2) begin
                                rd_ix   <= e_ix + 10'sd1;
                                rd_addr <= e_addr + 20'(I.in_cb);
                            end else if (rd_j == 5'd8) begin
                                rd_ix   <= u_ix0 + 10'sd1;
                                rd_addr <= e_row + 20'(I.in_cb);
                            end else begin
                                rd_ix   <= e_ix + 10'sd2;
                                rd_addr <= e_addr + 20'({I.in_cb, 1'b0});
                            end
                            if (ustart) begin
                                rd_iy  <= e_iy;
                                rd_row <= e_row;
                                w_row  <= e_w;
                            end
                        end
                    end
                end

                // (every tap issued with its read)
                D_TAIL: dstate <= D_DONE;

                D_DONE: if (state == S_FINISH) dstate <= D_IDLE;

                default: dstate <= D_IDLE;
            endcase
        end
    end

    // pixel 0's walker state (stride/pad offsets of the tile origin)
    always_comb begin
        w_origin.x   = '0;
        w_origin.wy  = -10'(I.pad);
        w_origin.wx  = -10'(I.pad);
        w_origin.wyb = -20'(PADR);
        w_origin.wxb = -20'(PADC);
    end

endmodule

`default_nettype wire
