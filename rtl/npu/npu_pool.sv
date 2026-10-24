// npu_pool: MAXPOOL on channel-blocked activations (8 lanes per word).
//
// For each output pixel and channel block: read the k x k window (one word per cycle,
// out-of-bounds taps skipped), take the per-lane max, clamp to [act_min, act_max] and write.
// (AVGPOOL is lowered by the compiler to a ones-weight DWCONV.)
// Addresses are incremental; the input row stride in_w * in_cb is formed by shift-and-add
// at the start of the instruction (S_MUL).

`default_nettype none

module npu_pool
    import npu_pkg::*;
(
    input  wire logic              clk,
    input  wire logic              rst,
    input  wire logic              start,
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire instr_t            ins,           // held by the sequencer while the unit runs
    /* verilator lint_on UNUSEDSIGNAL */
    output      logic              done,
    output      logic [ACT_AW-1:0] act_raddr,
    input  wire logic [WORD_W-1:0] act_rdata,
    output      logic              act_we,
    output      logic [ACT_AW-1:0] act_waddr,
    output      logic [WORD_W-1:0] act_wdata
);

    typedef enum logic [2:0] {S_IDLE, S_INIT, S_MUL, S_TAPS, S_TAIL, S_WRITE, S_DONE} state_e;
    state_e state;

    logic [6:0]  oy, ox;
    logic [5:0]  cb;
    logic [2:0]  ky, kx;
    logic [12:0] RS;                // in_w * in_cb
    logic [12:0] m_a;               // shift-and-add operands
    logic [5:0]  m_b;
    logic [2:0]  m_cnt;

    // window corner of the current output pixel: wy = oy*stride - pad (wx likewise) and its
    // word address (in_base + (wy*in_w + wx)*in_cb, possibly out of range); row start too
    logic signed [9:0]  wy, wx, iy, ix;
    logic signed [19:0] c_addr, r_addr, t_addr, t_row;
    logic [ACT_AW-1:0]  w_addr;     // output word: out_base + p*in_cb + cb
    logic              inb;
    always_comb begin
        iy  = wy + 10'(ky);
        ix  = wx + 10'(kx);
        inb = iy >= 0 && iy < 10'(ins.in_h) && ix >= 0 && ix < 10'(ins.in_w);
        act_raddr = ACT_AW'(t_addr);
    end

    function automatic logic [19:0] mul3(input logic [12:0] a, input logic [2:0] b);
        return (b[0] ? 20'(a) : 20'd0) + (b[1] ? 20'(a) << 1 : 20'd0) + (b[2] ? 20'(a) << 2 : 20'd0);
    endfunction

    logic        d_valid, d_inb, last_tap;
    logic signed [7:0]  lane_max [N];

    assign last_tap = (ky == ins.ksize - 3'd1) && (kx == ins.ksize - 3'd1);

    always_comb begin
        act_we    = (state == S_WRITE);
        act_waddr = w_addr;
        for (int l = 0; l < N; l++) begin
            if (lane_max[l] < ins.act_min)      act_wdata[8*l +: 8] = ins.act_min;
            else if (lane_max[l] > ins.act_max) act_wdata[8*l +: 8] = ins.act_max;
            else                                act_wdata[8*l +: 8] = lane_max[l];
        end
    end

    always_ff @(posedge clk) begin
        done    <= 1'b0;
        d_valid <= 1'b0;

        // reduce the word that arrived from the previous cycle's read
        if (d_valid && d_inb) begin
            for (int l = 0; l < N; l++) begin
                logic signed [7:0] x;
                x = 8'(act_rdata >> (8 * l));
                if (x > lane_max[l]) lane_max[l] <= x;
            end
        end

        if (rst) begin
            state <= S_IDLE;
        end else begin
            unique case (state)
                S_IDLE: if (start) state <= S_INIT;

                S_INIT: begin
                    oy <= '0; ox <= '0; cb <= '0; ky <= '0; kx <= '0;
                    for (int l = 0; l < N; l++) lane_max[l] <= -8'sd128;
                    RS    <= '0;
                    m_a   <= 13'(ins.in_w);
                    m_b   <= ins.in_cb;
                    m_cnt <= '0;
                    state <= S_MUL;
                end

                // RS = in_cb * in_w (6 steps), then the first pixel's corner
                S_MUL: begin
                    m_cnt <= m_cnt + 3'd1;
                    if (m_cnt != 3'd6) begin
                        if (m_b[0]) RS <= RS + m_a;
                        m_a <= m_a << 1;
                        m_b <= m_b >> 1;
                    end else begin
                        logic signed [19:0] c0;
                        c0 = 20'(ins.in_base) - 20'(mul3(RS, ins.pad)) - 20'(mul3(13'(ins.in_cb), ins.pad));
                        wy     <= -10'(ins.pad);
                        wx     <= -10'(ins.pad);
                        c_addr <= c0;
                        r_addr <= c0;
                        t_addr <= c0;
                        t_row  <= c0;
                        w_addr <= ACT_AW'(ins.out_base);
                        state  <= S_TAPS;
                    end
                end

                S_TAPS: begin
                    d_valid <= 1'b1;
                    d_inb   <= inb;
                    if (last_tap) begin
                        state <= S_TAIL;
                    end else if (kx == ins.ksize - 3'd1) begin
                        kx     <= '0;
                        ky     <= ky + 3'd1;
                        t_row  <= t_row + 20'(RS);
                        t_addr <= t_row + 20'(RS);
                    end else begin
                        kx     <= kx + 3'd1;
                        t_addr <= t_addr + 20'(ins.in_cb);
                    end
                end

                // the last word is reduced this cycle
                S_TAIL: state <= S_WRITE;

                // write this block's result and move on: next block, or next pixel
                S_WRITE: begin
                    logic signed [19:0] c;
                    logic [5:0] nb;
                    nb = (cb == ins.in_cb - 6'd1) ? 6'd0 : cb + 6'd1;
                    w_addr <= w_addr + ACT_AW'(1);
                    ky <= '0; kx <= '0;
                    for (int l = 0; l < N; l++) lane_max[l] <= -8'sd128;
                    c = c_addr;
                    state <= S_TAPS;
                    if (cb == ins.in_cb - 6'd1) begin
                        cb <= '0;
                        if (oy == ins.out_h - 7'd1 && ox == ins.out_w - 7'd1) begin
                            state <= S_DONE;
                        end else if (ox == ins.out_w - 7'd1) begin
                            ox <= '0;
                            oy <= oy + 7'd1;
                            wx <= -10'(ins.pad);
                            wy <= wy + 10'(ins.stride);
                            c  = r_addr + 20'(mul3(RS, {1'b0, ins.stride}));
                            r_addr <= c;
                        end else begin
                            ox <= ox + 7'd1;
                            wx <= wx + 10'(ins.stride);
                            c  = c_addr + 20'(mul3(13'(ins.in_cb), {1'b0, ins.stride}));
                        end
                        c_addr <= c;
                    end else begin
                        cb <= cb + 6'd1;
                    end
                    // the next taps start at the corner of the (new) pixel, plus the block
                    t_addr <= c + 20'(nb);
                    t_row  <= c + 20'(nb);
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
