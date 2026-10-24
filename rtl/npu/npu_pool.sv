// npu_pool: MAXPOOL and AVGPOOL on channel-blocked activations (8 lanes per word).
//
// For each output pixel and channel block: read the k x k window (one word per cycle,
// out-of-bounds taps skipped), reduce per lane, then
//   MAXPOOL: max                          AVGPOOL (TFLite reference int8):
//                                         q = sign(s) * ((|s| + n/2) / n), n = in-bounds taps
// and clamp to [act_min, act_max]. The division is an 8-lane restoring divider (16 cycles).

`default_nettype none

module npu_pool
    import npu_pkg::*;
(
    input  wire logic              clk,
    input  wire logic              rst,
    input  wire logic              start,
    input  wire instr_t            ins,
    output      logic              done,
    output      logic [ACT_AW-1:0] act_raddr,
    input  wire logic [WORD_W-1:0] act_rdata,
    output      logic              act_we,
    output      logic [ACT_AW-1:0] act_waddr,
    output      logic [WORD_W-1:0] act_wdata
);

    typedef enum logic [2:0] {S_IDLE, S_INIT, S_TAPS, S_TAIL, S_DIV, S_WRITE, S_NEXT, S_DONE} state_e;
    state_e state;

    /* verilator lint_off UNUSEDSIGNAL */
    instr_t I;                      // each unit uses a subset of the fields
    /* verilator lint_on UNUSEDSIGNAL */
    logic        is_avg;
    logic [13:0] p, n_pix;
    logic [6:0]  oy, ox;
    logic [5:0]  cb;
    logic [2:0]  ky, kx;

    // current tap address
    logic signed [9:0] iy, ix;
    logic              inb;
    always_comb begin
        iy  = 10'(signed'({3'b0, oy}) * signed'({8'b0, I.stride})) + 10'(ky) - 10'(I.pad);
        ix  = 10'(signed'({3'b0, ox}) * signed'({8'b0, I.stride})) + 10'(kx) - 10'(I.pad);
        inb = iy >= 0 && iy < 10'(I.in_h) && ix >= 0 && ix < 10'(I.in_w);
        act_raddr = ACT_AW'(24'(I.in_base)
                    + (24'(unsigned'(iy)) * 24'(I.in_w) + 24'(unsigned'(ix))) * 24'(I.in_cb) + 24'(cb));
    end

    logic        d_valid, d_inb, last_tap;
    logic signed [7:0]  lane_max [N];
    logic signed [15:0] lane_sum [N];
    logic [5:0]  cnt;

    // divider state
    logic [15:0] num  [N];
    logic [6:0]  rem  [N];
    logic [7:0]  quo  [N];
    logic [4:0]  dstep;

    assign last_tap = (ky == I.ksize - 3'd1) && (kx == I.ksize - 3'd1);

    always_comb begin
        act_we    = (state == S_WRITE);
        act_waddr = ACT_AW'(24'(I.out_base) + 24'(p) * 24'(I.in_cb) + 24'(cb));
        for (int l = 0; l < N; l++) begin
            logic signed [8:0] v;
            if (is_avg) v = lane_sum[l] < 0 ? -9'(quo[l]) : 9'(quo[l]);
            else        v = 9'(lane_max[l]);
            if (v < 9'(I.act_min))      act_wdata[8*l +: 8] = I.act_min;
            else if (v > 9'(I.act_max)) act_wdata[8*l +: 8] = I.act_max;
            else                        act_wdata[8*l +: 8] = 8'(v);
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
                lane_sum[l] <= lane_sum[l] + 16'(x);
            end
            cnt <= cnt + 6'd1;
        end

        if (rst) begin
            state <= S_IDLE;
        end else begin
            unique case (state)
                S_IDLE: if (start) begin
                    I <= ins;
                    state <= S_INIT;
                end

                S_INIT: begin
                    is_avg <= (I.op == OP_AVGPOOL);
                    n_pix <= 14'(I.out_h) * 14'(I.out_w);
                    p  <= '0; oy <= '0; ox <= '0; cb <= '0; ky <= '0; kx <= '0;
                    cnt <= '0;
                    for (int l = 0; l < N; l++) begin
                        lane_max[l] <= -8'sd128;
                        lane_sum[l] <= '0;
                    end
                    state <= S_TAPS;
                end

                S_TAPS: begin
                    d_valid <= 1'b1;
                    d_inb   <= inb;
                    if (last_tap) begin
                        state <= S_TAIL;
                    end else if (kx == I.ksize - 3'd1) begin
                        kx <= '0;
                        ky <= ky + 3'd1;
                    end else begin
                        kx <= kx + 3'd1;
                    end
                end

                S_TAIL: begin
                    // last word is reduced this cycle
                    state <= S_DIV;
                    dstep <= '0;
                end

                S_DIV: begin
                    if (!is_avg) begin
                        state <= S_WRITE;
                    end else if (dstep == 5'd0) begin
                        // dividend |s| + n/2, quotient bits from the MSB down
                        for (int l = 0; l < N; l++) begin
                            num[l] <= 16'(lane_sum[l] < 0 ? -lane_sum[l] : lane_sum[l]) + 16'(cnt >> 1);
                            rem[l] <= '0;
                            quo[l] <= '0;
                        end
                        dstep <= 5'd1;
                    end else begin
                        for (int l = 0; l < N; l++) begin
                            logic [7:0] r;
                            r = {rem[l], num[l][15]};
                            if (r >= 8'(cnt)) begin
                                rem[l] <= 7'(r - 8'(cnt));
                                quo[l] <= {quo[l][6:0], 1'b1};
                            end else begin
                                rem[l] <= 7'(r);
                                quo[l] <= {quo[l][6:0], 1'b0};
                            end
                            num[l] <= num[l] << 1;
                        end
                        dstep <= dstep + 5'd1;
                        if (dstep == 5'd16) state <= S_WRITE;
                    end
                end

                S_WRITE: state <= S_NEXT;

                S_NEXT: begin
                    ky <= '0; kx <= '0; cnt <= '0;
                    for (int l = 0; l < N; l++) begin
                        lane_max[l] <= -8'sd128;
                        lane_sum[l] <= '0;
                    end
                    if (cb == I.in_cb - 6'd1) begin
                        cb <= '0;
                        if (p == n_pix - 14'd1) begin
                            state <= S_DONE;
                        end else begin
                            p  <= p + 14'd1;
                            ox <= (ox == I.out_w - 7'd1) ? 7'd0 : ox + 7'd1;
                            oy <= (ox == I.out_w - 7'd1) ? oy + 7'd1 : oy;
                            state <= S_TAPS;
                        end
                    end else begin
                        cb <= cb + 6'd1;
                        state <= S_TAPS;
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
