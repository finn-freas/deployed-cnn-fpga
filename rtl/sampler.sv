`default_nettype none

module sampler #(
    parameter int VOCAB = 3072
) (
    input  wire logic               clk,
    input  wire logic               rst,
    input  wire logic               seed_we,
    input  wire logic [31:0]        seed,
    input  wire logic               emo_we,
    input  wire logic [6:0]         emo_addr,
    input  wire logic               emo_mask,
    input  wire logic signed [15:0] emo_bias,
    input  wire logic               l_start,
    input  wire logic               l_valid,
    input  wire logic signed [15:0] l_data,
    input  wire logic               go,
    input  wire logic [11:0]        n,
    input  wire logic [11:0]        inv_t,
    input  wire logic [15:0]        cut,
    input  wire logic [15:0]        gain,
    input  wire logic [11:0]        prev,
    input  wire logic [4:0]         f_logit,
    output      logic [7:0]         ex_addr,
    input  wire logic [23:0]        ex_data,
    output      logic [9:0]         gb_addr,
    input  wire logic signed [15:0] gb_data,
    output      logic [11:0]        w_addr,
    input  wire logic signed [15:0] w_data,
    input  wire logic               best_clr,
    input  wire logic               cand_end,
    input  wire logic               cand_keep,
    output      logic               cmp_busy,
    output      logic signed [23:0] score,
    output      logic [7:0]         best_k,
    output      logic signed [31:0] best_s,
    output      logic [7:0]         best_n,
    input  wire logic [11:0]        dbg_addr,
    output      logic signed [15:0] dbg_data,
    output      logic               done,
    output      logic [11:0]        tok,
    output      logic [47:0]        em,
    output      logic [47:0]        total
);
    logic signed [15:0] lram [VOCAB];
    logic [11:0]        wa;
    logic signed [15:0] lmax;

    logic               mask [128];
    logic signed [15:0] bias [128];
    always_ff @(posedge clk) if (emo_we) begin
        mask[emo_addr] <= emo_mask;
        bias[emo_addr] <= emo_bias;
    end
    function automatic logic is_emo(input logic [11:0] t);
        return (t < 12'd128) && mask[t[6:0]];
    endfunction

    logic [31:0] xs;
    function automatic logic [31:0] xs_next(input logic [31:0] x);
        logic [31:0] y;
        y = x ^ (x << 13);
        y = y ^ (y >> 17);
        y = y ^ (y << 5);
        return y;
    endfunction
    wire [31:0] xn = xs_next(xs);

    typedef enum logic [3:0] { S_IDLE, S_A, S_DEC, S_MUL, S_B, S_SC0, S_SC1, S_LG, S_SC2, S_END, S_CMP } state_e;
    state_e st;

    logic [11:0] i;
    logic        v1, v2, v3, v4;
    logic [24:0] dlr;
    logic [24:0] d1r;
    logic [7:0]  ip4;
    logic        m3, m4;
    logic signed [23:0] zr;
    logic signed [16:0] bl;
    logic signed [29:0] bp;
    logic signed [15:0] bg, bg3;
    logic        bok, bok3;
    logic [11:0] bi, bi3;
    logic        okr;
    logic [11:0] ir;
    logic [11:0] i1;
    logic signed [15:0] lg1;
    logic        m1;
    logic signed [15:0] b1;
    logic [7:0]  ip2;
    logic        m2;
    logic signed [17:0] max_e, max_ne;
    logic        emoji;
    logic signed [17:0] floor_q;
    logic signed [23:0] best;
    logic [15:0] g;
    logic [15:0] du, dg;
    logic [63:0] pu, pe;
    logic [63:0] su, se;
    logic [3:0]  mb;
    logic [20:0] tl;
    logic [4:0]  lp;
    logic signed [25:0] acc_s;
    logic [7:0]  acc_n, cand_k;
    logic signed [33:0] ca, cp;
    logic signed [33:0] sa;
    logic [7:0]  na;
    logic        c_keep;

    assign cmp_busy = (st == S_CMP);
    assign w_addr   = tok;

    wire [11:0] l_ra = (st == S_IDLE) ? dbg_addr : (st == S_SC0) ? tok : i;
    logic signed [15:0] l_rd;
    always_ff @(posedge clk) begin
        if (l_start) begin
            wa   <= '0;
            lmax <= 16'sh8000;
        end else if (l_valid) begin
            lram[wa] <= l_data;
            wa       <= wa + 12'd1;
            if (l_data > lmax) lmax <= l_data;
        end
        l_rd <= lram[l_ra];
    end
    assign dbg_data = l_rd;
    assign lg1 = l_rd;

    wire [24:0] dl = 25'(17'(lmax) - 17'(lg1));
    wire [24:0] d1 = ((dlr << 8) + (dlr << 6) + (dlr << 5) + (dlr << 4) + dlr) >> f_logit;
    assign ex_addr = (st == S_LG) ? tl[7:0] : d1r[7:0];
    assign gb_addr = xn[31:22];

    wire signed [16:0] lgb1  = 17'(lg1) + ((emoji && m1) ? 17'(b1) : 17'sd0);
    wire               keep1 = emoji ? m1 : !m1;

    always_ff @(posedge clk) begin
        done <= 1'b0;
        v1   <= 1'b0;
        v2   <= 1'b0;
        v3   <= 1'b0;
        v4   <= 1'b0;
        if (rst) begin
            st     <= S_IDLE;
            xs     <= 32'h9E37_79B9;
            acc_s  <= '0;
            acc_n  <= '0;
            best_n <= '0;
            best_s <= '0;
            best_k <= '0;
            cand_k <= '0;
        end else begin
            if (seed_we) xs <= (seed == 0) ? 32'h9E37_79B9 : seed;

            if ((st == S_A || st == S_B) && i < n) begin
                m1 <= is_emo(i);
                b1 <= (i < 12'd128) ? bias[i[6:0]] : 16'sd0;
                i1 <= i;
                v1 <= 1'b1;
                i  <= i + 12'd1;
                if (st == S_B) xs <= xn;
            end

            unique case (st)
                S_IDLE: if (best_clr || cand_end) begin
                    c_keep <= cand_end && cand_keep && acc_n != 0;
                    if (best_clr) begin
                        best_n <= '0;
                        best_s <= '0;
                        best_k <= '0;
                        cand_k <= '0;
                    end else begin
                        cand_k <= cand_k + 8'd1;
                    end
                    sa <= 34'(acc_s);
                    na <= best_n;
                    ca <= '0;
                    mb <= '0;
                    st <= S_CMP;
                end else if (go) begin
                    i      <= '0;
                    total  <= '0;
                    em     <= '0;
                    max_e  <= 18'sh20000;
                    max_ne <= 18'sh20000;
                    g      <= is_emo(prev) ? 16'd256 : gain;
                    st     <= S_A;
                end

                S_A: begin
                    if (v1) begin
                        dlr <= dl;
                        m2  <= m1;
                        v2  <= 1'b1;
                        if (m1) begin
                            if (18'(lg1) + 18'(b1) > max_e) max_e <= 18'(lg1) + 18'(b1);
                        end else if (18'(lg1) > max_ne) begin
                            max_ne <= 18'(lg1);
                        end
                    end
                    if (v2) begin
                        d1r <= d1;
                        m3  <= m2;
                        v3  <= 1'b1;
                    end
                    if (v3) begin
                        ip4 <= ((d1r >> 8) > 25'd31) ? 8'd32 : 8'(d1r >> 8);
                        m4  <= m3;
                        v4  <= 1'b1;
                    end
                    if (v4) begin
                        logic [47:0] ev;
                        ev = (ip4 > 8'd31) ? 48'd0 : (48'(ex_data[15:0]) << 16) >> ip4;
                        total <= total + ev;
                        if (m4) em <= em + ev;
                    end
                    if (i == n && !v1 && !v2 && !v3 && !v4) st <= S_DEC;
                end

                S_DEC: begin
                    du <= xn[31:16];
                    dg <= g;
                    xs <= xn;
                    su <= 64'(total);
                    se <= 64'(em);
                    pu <= '0;
                    pe <= '0;
                    mb <= '0;
                    st <= S_MUL;
                end
                S_MUL: begin
                    logic [63:0] pu_n, pe_n;
                    pu_n = pu + (du[0] ? su : 64'd0);
                    pe_n = pe + (dg[0] ? se : 64'd0);
                    pu <= pu_n;
                    pe <= pe_n;
                    du <= du >> 1;
                    dg <= dg >> 1;
                    su <= su << 1;
                    se <= se << 1;
                    mb <= mb + 4'd1;
                    if (mb == 4'd15) begin
                        logic emo;
                        emo = 72'(pu_n) < (72'(pe_n) << 8);
                        emoji   <= emo;
                        floor_q <= (emo ? max_e : max_ne) - 18'(signed'({1'b0, cut}));
                        i       <= '0;
                        best    <= 24'sh80_0000;
                        st      <= S_B;
                    end
                end

                S_B: begin
                    if (v1) begin
                        bl  <= lgb1;
                        bg  <= gb_data;
                        bok <= keep1 && 18'(lgb1) >= floor_q;
                        bi  <= i1;
                        v2  <= 1'b1;
                    end
                    if (v2) begin
                        bp   <= 30'(bl) * signed'({1'b0, inv_t});
                        bg3  <= bg;
                        bok3 <= bok;
                        bi3  <= bi;
                        v3   <= 1'b1;
                    end
                    if (v3) begin
                        zr  <= 24'(bp >>> 8) + 24'(bg3);
                        okr <= bok3;
                        ir  <= bi3;
                        v4  <= 1'b1;
                    end
                    if (v4 && okr && zr > best) begin
                        best <= zr;
                        tok  <= ir;
                    end
                    if (i == n && !v1 && !v2 && !v3 && !v4) st <= S_SC0;
                end

                S_SC0: st <= S_SC1;
                S_SC1: begin
                    dlr <= dl;
                    tl  <= total[43:23];
                    lp  <= '0;
                    st  <= S_LG;
                end
                S_LG: if (|tl[20:9]) begin
                    d1r <= d1;
                    tl <= tl >> 1;
                    lp <= lp + 5'd1;
                end else begin
                    d1r <= d1;
                    st  <= S_SC2;
                end
                S_SC2: begin
                    logic signed [17:0] sc;
                    sc = 18'(w_data) - 18'(d1r) - 18'({lp, ex_data[23:16]});
                    score <= 24'(sc);
                    acc_s <= acc_s + 26'(sc);
                    acc_n <= acc_n + 8'd1;
                    st    <= S_END;
                end

                S_END: begin
                    done <= 1'b1;
                    st   <= S_IDLE;
                end

                S_CMP: begin
                    logic signed [33:0] ca_n;
                    ca_n = ca + (na[0] ? sa : 34'sd0);
                    ca <= ca_n;
                    na <= na >> 1;
                    sa <= sa <<< 1;
                    mb <= mb + 4'd1;
                    if (mb == 4'd7) begin
                        cp <= ca_n;
                        ca <= '0;
                        sa <= 34'(best_s);
                        na <= acc_n;
                    end
                    if (mb == 4'd15) begin
                        if (c_keep && (best_n == 0 || cp > ca_n)) begin
                            best_s <= 32'(acc_s);
                            best_n <= acc_n;
                            best_k <= cand_k - 8'd1;
                        end
                        acc_s <= '0;
                        acc_n <= '0;
                        st    <= S_IDLE;
                    end
                end

                default: st <= S_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
