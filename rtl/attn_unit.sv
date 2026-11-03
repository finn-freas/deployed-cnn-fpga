`default_nettype none

module attn_unit (
    input  wire logic         clk,
    input  wire logic         rst,
    input  wire logic         start,
    input  wire logic [7:0]   qbase,
    input  wire logic [7:0]   obase,
    input  wire logic [19:0]  kaddr,
    input  wire logic [19:0]  vaddr,
    input  wire logic [9:0]   npos,
    input  wire logic [1:0]   nkv,
    input  wire logic [3:0]   seq_sh,
    input  wire logic [5:0]   t_shift,
    input  wire logic [3:0]   v_shift,
    output      logic         done,

    output      logic         sr_start,
    output      logic [19:0]  sr_addr,
    output      logic [19:0]  sr_len,
    input  wire logic         s_valid,
    input  wire logic [63:0]  s_data,
    output      logic         s_ready,

    output      logic [7:0]   vm_raddr,
    input  wire logic [255:0] vm_rdata,
    output      logic         vm_we,
    output      logic [7:0]   vm_waddr,
    output      logic [15:0]  vm_wmask,
    output      logic [15:0]  vm_wdata,

    output      logic [7:0]   ex_addr,
    input  wire logic [15:0]  ex_data
);
    typedef enum logic [4:0] {
        S_IDLE, S_QA, S_QW, S_QL, S_P1W, S_P1, S_FIN, S_PE0, S_PE1, S_PE2, S_PE3, S_PE4,
        S_V0, S_V1, S_P2, S_P2END, S_DIV0, S_DIV1, S_DIV, S_DIVW, S_DONE
    } state_e;
    state_e st;

    logic [7:0]  qb, ob;
    logic [19:0] va;
    logic [9:0]  np;
    logic [1:0]  nk;
    logic [3:0]  ssh;
    logic [5:0]  tsh;
    logic [3:0]  vsh;
    logic signed [39:0] rnd;
    wire  [4:0]  kvw = {nk, 3'b000};

    logic [9:0]  p;
    logic [3:0]  w;
    logic [1:0]  ph;
    logic        kvh;
    logic [1:0]  hh;
    logic [5:0]  qe;

    logic [63:0] qmem [64];
    logic [63:0] q_rd;
    wire         p1_adv = (st == S_P1) && s_valid;
    wire  [3:0]  w_nx   = (w + 4'd1 == 4'(kvw)) ? 4'd0 : w + 4'd1;
    wire  [5:0]  q_ra   = !p1_adv ? {w, ph} : (ph == 2'd3) ? {w_nx, 2'd0} : {w, ph + 2'd1};
    always_ff @(posedge clk) begin
        if (st == S_QL) qmem[qe] <= vm_rdata[64 * w[1:0] +: 64];
        q_rd <= qmem[q_ra];
    end

    logic [71:0] sram [1024];
    logic [9:0]  s_ra, s_wa;
    logic        s_we;
    logic [71:0] s_wd, s_rd;
    wire         p2_adv = (st == S_P2) && s_valid && ph == 2'd3;
    wire  [9:0]  p_nx   = p + 10'd1;
    wire  [9:0]  s_rx   = (st == S_P2 || st == S_V1) ? {w[3], p2_adv ? p_nx[8:0] : p[8:0]} : s_ra;
    always_ff @(posedge clk) begin
        if (s_we) sram[s_wa] <= s_wd;
        s_rd <= sram[s_rx];
    end

    logic signed [17:0] mx  [8];
    logic [24:0]        tot [8];

    logic signed [16:0] pa [4];
    logic [63:0] pd;
    logic        pw0, pv0;
    logic signed [24:0] pl [4], ph_ [4];
    logic        pw;
    logic        pv;
    always_ff @(posedge clk) begin
        for (int i = 0; i < 4; i++) begin
            pa[i] <= (st == S_P2) ? signed'({1'b0, s_rd[18 * ph +: 16]}) : 17'(signed'(q_rd[16 * i +: 16]));
            pl[i]  <= pa[i] * signed'(pd[8 * i +: 8]);
            ph_[i] <= pa[i] * signed'(pd[32 + 8 * i +: 8]);
        end
        pd   <= s_data;
        pw0  <= w[0];
        pv0  <= (st == S_P1 || st == S_P2) && s_valid;
        pw   <= pw0;
        pv   <= pv0;
    end
    logic signed [24:0] px [4];
    always_comb for (int i = 0; i < 4; i++) px[i] = pw ? ph_[i] : pl[i];
    wire signed [33:0] dot = 34'(px[0]) + 34'(px[1]) + 34'(px[2]) + 34'(px[3]);
    logic signed [39:0] sacc [4];
    logic signed [41:0] acc  [16];

    logic [2:0]         fin_n;
    logic               f1_v, f2_v;
    logic [1:0]         f1_i, f2_i, f3_i;
    logic signed [39:0] f1, f2;
    logic [9:0]         fin_a;
    logic               fin_last;
    logic [53:0]        fin_s;
    wire  signed [17:0] f2_c = (f2 > 40'sd131071) ? 18'sh1FFFF : (f2 < -40'sd131072) ? 18'sh20000 : f2[17:0];

    logic signed [17:0] sc [4];
    logic signed [18:0] dd;
    logic        pe_v;
    logic [1:0]  pe_hh;
    logic [10:0] pe_ip;
    logic [17:0] ev_w [3];
    assign dd      = 19'(mx[{kvh, hh}]) - 19'(sc[hh]);
    assign ex_addr = dd[7:0];

    logic [3:0]  dl;
    logic signed [41:0] num;
    logic [42:0] mag;
    logic [42:0] rem, dsh;
    logic [4:0]  bitn;
    logic [16:0] quo;
    wire  [43:0] rem_d = {1'b0, rem} - {1'b0, dsh};

    assign s_ready = (((st == S_P1) && w[0]) || (st == S_P2)) && ph == 2'd3 && s_valid;

    always_ff @(posedge clk) begin
        done     <= 1'b0;
        sr_start <= 1'b0;
        vm_we    <= 1'b0;
        s_we     <= 1'b0;
        pe_v     <= 1'b0;
        f1_v     <= 1'b0;
        f2_v     <= 1'b0;

        if (pv && st != S_P2 && st != S_P2END) begin
            for (int i = 0; i < 3; i++) sacc[i] <= sacc[i + 1];
            sacc[3] <= sacc[0] + 40'(dot);
        end
        if (pv && (st == S_P2 || st == S_P2END)) begin
            for (int l = 0; l < 12; l++) acc[l] <= acc[l + 4];
            for (int i = 0; i < 4; i++) acc[12 + i] <= acc[i] + 42'(px[i]);
        end

        if (f1_v) begin
            f2   <= f1 >>> tsh;
            f2_i <= f1_i;
            f2_v <= 1'b1;
        end
        if (f2_v) begin
            if (f2_c > mx[{fin_a[9], f2_i}]) mx[{fin_a[9], f2_i}] <= f2_c;
            fin_s[18 * f2_i +: 18] <= f2_c;
            if (f2_i == 2'd3) begin
                s_we <= 1'b1;
                s_wa <= fin_a;
                s_wd <= {f2_c, fin_s[53:36], fin_s[35:18], fin_s[17:0]};
            end
        end
        if (pe_v) begin
            logic [17:0] ev1;
            ev1 = (pe_ip >= 11'd16) ? 18'd0 : 18'(ex_data >> pe_ip[3:0]);
            tot[{kvh, pe_hh}] <= tot[{kvh, pe_hh}] + 25'(ev1);
            if (pe_hh != 2'd3) ev_w[pe_hh] <= ev1;
            else begin
                s_we <= 1'b1;
                s_wa <= {kvh, p[8:0]};
                s_wd <= {ev1, ev_w[2], ev_w[1], ev_w[0]};
            end
        end

        if (rst) begin
            st <= S_IDLE;
        end else begin
            unique case (st)
                S_IDLE: if (start) begin
                    qb  <= qbase;
                    ob  <= obase;
                    va  <= vaddr;
                    np  <= npos;
                    nk  <= nkv;
                    ssh <= seq_sh;
                    tsh <= t_shift;
                    vsh <= v_shift;
                    rnd <= 40'sd1 <<< (t_shift - 6'd1);
                    qe  <= '0;
                    for (int h = 0; h < 8; h++) begin
                        mx[h]  <= 18'sh20000;
                        tot[h] <= '0;
                    end
                    for (int i = 0; i < 4; i++) sacc[i] <= '0;
                    sr_start <= 1'b1;
                    sr_addr  <= kaddr;
                    sr_len   <= (nkv == 2'd2) ? 20'(npos) << 3 : 20'(npos) << 2;
                    st <= S_QA;
                end

                S_QA: begin
                    w        <= qe[5:2];
                    vm_raddr <= qb + 8'({qe[5], qe[1:0], 1'b0}) + 8'(qe[4]);
                    st       <= S_QW;
                end
                S_QW: st <= S_QL;
                S_QL: begin
                    qe <= qe + 6'd1;
                    if (qe == ((nk == 2'd2) ? 6'd63 : 6'd31)) begin
                        p  <= '0;
                        w  <= '0;
                        ph <= '0;
                        st <= S_P1W;
                    end else begin
                        st <= S_QA;
                    end
                end

                S_P1W: st <= S_P1;

                S_P1: if (s_valid) begin
                    ph <= ph + 2'd1;
                    if (ph == 2'd3) begin
                        w <= w_nx;
                        if (w_nx == 4'd0) p <= (p + 10'd1 == np) ? 10'd0 : p + 10'd1;
                        if (w[2:0] == 3'd7) begin
                            fin_a    <= {w[3], p[8:0]};
                            fin_last <= (w_nx == 4'd0) && (p + 10'd1 == np);
                            fin_n    <= '0;
                            st       <= S_FIN;
                        end
                    end
                end
                S_FIN: begin
                    if (!pv && !pv0 && fin_n != 3'd4) begin
                        f1   <= sacc[0] + rnd;
                        f1_i <= fin_n[1:0];
                        f1_v <= 1'b1;
                        for (int i = 0; i < 3; i++) sacc[i] <= sacc[i + 1];
                        sacc[3] <= '0;
                        fin_n <= fin_n + 3'd1;
                    end
                    if (fin_n == 3'd4 && !f1_v && !f2_v) begin
                        kvh <= 1'b0;
                        st  <= fin_last ? S_PE0 : S_P1;
                    end
                end

                S_PE0: begin
                    s_ra <= {kvh, p[8:0]};
                    st   <= S_PE1;
                end
                S_PE1: st <= S_PE2;
                S_PE2: begin
                    for (int i = 0; i < 4; i++) sc[i] <= s_rd[18*i +: 18];
                    hh <= '0;
                    st <= S_PE3;
                end
                S_PE3: begin
                    pe_v  <= 1'b1;
                    pe_hh <= hh;
                    pe_ip <= 11'(dd >>> 8);
                    hh    <= hh + 2'd1;
                    if (hh == 2'd3) st <= S_PE4;
                end
                S_PE4: begin
                    if (p + 10'd1 == np) begin
                        p <= '0;
                        if (kvh || nk == 2'd1) begin
                            kvh <= 1'b0;
                            w   <= '0;
                            st  <= S_V0;
                        end else begin
                            kvh <= 1'b1;
                            st  <= S_PE0;
                        end
                    end else begin
                        p  <= p + 10'd1;
                        st <= S_PE0;
                    end
                end

                S_V0: begin
                    sr_start <= 1'b1;
                    sr_addr  <= va + (20'(w[3:1]) << ssh);
                    sr_len   <= 20'(np);
                    p        <= '0;
                    ph       <= '0;
                    for (int l = 0; l < 16; l++) acc[l] <= '0;
                    st <= S_V1;
                end
                S_V1: st <= S_P2;
                S_P2: if (s_valid) begin
                    ph <= ph + 2'd1;
                    if (ph == 2'd3) begin
                        if (p + 10'd1 == np) st <= S_P2END;
                        else                 p <= p + 10'd1;
                    end
                end
                S_P2END: if (!pv0) begin
                    dl <= '0;
                    st <= S_DIV0;
                end

                S_DIV0: begin
                    logic [24:0] t;
                    t    = tot[{w[3], dl[3:2]}];
                    num  <= acc[0];
                    mag  <= (acc[0] < 0) ? 43'(-acc[0]) : 43'(acc[0]);
                    dsh  <= 43'(t) << 16;
                    bitn <= 5'd16;
                    quo  <= '0;
                    for (int l = 0; l < 15; l++) acc[l] <= acc[l + 1];
                    st   <= S_DIV1;
                end
                S_DIV1: begin
                    rem <= (mag << vsh) + (dsh >> 17);
                    st  <= S_DIV;
                end
                S_DIV: begin
                    if (!rem_d[43]) begin
                        rem       <= rem_d[42:0];
                        quo[bitn] <= 1'b1;
                    end
                    dsh <= dsh >> 1;
                    if (bitn == 0) st <= S_DIVW;
                    else           bitn <= bitn - 5'd1;
                end
                S_DIVW: begin
                    logic [7:0] el;
                    el = {w[3], dl[3:2], w[2:0], dl[1:0]};
                    vm_we    <= 1'b1;
                    vm_waddr <= ob + 8'(el[7:4]);
                    vm_wmask <= 16'(1) << el[3:0];
                    vm_wdata <= (num < 0) ? -16'(quo) : 16'(quo);
                    dl <= dl + 4'd1;
                    if (dl != 4'd15) st <= S_DIV0;
                    else if (w + 4'd1 == 4'(kvw)) st <= S_DONE;
                    else begin
                        w  <= w + 4'd1;
                        st <= S_V0;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                    st   <= S_IDLE;
                end

                default: st <= S_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
