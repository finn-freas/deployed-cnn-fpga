`default_nettype none

module vec_unit (
    input  wire logic         clk,
    input  wire logic         rst,
    input  wire logic         start,
    input  wire logic [3:0]   op,
    input  wire logic [7:0]   abase,
    input  wire logic [7:0]   bbase,
    input  wire logic [7:0]   obase,
    input  wire logic [5:0]   nrows,
    input  wire logic [19:0]  addr,
    input  wire logic [19:0]  vaddr,
    input  wire logic [3:0]   seq_sh,
    input  wire logic [7:0]   kv_sh,
    output      logic         done,

    input  wire logic [39:0]  eps,
    input  wire logic [7:0]   h_shift,
    input  wire logic [7:0]   ug_shift,

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
    output      logic [255:0] vm_wdata,

    output      logic [9:0]   rs_addr,
    input  wire logic [23:0]  rs_data,
    output      logic [8:0]   sg_addr,
    input  wire logic [15:0]  sg_data,

    output      logic         wr_req,
    output      logic [19:0]  wr_addr,
    output      logic [63:0]  wr_data,
    input  wire logic         wr_ack,
    input  wire logic         sd_idle
);
    localparam logic [3:0] O_EMBED = 4'd1, O_RMS = 4'd2, O_ROPE = 4'd4, O_KVW = 4'd5, O_ADD = 4'd7,
                           O_SILU = 4'd8, O_GAIN = 4'd9;

    function automatic logic signed [15:0] sat16(input logic signed [47:0] v);
        if (v > 48'sd32767) return 16'sh7FFF;
        if (v < -48'sd32768) return 16'sh8000;
        return v[15:0];
    endfunction
    function automatic logic signed [15:0] rq(input logic signed [47:0] v, input logic [5:0] s);
        logic signed [47:0] r;
        r = (v + (48'sd1 <<< (s - 6'd1))) >>> s;
        return sat16(r);
    endfunction

    typedef enum logic [3:0] {
        S_IDLE, S_PRE, S_LA, S_LA2, S_LAW, S_LBW, S_LS, S_LANE, S_STORE, S_RSQ0, S_RSQ1, S_RSQ2, S_KV, S_DONE
    } state_e;
    state_e st;

    logic [3:0]  o;
    logic [5:0]  r, n;
    logic        pass2;
    logic signed [15:0] A [16];
    logic signed [15:0] B [16];
    logic signed [15:0] Y [16];
    logic [15:0] em;
    logic [5:0]  es;
    logic [2:0]  wi;
    logic [47:0] ss;
    logic [23:0] t;
    logic [5:0]  t_sh;
    logic [7:0]  ab, bb, ob;
    logic [19:0] a0, vg;
    logic [3:0]  ssh;
    logic [7:0]  kvs;
    logic [1:0]  kv_half;

    logic [4:0]  k;
    logic        v1, v1a, v1b, v2, v2b, v3;
    logic [3:0]  k1, k1a, k1b, k2, k2b;
    logic signed [15:0] a1, b1, pa1, pb1, cos1, sin1;
    logic signed [15:0] a1a, b1a, a1b, b1b;
    logic signed [15:0] b2;
    logic signed [47:0] m2, m2b;

    logic signed [15:0] mx_a, my_a;
    logic signed [24:0] mx_b;
    logic signed [15:0] my_b;
    always_comb begin
        mx_a = a1;
        mx_b = 25'(b1);
        my_a = pb1;
        my_b = k1[0] ? cos1 : sin1;
        unique case (o)
            O_EMBED: begin mx_a = b1; mx_b = 25'(signed'({1'b0, em})); end
            O_RMS:   mx_b = pass2 ? 25'(signed'({1'b0, t})) : 25'(a1);
            O_ROPE:  begin mx_a = pa1; mx_b = 25'(k1[0] ? sin1 : cos1); end
            O_SILU:  mx_b = 25'(signed'({1'b0, sg_data}));
            default: ;
        endcase
    end
    logic signed [15:0] mx_ar, my_ar, my_br;
    logic signed [24:0] mx_br;
    logic signed [40:0] mx;
    logic signed [31:0] my;
    wire signed [15:0] s16 = m2[15:0];
    logic signed [31:0] mz;
    logic signed [47:0] y_v, yv3, yrnd;
    logic [5:0]         y_s, ysh;
    logic [3:0]         k3;
    always_comb begin
        y_v = m2b;
        unique case (o)
            O_EMBED: y_s = es;
            O_RMS:   y_s = t_sh;
            O_ROPE:  y_s = 6'd14;
            O_SILU:  begin y_v = 48'(mz); y_s = h_shift[5:0]; end
            O_KVW:   y_s = 6'(kv_half[0] ? kvs[7:4] : kvs[3:0]);
            default: y_s = ug_shift[5:0];
        endcase
    end

    wire        lane_go = (st == S_LANE) && k < 5'd16;
    wire [3:0]  kk = k[3:0];
    wire signed [15:0] ka = A[{kk[3:1], 1'b0}], kb = A[{kk[3:1], 1'b1}];

    wire signed [16:0] sg_ix = 17'(A[kk] >>> 6) + 17'sd256;
    assign sg_addr = (sg_ix < 0) ? 9'd0 : (sg_ix > 17'sd511) ? 9'd511 : sg_ix[8:0];

    always_ff @(posedge clk) begin
        v1  <= lane_go;
        v1a <= v1;
        v1b <= v1a;
        v2  <= v1b;
        v2b <= v2;
        v3  <= v2b;
        ysh  <= y_s;
        yrnd <= 48'sd1 <<< (y_s - 6'd1);
        if (lane_go) begin
            k1 <= kk;
            a1 <= A[kk];
            b1 <= B[kk];
            pa1  <= ka;
            pb1  <= kb;
            cos1 <= B[{kk[3:1], 1'b0}];
            sin1 <= B[{kk[3:1], 1'b1}];
        end
        begin
            mx_ar <= mx_a;
            mx_br <= mx_b;
            my_ar <= my_a;
            my_br <= my_b;
            k1a   <= k1;
            a1a   <= a1;
            b1a   <= b1;
        end
        begin
            mx  <= mx_ar * mx_br;
            my  <= my_ar * my_br;
            k1b <= k1a;
            a1b <= a1a;
            b1b <= b1a;
        end
        if (v1b) begin
            k2 <= k1b;
            b2 <= b1b;
            unique case (o)
                O_EMBED, O_RMS, O_GAIN: m2 <= 48'(mx);
                O_ROPE:  m2 <= !k1b[0] ? 48'(mx) - 48'(my)
                                       : 48'(mx) + 48'(my);
                O_ADD:   m2 <= 48'(a1b) + 48'(b1b);
                O_KVW:   m2 <= 48'(a1b);
                O_SILU:  m2 <= 48'(rq(48'(mx), 6'd15));
                default: m2 <= '0;
            endcase
        end
        begin
            mz  <= s16 * b2;
            m2b <= m2;
            k2b <= k2;
        end
        if (v2b) begin
            yv3 <= (o == O_ADD) ? m2b : y_v + yrnd;
            k3  <= k2b;
        end
        if (v3) Y[k3] <= (o == O_ADD) ? sat16(yv3) : sat16(yv3 >>> ysh);
    end

    wire [7:0] a_row = (o == O_KVW && kv_half[0]) ? bb + 8'(r) : ab + 8'(r);
    always_comb for (int j = 0; j < 16; j++) A[j] = vm_rdata[16*j +: 16];

    wire acc_sq = (o == O_RMS) && !pass2 && v2;

    wire needs_b = (o == O_ADD) || (o == O_SILU);
    wire b_from_stream = (o == O_EMBED) || (o == O_GAIN) || (o == O_ROPE);
    wire stores = !(o == O_RMS && !pass2) && o != O_KVW;

    logic [1:0] kv_w;
    function automatic logic [7:0] sat8(input logic signed [15:0] v);
        if (v > 16'sd127)  return 8'h7F;
        if (v < -16'sd128) return 8'h80;
        return v[7:0];
    endfunction
    logic       kv_have;

    always_ff @(posedge clk) begin
        done     <= 1'b0;
        sr_start <= 1'b0;
        vm_we    <= 1'b0;
        if (wr_req && wr_ack) wr_req <= 1'b0;
        if (acc_sq) ss <= ss + 48'(m2);
        if (rst) begin
            st     <= S_IDLE;
            wr_req <= 1'b0;
        end else begin
            unique case (st)
                S_IDLE: if (start) begin
                    o     <= op;
                    n     <= nrows;
                    ab    <= abase;
                    bb    <= bbase;
                    ob    <= obase;
                    a0    <= addr;
                    vg    <= vaddr;
                    ssh   <= seq_sh;
                    kvs   <= kv_sh;
                    r     <= '0;
                    pass2 <= 1'b0;
                    ss    <= 48'(eps);
                    t_sh  <= 6'd11;
                    wi    <= '0;
                    unique case (op)
                        O_EMBED: begin
                            sr_start <= 1'b1; sr_addr <= addr; sr_len <= 20'd33;
                            st <= S_PRE;
                        end
                        O_ROPE: begin
                            sr_start <= 1'b1; sr_addr <= addr; sr_len <= 20'd8;
                            st <= S_LA;
                        end
                        O_GAIN: begin
                            sr_start <= 1'b1; sr_addr <= addr; sr_len <= 20'(nrows) << 2;
                            st <= S_LA;
                        end
                        O_KVW: begin
                            kv_half <= '0;
                            kv_w    <= '0;
                            st      <= S_LA;
                        end
                        default: st <= S_LA;
                    endcase
                end

                S_PRE: if (s_valid) begin
                    em <= s_data[15:0];
                    es <= s_data[21:16];
                    st <= S_LA;
                end

                S_LA: begin
                    vm_raddr <= needs_b ? bb + 8'(r) : a_row;
                    st       <= S_LA2;
                end
                S_LA2: begin
                    vm_raddr <= a_row;
                    st       <= S_LAW;
                end
                S_LAW: begin
                    if (needs_b) for (int j = 0; j < 16; j++) B[j] <= vm_rdata[16*j +: 16];
                    st <= S_LBW;
                end
                S_LBW: begin
                    wi <= '0;
                    k  <= '0;
                    st <= b_from_stream ? S_LS : S_LANE;
                end
                S_LS: if (s_valid) begin
                    if (o == O_EMBED) begin
                        for (int j = 0; j < 8; j++) B[{wi[0], 3'(j)}] <= 16'(signed'(s_data[8*j +: 8]));
                        wi <= wi + 3'd1;
                        if (wi[0]) begin k <= '0; st <= S_LANE; end
                    end else begin
                        for (int j = 0; j < 4; j++) B[{wi[1:0], 2'(j)}] <= s_data[16*j +: 16];
                        wi <= wi + 3'd1;
                        if (wi[1:0] == 2'd3) begin k <= '0; st <= S_LANE; end
                    end
                end

                S_LANE: begin
                    if (k < 5'd16) k <= k + 5'd1;
                    else if (!v1 && !v1a && !v1b && !v2 && !v2b && !v3) st <= S_STORE;
                end

                S_STORE: begin
                    if (stores) begin
                        vm_we    <= 1'b1;
                        vm_waddr <= ob + 8'(r);
                        for (int j = 0; j < 16; j++) vm_wdata[16*j +: 16] <= Y[j];
                    end
                    if (o == O_KVW) st <= S_KV;
                    else if (r + 6'd1 == n) begin
                        r <= '0;
                        if (o == O_RMS && !pass2) st <= S_RSQ0;
                        else                      st <= S_DONE;
                    end else begin
                        r  <= r + 6'd1;
                        st <= S_LA;
                        if (o == O_ROPE && r[0]) begin
                            sr_start <= 1'b1; sr_addr <= a0; sr_len <= 20'd8;
                        end
                    end
                end

                S_RSQ0: begin
                    if (ss >= 48'd1024) begin
                        ss   <= ss >> 2;
                        t_sh <= t_sh + 6'd1;
                    end else begin
                        rs_addr <= 10'(ss - 48'd256);
                        st      <= S_RSQ1;
                    end
                end
                S_RSQ1: st <= S_RSQ2;
                S_RSQ2: begin
                    t     <= rs_data;
                    pass2 <= 1'b1;
                    st    <= S_LA;
                end

                S_KV: if (!wr_req && sd_idle) begin
                    wr_req  <= 1'b1;
                    wr_addr <= kv_half[0] ? vg + (20'({r[1:0], kv_w[0]}) << ssh) : a0 + 20'({r[1:0], kv_w[0]});
                    for (int j = 0; j < 8; j++) wr_data[8*j +: 8] <= sat8(Y[{kv_w[0], 3'(j)}]);
                    kv_w <= {1'b0, ~kv_w[0]};
                    if (kv_w[0]) begin
                        if (r + 6'd1 == n) begin
                            r <= '0;
                            if (kv_half[0]) st <= S_DONE;
                            else begin kv_half <= 2'd1; st <= S_LA; end
                        end else begin
                            r  <= r + 6'd1;
                            st <= S_LA;
                        end
                    end
                end

                S_DONE: if (!wr_req && sd_idle) begin
                    done <= 1'b1;
                    st   <= S_IDLE;
                end

                default: st <= S_IDLE;
            endcase
        end
    end

    assign s_ready = (st == S_PRE) || (st == S_LS);
endmodule

`default_nettype wire
