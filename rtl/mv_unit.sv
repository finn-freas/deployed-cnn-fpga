`default_nettype none

module mv_unit #(
    parameter int MAX_COLS = 704,
    parameter int G        = 32
) (
    input  wire logic         clk,
    input  wire logic         rst,
    input  wire logic         start,
    input  wire logic         tab8,
    input  wire logic [11:0]  rows,
    input  wire logic [11:0]  cols,
    output      logic         busy,

    input  wire logic         s_valid,
    input  wire logic [63:0]  s_data,
    output      logic         s_ready,

    output      logic [5:0]   x_addr,
    input  wire logic [255:0] x_rdata,

    output      logic         o_valid,
    output      logic [15:0]  o_data
);
    localparam int MAX_G  = MAX_COLS / G;
    localparam int GW_MAX = (MAX_G + 7) / 8;
    localparam int WPG    = G / 16;

    typedef enum logic [1:0] { S_IDLE, S_HDR, S_G8, S_W } state_e;
    state_e st;

    logic        t8;
    logic [11:0] rows_left;
    logic [5:0]  n_gw, n_w;
    logic [5:0]  cnt;
    logic [15:0] m_r;
    logic [7:0]  s_r;
    logic [7:0]  g8 [GW_MAX*8];
    logic [4:0]  gidx;
    localparam int WB = $clog2(WPG + 1);
    localparam logic [WB-1:0] WIG_LAST = WB'(WPG - 1);
    logic [WB-1:0] wig;

    logic a_v, b_v;
    logic busy_pipe;

    assign s_ready = (st != S_IDLE);
    assign x_addr  = t8 ? 6'(cnt >> 1) : cnt;
    assign busy    = (st != S_IDLE) || busy_pipe;

    logic        a_gend, a_rend, a_t8, a_hi;
    logic [7:0]  a_g8;
    logic [15:0] a_m;
    logic [7:0]  a_s;
    logic [63:0] a_w;

    always_ff @(posedge clk) begin
        a_v <= 1'b0;
        if (rst) begin
            st <= S_IDLE;
        end else begin
            unique case (st)
                S_IDLE: if (start && rows != 0) begin
                    t8        <= tab8;
                    rows_left <= rows;
                    n_gw      <= 6'((cols / 12'(G) + 12'd7) >> 3);
                    n_w       <= tab8 ? 6'(cols >> 3) : 6'(cols >> 4);
                    st        <= S_HDR;
                end
                S_HDR: if (s_valid) begin
                    m_r  <= t8 ? s_data[39:24] : s_data[15:0];
                    s_r  <= t8 ? s_data[47:40] : s_data[23:16];
                    cnt  <= '0;
                    gidx <= '0;
                    wig  <= '0;
                    st   <= t8 ? S_W : S_G8;
                end
                S_G8: if (s_valid) begin
                    for (int b = 0; b < 8; b++) g8[5'({cnt[1:0], 3'b000}) + 5'(b)] <= s_data[8*b +: 8];
                    cnt <= cnt + 6'd1;
                    if (cnt + 6'd1 == n_gw) begin
                        cnt <= '0;
                        st  <= S_W;
                    end
                end
                S_W: if (s_valid) begin
                    a_v    <= 1'b1;
                    a_w    <= s_data;
                    a_t8   <= t8;
                    a_hi   <= cnt[0];
                    a_g8   <= t8 ? 8'd1 : g8[gidx];
                    a_m    <= m_r;
                    a_s    <= s_r;
                    a_gend <= t8 ? (cnt + 6'd1 == n_w) : (wig == WIG_LAST);
                    a_rend <= (cnt + 6'd1 == n_w);
                    if (wig == WIG_LAST) begin
                        wig  <= '0;
                        gidx <= gidx + 5'd1;
                    end else begin
                        wig <= wig + 1'b1;
                    end
                    cnt <= cnt + 6'd1;
                    if (cnt + 6'd1 == n_w) begin
                        rows_left <= rows_left - 12'd1;
                        st        <= (rows_left == 12'd1) ? S_IDLE : S_HDR;
                    end
                end
                default: st <= S_IDLE;
            endcase
        end
    end

    logic signed [7:0]  xw [16];
    logic signed [15:0] xx [16];
    logic        x_v, x_gend, x_rend;
    logic [7:0]  x_g8;
    logic [15:0] x_m;
    logic [7:0]  x_s;

    always_ff @(posedge clk) begin
        x_v <= a_v && !rst;
        begin
            for (int k = 0; k < 8; k++) begin
                xw[k] <= a_t8 ? signed'(a_w[8*k +: 8]) : 8'(signed'(a_w[4*k +: 4]));
                xx[k] <= (a_t8 && a_hi) ? signed'(x_rdata[16*(k+8) +: 16]) : signed'(x_rdata[16*k +: 16]);
            end
            for (int k = 8; k < 16; k++) begin
                xw[k] <= a_t8 ? 8'sd0 : 8'(signed'(a_w[4*k +: 4]));
                xx[k] <= signed'(x_rdata[16*k +: 16]);
            end
            x_gend <= a_gend;
            x_rend <= a_rend;
            x_g8   <= a_g8;
            x_m    <= a_m;
            x_s    <= a_s;
        end
    end

    logic signed [23:0] pr [16];
    logic        p_v, p_gend, p_rend;
    logic [7:0]  p_g8;
    logic [15:0] p_m;
    logic [7:0]  p_s;

    always_ff @(posedge clk) begin
        p_v <= x_v && !rst;
        begin
            for (int k = 0; k < 16; k++) pr[k] <= xw[k] * xx[k];
            p_gend <= x_gend;
            p_rend <= x_rend;
            p_g8   <= x_g8;
            p_m    <= x_m;
            p_s    <= x_s;
        end
    end

    logic signed [27:0] lane_sum;
    always_comb begin
        lane_sum = '0;
        for (int k = 0; k < 16; k++) lane_sum += 28'(pr[k]);
    end

    logic signed [27:0] b_sum;
    logic        b_gend, b_rend;
    logic [7:0]  b_g8;
    logic [15:0] b_m;
    logic [7:0]  b_s;

    always_ff @(posedge clk) begin
        b_v <= p_v && !rst;
        if (p_v) begin
            b_sum  <= lane_sum;
            b_gend <= p_gend;
            b_rend <= p_rend;
            b_g8   <= p_g8;
            b_m    <= p_m;
            b_s    <= p_s;
        end
    end

    logic signed [33:0] part;
    logic signed [47:0] acc;
    logic               c_v;
    logic signed [47:0] c_acc;
    logic [15:0]        c_m;
    logic [7:0]         c_s;

    wire logic signed [33:0] part_n = part + 34'(b_sum);

    logic               c1_v, c1_rend, c2_v, c2_rend;
    logic signed [33:0] c1_sum;
    logic [7:0]         c1_g8;
    logic signed [47:0] c2_prod;
    logic [15:0]        c1_m, c2_m;
    logic [7:0]         c1_s, c2_s;

    always_ff @(posedge clk) begin
        c1_v <= 1'b0;
        if (rst) begin
            part <= '0;
        end else if (b_v) begin
            if (b_gend) begin
                part    <= '0;
                c1_v    <= 1'b1;
                c1_sum  <= part_n;
                c1_g8   <= b_g8;
                c1_rend <= b_rend;
                c1_m    <= b_m;
                c1_s    <= b_s;
            end else begin
                part <= part_n;
            end
        end
    end

    always_ff @(posedge clk) begin
        c2_v <= c1_v && !rst;
        begin
            c2_prod <= 48'(c1_sum * signed'({1'b0, c1_g8}));
            c2_rend <= c1_rend;
            c2_m    <= c1_m;
            c2_s    <= c1_s;
        end
    end

    wire logic signed [47:0] acc_n = acc + c2_prod;

    always_ff @(posedge clk) begin
        c_v <= 1'b0;
        if (rst) begin
            acc <= '0;
        end else if (c2_v) begin
            if (c2_rend) begin
                acc   <= '0;
                c_v   <= 1'b1;
                c_acc <= acc_n;
                c_m   <= c2_m;
                c_s   <= c2_s;
            end else begin
                acc <= acc_n;
            end
        end
    end

    logic               d0_v;
    logic signed [64:0] d0_prod;
    logic [7:0]         d0_s;
    always_ff @(posedge clk) begin
        d0_v <= c_v && !rst;
        begin
            d0_prod <= 65'(c_acc) * signed'({1'b0, c_m});
            d0_s    <= c_s;
        end
    end
    wire logic signed [64:0] rnd  = 65'sd1 <<< (d0_s - 8'd1);
    wire logic signed [48:0] sh16 = 49'((d0_prod + rnd) >>> 16);
    logic               d_v;
    logic signed [48:0] d_sh16;
    logic [4:0]         d_s;
    wire logic signed [48:0] shd  = d_sh16 >>> (d_s - 5'd16);

    always_ff @(posedge clk) begin
        d_v <= d0_v && !rst;
        if (d0_v) begin
            d_sh16 <= sh16;
            d_s    <= d0_s[4:0];
        end
        o_valid <= d_v && !rst;
        if (d_v) begin
            if (shd > 49'sd32767)       o_data <= 16'h7FFF;
            else if (shd < -49'sd32768) o_data <= 16'h8000;
            else                        o_data <= shd[15:0];
        end
    end

    assign busy_pipe = a_v || x_v || p_v || b_v || c1_v || c2_v || c_v || d0_v || d_v || o_valid;

endmodule

`default_nettype wire
