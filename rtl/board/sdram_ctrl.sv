`default_nettype none

module sdram_ctrl #(
    parameter int CLK_HZ = 54_000_000,
    parameter int CAS    = 2
) (
    input  wire logic        clk,
    input  wire logic        rst,
    output      logic        ready,
    output      logic        cal_err,
    output      logic [1:0]  cal_rl,
    output      logic        idle,
    input  wire logic        rd_req,
    input  wire logic [16:0] rd_addr,
    input  wire logic [15:0] rd_len,
    output      logic        rd_ack,
    output      logic        rvalid,
    output      logic [63:0] rdata,
    input  wire logic        wr_req,
    input  wire logic [16:0] wr_addr,
    input  wire logic [63:0] wr_data,
    output      logic        wr_ack,
    output      logic        sd_cke,
    output      logic        sd_cs_n,
    output      logic        sd_ras_n,
    output      logic        sd_cas_n,
    output      logic        sd_we_n,
    output      logic [1:0]  sd_ba,
    output      logic [10:0] sd_addr,
    output      logic [3:0]  sd_dqm,
    output      logic [31:0] sd_dq_o,
    output      logic        sd_dq_oe,
    input  wire logic [31:0] sd_dq_i
);

    function automatic int cyc(input int ns);
        return (int'((64'(ns) * 64'(CLK_HZ) + 64'd999_999_999) / 64'd1_000_000_000));
    endfunction
    localparam int T_RP   = (cyc(15) < 1) ? 1 : cyc(15);
    localparam int T_RCD  = (cyc(15) < 1) ? 1 : cyc(15);
    localparam int T_RC   = cyc(60) + 1;
    localparam int T_WR   = 2;
    localparam int T_INIT = cyc(200_000);
    localparam int REF_CYC = cyc(7_800);
    localparam int RL_MIN = CAS - 1, RL_N = 4;
    localparam int RL_MAX = RL_MIN + RL_N - 1;

    localparam logic [2:0] C_NOP = 3'b111, C_ACT = 3'b011, C_RD = 3'b101, C_WR = 3'b100,
                           C_PRE = 3'b010, C_REF = 3'b001, C_MRS = 3'b000;
    logic [2:0] cmd;
    assign {sd_ras_n, sd_cas_n, sd_we_n} = cmd;
    assign sd_cke  = 1'b1;
    assign sd_cs_n = 1'b0;

    typedef enum logic [3:0] {
        I_WAIT, I_PRE, I_REF, I_MRS, CAL_WR, CAL_RD, CAL_CHK, IDLE, RD_RUN, RD_DRAIN, WR_ACT,
        WR_LO, WR_HI, WAIT, REF
    } state_e;
    state_e st, after;
    logic [15:0] wt;
    logic [3:0]  n_ref;
    logic [15:0] ref_cnt;
    logic        ref_due;

    logic [14:0] b_next;
    logic [15:0] bursts_left;
    logic [2:0]  ph;
    logic [15:0] skip;
    logic [15:0] words_left;
    logic [RL_MAX+8:0] rd_mark;
    logic [1:0]  rl_sel;
    logic [2:0]  beat;
    logic        beat_v;
    logic [31:0] dq_q;
    logic [6:0]  wcol;
    logic [31:0] lo;

    always_comb begin
        beat_v = 1'b0;
        for (int k = 0; k < 8; k++)
            if (rd_mark[RL_MIN + int'(rl_sel) + k]) beat_v = 1'b1;
    end

    always_ff @(posedge clk) begin
        rd_mark <= {rd_mark[RL_MAX+7:0], cmd == C_RD};
        dq_q    <= sd_dq_i;
    end

    function automatic logic [31:0] pat(input logic [2:0] i);
        return 32'hA5C3_0F00 ^ {i, 5'b0, ~i, 5'b0, i, 13'h1F00};
    endfunction
    logic [2:0] cal_i;
    logic [RL_N-1:0] cal_ok;

    assign rvalid = (st == RD_RUN || st == RD_DRAIN) && beat_v && beat[0] && skip == 0
                    && words_left != 0;
    assign rdata  = {dq_q, lo};

    assign cal_rl = rl_sel;
    assign idle   = (st == IDLE);

    assign rd_ack = (st == IDLE) && ready && rd_req && !ref_due && !wr_req && rd_len != 0;
    assign wr_ack = (st == IDLE) && ready && wr_req && !ref_due;

    always_ff @(posedge clk) begin
        cmd      <= C_NOP;
        sd_dq_oe <= 1'b0;
        sd_dqm   <= 4'b0000;
        if (ref_cnt != 0) ref_cnt <= ref_cnt - 16'd1;
        else              ref_due <= 1'b1;

        if (st == RD_RUN || st == RD_DRAIN || st == CAL_RD || st == CAL_CHK) begin
            if (beat_v) begin
                beat <= beat + 3'd1;
                if (!beat[0]) begin
                    lo <= dq_q;
                end else if (st == RD_RUN || st == RD_DRAIN) begin
                    if (skip != 0) begin
                        skip <= skip - 16'd1;
                    end else if (words_left != 0) begin
                        words_left <= words_left - 16'd1;
                    end
                end
            end
        end

        if (rst) begin
            st      <= I_WAIT;
            wt      <= 16'(T_INIT);
            ready   <= 1'b0;
            cal_err <= 1'b0;
            ref_cnt <= 16'(REF_CYC);
            ref_due <= 1'b0;
            rl_sel  <= '0;
            beat    <= '0;
        end else begin
            unique case (st)
                I_WAIT: if (wt != 0) wt <= wt - 16'd1; else st <= I_PRE;

                I_PRE: begin
                    cmd        <= C_PRE;
                    sd_addr    <= 11'b100_0000_0000;
                    n_ref      <= 4'd8;
                    wt         <= 16'(T_RP);
                    after      <= I_REF;
                    st         <= WAIT;
                end

                I_REF: begin
                    cmd   <= C_REF;
                    n_ref <= n_ref - 4'd1;
                    wt    <= 16'(T_RC);
                    after <= (n_ref == 4'd1) ? I_MRS : I_REF;
                    st    <= WAIT;
                end

                I_MRS: begin
                    cmd     <= C_MRS;
                    sd_ba   <= 2'b00;
                    sd_addr <= {1'b0, 1'b1, 2'b00, 3'(CAS), 1'b0, 3'b011};
                    wt      <= 16'd3;
                    cal_i   <= '0;
                    after   <= CAL_WR;
                    st      <= WAIT;
                end

                CAL_WR: begin
                    cmd     <= C_ACT;
                    sd_ba   <= 2'b00;
                    sd_addr <= '0;
                    st      <= WAIT;
                    wt      <= 16'(T_RCD);
                    after   <= WR_LO;
                end

                CAL_RD: begin
                    cmd     <= C_ACT;
                    sd_ba   <= 2'b00;
                    sd_addr <= '0;
                    rl_sel  <= '0;
                    wt      <= 16'(T_RCD);
                    after   <= CAL_CHK;
                    st      <= WAIT;
                    cal_ok  <= '1;
                end

                CAL_CHK: begin
                    if (wt == 16'hFFFF) begin
                        cmd     <= C_RD;
                        sd_addr <= 11'b100_0000_0000;
                        wt      <= 16'(RL_MAX + 9);
                        cal_i   <= '0;
                    end else if (wt != 0) begin
                        wt <= wt - 16'd1;
                        for (int c = 0; c < RL_N; c++)
                            for (int k = 0; k < 8; k++)
                                if (rd_mark[RL_MIN + c + k] && dq_q != pat(3'(k))) cal_ok[c] <= 1'b0;
                    end else begin
                        ready   <= 1'b1;
                        cal_err <= (cal_ok == 0);
                        rl_sel  <= cal_ok[0] ? 2'd0 : cal_ok[1] ? 2'd1 : cal_ok[2] ? 2'd2 : 2'd3;
                        st      <= IDLE;
                    end
                end

                IDLE: begin
                    beat <= '0;
                    if (ref_due) begin
                        cmd     <= C_REF;
                        ref_due <= 1'b0;
                        ref_cnt <= 16'(REF_CYC);
                        wt      <= 16'(T_RC);
                        after   <= IDLE;
                        st      <= WAIT;
                    end else if (wr_ack) begin
                        wcol    <= {wr_addr[8:4], wr_addr[1:0]};
                        cmd     <= C_ACT;
                        sd_ba   <= wr_addr[3:2];
                        sd_addr <= 11'(wr_addr[16:9]);
                        wt      <= 16'(T_RCD);
                        after   <= WR_LO;
                        st      <= WAIT;
                    end else if (rd_ack) begin
                        b_next      <= rd_addr[16:2];
                        bursts_left <= 16'((18'(rd_addr[1:0]) + 18'(rd_len) + 18'd3) >> 2);
                        skip        <= 16'(rd_addr[1:0]);
                        words_left  <= rd_len;
                        ph          <= '0;
                        st          <= RD_RUN;
                    end
                end

                WR_LO: begin
                    cmd      <= C_WR;
                    sd_dq_oe <= 1'b1;
                    if (!ready) begin
                        sd_addr <= {3'b000, 5'b0, cal_i};
                        sd_dq_o <= pat(cal_i);
                        cal_i   <= cal_i + 3'd1;
                        if (cal_i == 3'd7) begin
                            sd_addr[10] <= 1'b1;
                            wt    <= 16'(T_WR + T_RP + 1);
                            after <= CAL_RD;
                            st    <= WAIT;
                        end
                    end else begin
                        sd_addr <= {3'b000, wcol, 1'b0};
                        sd_dq_o <= wr_data[31:0];
                        st      <= WR_HI;
                    end
                end

                WR_HI: begin
                    cmd      <= C_WR;
                    sd_dq_oe <= 1'b1;
                    sd_addr  <= {3'b100, wcol, 1'b1};
                    sd_dq_o  <= wr_data[63:32];
                    wt       <= 16'(T_WR + T_RP + 1);
                    after    <= IDLE;
                    st       <= WAIT;
                end

                RD_RUN: begin
                    ph <= ph + 3'd1;
                    if (ph == 0) begin
                        if (bursts_left == 0 || ref_due) begin
                            st <= RD_DRAIN;
                            wt <= 16'(RL_MAX + 8 + T_RP + 2);
                        end else begin
                            cmd     <= C_ACT;
                            sd_ba   <= b_next[1:0];
                            sd_addr <= 11'(b_next[14:7]);
                        end
                    end else if (ph == 3'(T_RCD)) begin
                        cmd         <= C_RD;
                        sd_ba       <= b_next[1:0];
                        sd_addr     <= {3'b100, b_next[6:2], 3'b000};
                        b_next      <= b_next + 15'd1;
                        bursts_left <= bursts_left - 16'd1;
                    end
                end

                RD_DRAIN: begin
                    if (wt != 0) begin
                        wt <= wt - 16'd1;
                    end else if (bursts_left != 0) begin
                        cmd     <= C_REF;
                        ref_due <= 1'b0;
                        ref_cnt <= 16'(REF_CYC);
                        wt      <= 16'(T_RC);
                        after   <= RD_RUN;
                        ph      <= '0;
                        st      <= WAIT;
                    end else begin
                        st <= IDLE;
                    end
                end

                WAIT: begin
                    if (wt > 1) begin
                        wt <= wt - 16'd1;
                    end else begin
                        st <= after;
                        if (after == CAL_CHK) wt <= 16'hFFFF;
                    end
                end

                default: st <= IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
