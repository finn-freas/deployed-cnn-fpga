// sdram_ctrl: the GW2AR-18's in-package 64 Mbit SDR SDRAM (32-bit, 4 banks x 2K rows x
// 256 columns) as a 64-bit word store with a streaming read port for the NPU.
//
// Reads use bursts of 8 (4 words) with auto-precharge; consecutive bursts go to
// consecutive banks, so an ACTIVATE for the next burst issues while the current one
// streams and a long read delivers one 64-bit word every 2 cycles. Writes are single
// 64-bit words (two single-location WRITEs). Mapping of word w: burst b = w/4, bank b%4,
// column {b/4 % 32, w%4, half}, row b/128.
//
// The read data latency (command to first beat at our clock edge) depends on the board's
// clock and I/O delays, so it is measured after initialisation: a pattern is written and
// read back, and the first of the candidate latencies that returns it is used. `cal_err`
// reports that none did.
//
// Refresh: one AUTO REFRESH every REF_CYC cycles, between bursts.

`default_nettype none

module sdram_ctrl #(
    parameter int CLK_HZ = 54_000_000,
    parameter int CAS    = 2
) (
    input  wire logic        clk,
    input  wire logic        rst,
    output      logic        ready,         // initialised and calibrated
    output      logic        cal_err,
    // read stream (like the NPU's external-store port)
    input  wire logic        rd_req,
    input  wire logic [16:0] rd_addr,       // 64-bit word address
    input  wire logic [15:0] rd_len,
    output      logic        rd_ack,
    output      logic        rvalid,
    output      logic [63:0] rdata,
    // single-word writes
    input  wire logic        wr_req,
    input  wire logic [16:0] wr_addr,
    input  wire logic [63:0] wr_data,
    output      logic        wr_ack,
    // SDRAM pins (dq split for simulation; the top level makes it bidirectional)
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

    function automatic int cyc(input int ns);         // cycles covering ns
        return (int'((64'(ns) * 64'(CLK_HZ) + 64'd999_999_999) / 64'd1_000_000_000));
    endfunction
    localparam int T_RP   = (cyc(15) < 1) ? 1 : cyc(15);
    localparam int T_RCD  = (cyc(15) < 1) ? 1 : cyc(15);
    localparam int T_RC   = cyc(60) + 1;
    localparam int T_WR   = 2;
    localparam int T_INIT = cyc(200_000);
    localparam int REF_CYC = cyc(7_800);               // 8192 per 64 ms: twice the minimum
    // read latency RL: beat i of a READ set in cycle t is in dq_q in cycle t+2+RL+i
    localparam int RL_MIN = CAS - 1, RL_N = 4;         // candidate latencies
    localparam int RL_MAX = RL_MIN + RL_N - 1;

    // commands {ras_n, cas_n, we_n}
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
    logic [15:0] wt;                        // wait counter
    logic [3:0]  n_ref;
    logic [15:0] ref_cnt;
    logic        ref_due;

    // ---- read stream state
    logic [14:0] b_next;                    // next burst to issue
    logic [15:0] bursts_left;               // bursts still to issue
    logic [2:0]  ph;                        // position in the 8-cycle burst slot
    logic [15:0] skip;                      // leading words to drop
    logic [15:0] words_left;                // words still to deliver
    // data capture: a READ issued in cycle t returns beats 0..7 in cycles t+RL..t+RL+7
    logic [RL_MAX+8:0] rd_mark;             // rd_mark[k]: a READ issued k cycles ago
    logic [1:0]  rl_sel;                    // calibrated latency - RL_MIN
    logic [2:0]  beat;
    logic        beat_v;
    logic [31:0] dq_q;                      // DQ sampled at our clock edge
    logic [31:0] lo;                        // first beat of a word
    /* verilator lint_off UNUSEDSIGNAL */
    logic [16:0] wa;                        // write being performed (bank/row used at ACT)
    /* verilator lint_on UNUSEDSIGNAL */
    logic [63:0] wd;

    // beat valid and data at the selected latency
    always_comb begin
        beat_v = 1'b0;
        for (int k = 0; k < 8; k++)
            if (rd_mark[RL_MIN + int'(rl_sel) + k]) beat_v = 1'b1;
    end

    always_ff @(posedge clk) begin
        rd_mark <= {rd_mark[RL_MAX+7:0], cmd == C_RD};
        dq_q    <= sd_dq_i;
    end

    // ---- calibration pattern
    function automatic logic [31:0] pat(input logic [2:0] i);
        return 32'hA5C3_0F00 ^ {i, 5'b0, ~i, 5'b0, i, 13'h1F00};
    endfunction
    logic [2:0] cal_i;
    logic [RL_N-1:0] cal_ok;

    assign rd_ack = (st == IDLE) && ready && rd_req && !ref_due && !wr_req && rd_len != 0;
    assign wr_ack = (st == IDLE) && ready && wr_req && !ref_due;

    always_ff @(posedge clk) begin
        cmd      <= C_NOP;
        sd_dq_oe <= 1'b0;
        sd_dqm   <= 4'b0000;
        rvalid   <= 1'b0;
        if (ref_cnt != 0) ref_cnt <= ref_cnt - 16'd1;
        else              ref_due <= 1'b1;

        // ---- read data: two beats per word, words outside the request dropped
        if (st == RD_RUN || st == RD_DRAIN || st == CAL_RD || st == CAL_CHK) begin
            if (beat_v) begin
                beat <= beat + 3'd1;
                if (!beat[0]) begin
                    lo <= dq_q;
                end else if (st == RD_RUN || st == RD_DRAIN) begin
                    if (skip != 0) begin
                        skip <= skip - 16'd1;
                    end else if (words_left != 0) begin
                        rvalid     <= 1'b1;
                        rdata      <= {dq_q, lo};
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
                // ---------------------------------------------------------- init
                I_WAIT: if (wt != 0) wt <= wt - 16'd1; else st <= I_PRE;

                I_PRE: begin
                    cmd        <= C_PRE;
                    sd_addr    <= 11'b100_0000_0000;        // all banks
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
                    // single-location writes, CAS latency, sequential bursts of 8
                    cmd     <= C_MRS;
                    sd_ba   <= 2'b00;
                    sd_addr <= {1'b0, 1'b1, 2'b00, 3'(CAS), 1'b0, 3'b011};
                    wt      <= 16'd3;
                    cal_i   <= '0;
                    after   <= CAL_WR;
                    st      <= WAIT;
                end

                // ---------------------------------------------------------- calibration
                // write the pattern to bank 0, row 0, columns 0..7, then read it as a burst
                CAL_WR: begin
                    cmd     <= C_ACT;
                    sd_ba   <= 2'b00;
                    sd_addr <= '0;
                    st      <= WAIT;
                    wt      <= 16'(T_RCD);
                    after   <= WR_LO;                     // reuse the write sequence
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
                    // issued once (wt is re-armed below); then compare every candidate
                    if (wt == 16'hFFFF) begin
                        cmd     <= C_RD;
                        sd_addr <= 11'b100_0000_0000;     // auto precharge, column 0
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

                // ---------------------------------------------------------- idle
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
                        wa      <= wr_addr;
                        wd      <= wr_data;
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

                // ---------------------------------------------------------- writes
                WR_LO: begin
                    cmd      <= C_WR;
                    sd_dq_oe <= 1'b1;
                    if (!ready) begin                       // calibration pattern
                        sd_addr <= {3'b000, 5'b0, cal_i};
                        sd_dq_o <= pat(cal_i);
                        cal_i   <= cal_i + 3'd1;
                        if (cal_i == 3'd7) begin
                            sd_addr[10] <= 1'b1;            // last: auto precharge
                            wt    <= 16'(T_WR + T_RP + 1);
                            after <= CAL_RD;
                            st    <= WAIT;
                        end
                    end else begin
                        sd_addr <= {3'b000, wa[8:4], wa[1:0], 1'b0};
                        sd_dq_o <= wd[31:0];
                        st      <= WR_HI;
                    end
                end

                WR_HI: begin
                    cmd      <= C_WR;
                    sd_dq_oe <= 1'b1;
                    sd_addr  <= {3'b100, wa[8:4], wa[1:0], 1'b1};   // auto precharge
                    sd_dq_o  <= wd[63:32];
                    wt       <= 16'(T_WR + T_RP + 1);
                    after    <= IDLE;
                    st       <= WAIT;
                end

                // ---------------------------------------------------------- read stream
                // slot of 8 cycles per burst: ACT at ph 0, READ (auto precharge) at ph T_RCD
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

                // let the last bursts arrive and precharge, then continue or finish
                RD_DRAIN: begin
                    if (wt != 0) begin
                        wt <= wt - 16'd1;
                    end else if (bursts_left != 0) begin    // paused for a refresh
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
