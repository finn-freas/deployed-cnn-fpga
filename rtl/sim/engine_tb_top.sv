`default_nettype none

module engine_tb_top (
    input  wire logic         clk,
    input  wire logic         clk_f,
    input  wire logic         rst,
    output      logic         sd_ready,
    input  wire logic         init,
    input  wire logic         go,
    input  wire logic [11:0]  tok,
    input  wire logic         logits,
    input  wire logic         pos_we,
    input  wire logic [9:0]   pos_d,
    output      logic         busy,
    output      logic         ready,
    output      logic [9:0]   pos,
    input  wire logic         seed_we,
    input  wire logic [31:0]  seed,
    input  wire logic [11:0]  inv_t,
    input  wire logic [15:0]  cut,
    input  wire logic [15:0]  gain,
    input  wire logic         emo_we,
    input  wire logic [6:0]   emo_addr,
    input  wire logic         emo_mask,
    input  wire logic [15:0]  emo_bias,
    output      logic [11:0]  s_tok,
    output      logic [47:0]  s_em,
    output      logic [47:0]  s_total,
    output      logic [23:0]  s_score,
    input  wire logic         best_clr,
    input  wire logic         cand_end,
    input  wire logic         cand_keep,
    output      logic         cmp_busy,
    output      logic [7:0]   best_k,
    output      logic [31:0]  best_s,
    output      logic [7:0]   best_n,
    input  wire logic         step,
    input  wire logic         step_go,
    output      logic         stopped,
    output      logic [63:0]  op_word,
    input  wire logic [7:0]   dbg_row,
    output      logic [255:0] dbg_vm,
    input  wire logic [11:0]  dbg_logit,
    output      logic [15:0]  dbg_l
);
    localparam int CLK_HZ = 54_000_000;
    logic        cal_err, sd_idle, rd_req, rd_ack, rvalid, wr_req, wr_ack;
    logic [19:0] rd_addr, wr_addr;
    logic [15:0] rd_len;
    logic [63:0] rdata, wr_data;
    logic        cke, cs_n, ras_n, cas_n, we_n, dq_oe_c, dq_oe_m;
    logic [1:0]  ba;
    logic [10:0] addr;
    logic [3:0]  dqm;
    logic [31:0] dq_c, dq_m;

    logic        f_ready, f_cal_err, f_idle, f_rd_req, f_rd_ack, f_rvalid, f_wr_req, f_wr_ack;
    logic [15:0] f_diag, f_rd_len, sd_diag;
    logic [19:0] f_rd_addr, f_wr_addr;
    logic [63:0] f_rdata, f_wr_data;
    logic        rst_f;
    reset_sync u_rst_f (.clk(clk_f), .rst_async_in(rst), .rst_out(rst_f));
    sdram_ctrl #(.CLK_HZ(2 * CLK_HZ), .CAS(3)) u_sdram (
        .clk(clk_f), .rst(rst_f), .ready(f_ready), .cal_err(f_cal_err), .diag(f_diag), .idle(f_idle),
        .rd_req(f_rd_req), .rd_addr(f_rd_addr), .rd_len(f_rd_len), .rd_ack(f_rd_ack), .rvalid(f_rvalid),
        .rdata(f_rdata), .wr_req(f_wr_req), .wr_addr(f_wr_addr), .wr_data(f_wr_data), .wr_ack(f_wr_ack),
        .sd_cke(cke), .sd_cs_n(cs_n), .sd_ras_n(ras_n), .sd_cas_n(cas_n), .sd_we_n(we_n), .sd_ba(ba),
        .sd_addr(addr), .sd_dqm(dqm), .sd_dq_o(dq_c), .sd_dq_oe(dq_oe_c), .sd_dq_i(dq_m));
    sdram_cdc u_cdc (
        .clk_s(clk), .rst_s(rst), .ready(sd_ready), .cal_err, .diag(sd_diag), .idle(sd_idle),
        .rd_req, .rd_addr, .rd_len, .rd_ack, .rvalid, .rdata, .wr_req, .wr_addr, .wr_data, .wr_ack,
        .clk_f, .rst_f, .f_ready, .f_cal_err, .f_diag, .f_idle, .f_rd_req, .f_rd_addr, .f_rd_len, .f_rd_ack,
        .f_rvalid, .f_rdata, .f_wr_req, .f_wr_addr, .f_wr_data, .f_wr_ack);
    sdram_model #(.INIT("../build/sim/engine_mem.hex")) u_mem (
        .clk(clk_f), .cke, .cs_n, .ras_n, .cas_n, .we_n, .ba, .addr, .dqm, .dq_i(dq_c), .dq_o(dq_m), .dq_oe(dq_oe_m));

    engine u_eng (.clk, .rst, .init, .go, .tok, .logits, .pos_we, .pos_d, .busy, .ready, .pos,
                  .seed_we, .seed, .inv_t, .cut, .gain, .emo_we, .emo_addr, .emo_mask, .emo_bias(emo_bias),
                  .s_tok, .s_em, .s_total, .s_score, .best_clr, .cand_end, .cand_keep, .cmp_busy, .best_k,
                  .best_s, .best_n, .rd_row(dbg_row), .rd_vm(dbg_vm), .rd_logit(dbg_logit), .rd_l(dbg_l),
                  .step, .step_go, .stopped, .op_word, .rd_req, .rd_addr, .rd_len, .rd_ack, .rvalid, .rdata,
                  .wr_req, .wr_addr, .wr_data, .wr_ack, .sd_idle);
    logic [31:0] pc_busy, pc_mv, pc_mvstall, pc_vec, pc_vecwait, pc_att, pc_attwait;
    always_ff @(posedge clk) begin
        if (go) begin
            pc_busy <= '0; pc_mv <= '0; pc_mvstall <= '0; pc_vec <= '0; pc_vecwait <= '0; pc_att <= '0;
            pc_attwait <= '0;
        end else begin
            if (busy) pc_busy <= pc_busy + 1;
            if (u_eng.un == 3'd3) pc_mv <= pc_mv + 1;
            if (u_eng.un == 3'd3 && u_eng.u_mv.s_ready && !u_eng.s_valid) pc_mvstall <= pc_mvstall + 1;
            if (u_eng.un == 3'd2) pc_vec <= pc_vec + 1;
            if (u_eng.un == 3'd2 && !u_eng.s_valid) pc_vecwait <= pc_vecwait + 1;
            if (u_eng.un == 3'd4) pc_att <= pc_att + 1;
            if (u_eng.un == 3'd4 && !u_eng.s_valid) pc_attwait <= pc_attwait + 1;
        end
    end
endmodule

`default_nettype wire
