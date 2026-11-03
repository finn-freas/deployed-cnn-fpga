`default_nettype none

module link_tb_top (
    input  wire logic clk,
    input  wire logic clk_f,
    input  wire logic rst,
    input  wire logic uart_rx,
    output      logic uart_tx,
    output      logic sd_ready
);
    localparam int CLK_HZ = 54_000_000;
    logic        cal_err, sd_idle, rd_req, rd_ack, rvalid, wr_req, wr_ack, busy;
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
    sdram_model u_mem (.clk(clk_f), .cke, .cs_n, .ras_n, .cas_n, .we_n, .ba, .addr, .dqm, .dq_i(dq_c),
                       .dq_o(dq_m), .dq_oe(dq_oe_m));
    host_link #(.CLK_HZ(CLK_HZ), .BAUD(CLK_HZ / 8)) u_link (
        .clk, .rst, .uart_rx, .uart_tx, .busy, .sd_ready, .cal_err, .sd_diag, .sd_idle,
        .rd_req, .rd_addr, .rd_len, .rd_ack, .rvalid, .rdata, .wr_req, .wr_addr, .wr_data, .wr_ack,
        .e_init(), .e_go(), .e_tok(), .e_logits(), .e_pos_we(), .e_pos_d(), .e_best_clr(), .e_cand_end(), .e_cand_keep(),
        .e_cmp_busy(1'b0), .e_s_score(24'd0), .e_best_k(8'd0), .e_best_s(32'd0), .e_best_n(8'd0), .e_busy(1'b0), .e_pos(10'd0), .e_seed_we(),
        .e_seed(), .e_inv_t(), .e_cut(), .e_gain(), .e_emo_we(), .e_emo_addr(), .e_emo_mask(), .e_emo_bias(),
        .e_s_tok(12'd0), .e_s_em(48'd0), .e_s_total(48'd0));
endmodule

`default_nettype wire
