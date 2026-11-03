`default_nettype none

module roastbot_top #(
    parameter int BAUD = 1_000_000,
    parameter int MHZ  = 54,
    parameter int SD_MHZ = 108
) (
    input  wire logic        clk_27,
    input  wire logic        btn_s1,
    output      logic [5:0]  led_n,
    input  wire logic        uart_rx,
    output      logic        uart_tx,
    output      logic        O_sdram_clk,
    output      logic        O_sdram_cke,
    output      logic        O_sdram_cs_n,
    output      logic        O_sdram_cas_n,
    output      logic        O_sdram_ras_n,
    output      logic        O_sdram_wen_n,
    output      logic [3:0]  O_sdram_dqm,
    output      logic [10:0] O_sdram_addr,
    output      logic [1:0]  O_sdram_ba,
    inout  wire logic [31:0] IO_sdram_dq
);
    localparam int CLK_HZ = MHZ * 1_000_000;

    logic clk, clk_f, clk_p, lock, rst, rst_f;
    gw_pll_2x u_pll (.clk_27, .clk_f, .clk_p, .clk_s(clk), .locked(lock));
    reset_sync u_rst (.clk, .rst_async_in(!lock || btn_s1), .rst_out(rst));
    reset_sync u_rst_f (.clk(clk_f), .rst_async_in(!lock || btn_s1), .rst_out(rst_f));

    logic [31:0] sd_dq_o;
    logic        sd_dq_oe;
    assign IO_sdram_dq = sd_dq_oe ? sd_dq_o : 'z;
    gw_sdram_clk u_sdclk (.clk(clk_p), .sdram_clk(O_sdram_clk));

    logic        f_ready, f_cal_err, f_idle, f_rd_req, f_rd_ack, f_rvalid, f_wr_req, f_wr_ack;
    logic [15:0] f_diag, f_rd_len;
    logic [19:0] f_rd_addr, f_wr_addr;
    logic [63:0] f_rdata, f_wr_data;
    sdram_ctrl #(.CLK_HZ(SD_MHZ * 1_000_000), .CAS(3)) u_sdram (
        .clk(clk_f), .rst(rst_f), .ready(f_ready), .cal_err(f_cal_err), .diag(f_diag), .idle(f_idle),
        .rd_req(f_rd_req), .rd_addr(f_rd_addr), .rd_len(f_rd_len), .rd_ack(f_rd_ack), .rvalid(f_rvalid),
        .rdata(f_rdata), .wr_req(f_wr_req), .wr_addr(f_wr_addr), .wr_data(f_wr_data), .wr_ack(f_wr_ack),
        .sd_cke(O_sdram_cke), .sd_cs_n(O_sdram_cs_n), .sd_ras_n(O_sdram_ras_n),
        .sd_cas_n(O_sdram_cas_n), .sd_we_n(O_sdram_wen_n), .sd_ba(O_sdram_ba),
        .sd_addr(O_sdram_addr), .sd_dqm(O_sdram_dqm), .sd_dq_o, .sd_dq_oe, .sd_dq_i(IO_sdram_dq));

    logic        sd_ready, cal_err, sd_idle, rd_req, rd_ack, rvalid, wr_req, wr_ack;
    logic [15:0] sd_diag;
    logic [19:0] rd_addr, wr_addr;
    logic [15:0] rd_len;
    logic [63:0] rdata, wr_data;
    sdram_cdc u_cdc (
        .clk_s(clk), .rst_s(rst), .ready(sd_ready), .cal_err, .diag(sd_diag), .idle(sd_idle),
        .rd_req, .rd_addr, .rd_len, .rd_ack, .rvalid, .rdata, .wr_req, .wr_addr, .wr_data, .wr_ack,
        .clk_f, .rst_f, .f_ready, .f_cal_err, .f_diag, .f_idle, .f_rd_req, .f_rd_addr, .f_rd_len, .f_rd_ack,
        .f_rvalid, .f_rdata, .f_wr_req, .f_wr_addr, .f_wr_data, .f_wr_ack);

    logic        h_rd_req, h_wr_req, n_rd_req, n_wr_req;
    logic [19:0] h_rd_addr, h_wr_addr, n_rd_addr, n_wr_addr;
    logic [15:0] h_rd_len, n_rd_len;
    logic [63:0] h_wr_data, n_wr_data;
    logic        e_busy;
    assign rd_req  = e_busy ? n_rd_req  : h_rd_req;
    assign rd_addr = e_busy ? n_rd_addr : h_rd_addr;
    assign rd_len  = e_busy ? n_rd_len  : h_rd_len;
    assign wr_req  = e_busy ? n_wr_req  : h_wr_req;
    assign wr_addr = e_busy ? n_wr_addr : h_wr_addr;
    assign wr_data = e_busy ? n_wr_data : h_wr_data;

    logic        busy;
    logic        e_init, e_go, e_logits, e_pos_we, e_ready, e_seed_we, e_emo_we, e_emo_mask;
    logic [9:0]  e_pos_d;
    logic        e_best_clr, e_cand_end, e_cand_keep, e_cmp_busy;
    logic [23:0] e_s_score;
    logic [7:0]  e_best_k, e_best_n;
    logic [31:0] e_best_s;
    logic [11:0] e_tok, e_inv_t, e_s_tok;
    logic [9:0]  e_pos;
    logic [31:0] e_seed;
    logic [15:0] e_cut, e_gain, e_emo_bias;
    logic [6:0]  e_emo_addr;
    logic [47:0] e_s_em, e_s_total;
    host_link #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_link (
        .clk, .rst, .uart_rx, .uart_tx, .busy, .sd_ready, .cal_err, .sd_diag, .sd_idle,
        .rd_req(h_rd_req), .rd_addr(h_rd_addr), .rd_len(h_rd_len), .rd_ack(rd_ack && !e_busy),
        .rvalid(rvalid && !e_busy), .rdata, .wr_req(h_wr_req), .wr_addr(h_wr_addr), .wr_data(h_wr_data),
        .wr_ack(wr_ack && !e_busy),
        .e_init, .e_go, .e_tok, .e_logits, .e_pos_we, .e_pos_d, .e_best_clr, .e_cand_end, .e_cand_keep,
        .e_cmp_busy, .e_s_score, .e_best_k, .e_best_s, .e_best_n, .e_busy, .e_pos, .e_seed_we, .e_seed, .e_inv_t,
        .e_cut, .e_gain, .e_emo_we, .e_emo_addr, .e_emo_mask, .e_emo_bias, .e_s_tok, .e_s_em, .e_s_total);

    logic        e_stopped;
    logic [63:0] e_op;
    logic [255:0] e_vm;
    logic [15:0] e_l;
    engine u_eng (.clk, .rst, .init(e_init), .go(e_go), .tok(e_tok), .logits(e_logits), .pos_we(e_pos_we), .pos_d(e_pos_d),
                  .busy(e_busy), .ready(e_ready), .pos(e_pos), .seed_we(e_seed_we), .seed(e_seed), .inv_t(e_inv_t),
                  .cut(e_cut), .gain(e_gain), .emo_we(e_emo_we), .emo_addr(e_emo_addr), .emo_mask(e_emo_mask),
                  .emo_bias(e_emo_bias), .s_tok(e_s_tok), .s_em(e_s_em), .s_total(e_s_total),
                  .s_score(e_s_score), .best_clr(e_best_clr), .cand_end(e_cand_end), .cand_keep(e_cand_keep),
                  .cmp_busy(e_cmp_busy), .best_k(e_best_k), .best_s(e_best_s), .best_n(e_best_n),
                  .rd_row(8'd0), .rd_vm(e_vm), .rd_logit(12'd0), .rd_l(e_l),
                  .step(1'b0), .step_go(1'b0), .stopped(e_stopped), .op_word(e_op),
                  .rd_req(n_rd_req), .rd_addr(n_rd_addr), .rd_len(n_rd_len), .rd_ack(rd_ack && e_busy),
                  .rvalid(rvalid && e_busy), .rdata, .wr_req(n_wr_req), .wr_addr(n_wr_addr), .wr_data(n_wr_data),
                  .wr_ack(wr_ack && e_busy), .sd_idle);

    logic [24:0] hb;
    logic [21:0] rx_led;
    always_ff @(posedge clk) begin
        hb <= hb + 25'd1;
        if (!uart_rx) rx_led <= '1;
        else if (rx_led != 0) rx_led <= rx_led - 22'd1;
    end

    assign led_n = ~{e_busy, rx_led != 0, busy, cal_err, sd_ready, hb[24]};
endmodule

`default_nettype wire
