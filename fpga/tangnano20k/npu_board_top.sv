`default_nettype none

module npu_board_top (
    input  wire logic       clk_27,
    input  wire logic       btn_s1,
    input  wire logic       btn_s2,
    output      logic [5:0] led_n,
    input  wire logic       uart_rx,
    output      logic       uart_tx,
    output      logic       flash_cs_n,
    output      logic       flash_clk,
    output      logic       flash_mosi,
    input  wire logic       flash_miso,
    output      logic       tmds_clk_p,
    output      logic       tmds_clk_n,
    output      logic [2:0] tmds_d_p,
    output      logic [2:0] tmds_d_n,
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
    localparam int CLK_HZ = 54_000_000;

    logic clk, clk_pix, clk_tmds, lock_s, lock_p, rst, rst_pix;
    gw_pll_sys u_pll (.clk_27, .clk_sys(clk), .locked(lock_s));
    gw_clk_gen u_clk (.clk_27, .clk_tmds, .clk_pix, .locked(lock_p));
    reset_sync u_rst (.clk, .rst_async_in(!lock_s || btn_s1), .rst_out(rst));
    reset_sync u_rstp (.clk(clk_pix), .rst_async_in(!lock_p || btn_s1), .rst_out(rst_pix));

    logic [1:0]  s2_sync;
    logic [17:0] s2_cnt;
    logic        s2_state, run;
    always_ff @(posedge clk) begin
        s2_sync <= {s2_sync[0], btn_s2};
        run     <= 1'b0;
        if (rst) begin
            s2_cnt   <= '0;
            s2_state <= 1'b0;
        end else if (s2_sync[1] == s2_state) begin
            s2_cnt <= '0;
        end else if (&s2_cnt) begin
            s2_state <= s2_sync[1];
            run      <= s2_sync[1];
        end else begin
            s2_cnt <= s2_cnt + 18'd1;
        end
    end

    logic        booted, boot_err, busy, res_valid;
    logic [7:0]  res_class;
    logic signed [7:0] res_score;
    logic [31:0] res_cycles;
    logic [15:0] res_count;
    logic [11:0] thumb_raddr;
    logic [3:0]  thumb_rdata;

    logic [31:0] sd_dq_o;
    logic        sd_dq_oe;
    assign IO_sdram_dq = sd_dq_oe ? sd_dq_o : 'z;
    gw_sdram_clk u_sdclk (.clk, .sdram_clk(O_sdram_clk));

    npu_system #(.CLK_HZ(CLK_HZ), .EXT_SDRAM(1'b1), .HAS_POOL(1'b0)) u_sys (
        .clk, .rst, .run, .uart_rx, .uart_tx,
        .spi_cs_n(flash_cs_n), .spi_clk(flash_clk), .spi_mosi(flash_mosi), .spi_miso(flash_miso),
        .booted, .boot_err, .busy, .res_valid, .res_class, .res_score, .res_cycles, .res_count,
        .thumb_clk(clk_pix), .thumb_raddr, .thumb_rdata,
        .sd_cke(O_sdram_cke), .sd_cs_n(O_sdram_cs_n), .sd_ras_n(O_sdram_ras_n),
        .sd_cas_n(O_sdram_cas_n), .sd_we_n(O_sdram_wen_n), .sd_ba(O_sdram_ba),
        .sd_addr(O_sdram_addr), .sd_dqm(O_sdram_dqm), .sd_dq_o, .sd_dq_oe, .sd_dq_i(IO_sdram_dq));

    logic        p_booted, p_boot_err, p_busy, p_res_valid;
    logic [7:0]  p_res_class, p_res_score;
    logic [31:0] p_res_cycles;
    logic [15:0] p_res_count;
    status_cdc u_cdc (
        .s_clk(clk), .s_rst(rst), .s_booted(booted), .s_boot_err(boot_err), .s_busy(busy),
        .s_res_valid(res_valid), .s_res_class(res_class), .s_res_score(res_score),
        .s_res_cycles(res_cycles), .s_res_count(res_count),
        .d_clk(clk_pix), .d_rst(rst_pix), .d_booted(p_booted), .d_boot_err(p_boot_err),
        .d_busy(p_busy), .d_res_valid(p_res_valid), .d_res_class(p_res_class),
        .d_res_score(p_res_score), .d_res_cycles(p_res_cycles), .d_res_count(p_res_count));

    logic       de, hsync, vsync;
    logic [7:0] r, g, b;
    npu_display #(.NPU_HZ(CLK_HZ)) u_disp (
        .clk(clk_pix), .rst(rst_pix), .booted(p_booted), .boot_err(p_boot_err), .busy(p_busy),
        .res_valid(p_res_valid), .res_class(p_res_class), .res_score(p_res_score),
        .res_cycles(p_res_cycles), .res_count(p_res_count),
        .thumb_raddr, .thumb_rdata, .de, .hsync, .vsync, .r, .g, .b);

    logic [9:0] tmds [3];
    dvi_tx u_dvi (.clk_pix, .rst(rst_pix), .de, .hsync, .vsync, .r, .g, .b, .tmds);
    gw_dvi_phy u_phy (.clk_pix, .clk_tmds, .rst(rst_pix), .tmds({tmds[2], tmds[1], tmds[0]}),
                      .tmds_clk_p, .tmds_clk_n, .tmds_d_p, .tmds_d_n);

    logic [25:0] hb;
    always_ff @(posedge clk) hb <= hb + 26'd1;

    logic person;
    assign person = res_valid && res_class == 8'd1;
    assign led_n = ~{hb[25], boot_err, booted, busy, res_valid && !person, person};

endmodule

`default_nettype wire
