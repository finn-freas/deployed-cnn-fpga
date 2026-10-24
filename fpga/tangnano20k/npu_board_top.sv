// npu_board_top: the person detector on the Tang Nano 20K.
//
//   * boots from the SPI flash: copies the weights/params into the in-package SDRAM, loads
//     the program and a demo image, and runs one inference
//   * S1 resets; S2 reloads the demo image and runs again
//   * USB-UART (1 Mbaud): send images, get class/score/cycles back (sw/board/demo.py)
//   * HDMI 640x480: thumbnail of the input, verdict, score, latency, run count
//   * LEDs (active-low): 0 person, 1 no person, 2 busy, 3 booted, 4 flash error, 5 heartbeat
//
// Needs gowin_pack --mspi_as_gpio so the configuration flash pins stay usable as I/O.

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
    // in-package SDRAM
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
    localparam int CLK_HZ = 25_200_000;

    logic clk_tmds, clk, locked, rst;
    gw_clk_gen u_clk (.clk_27, .clk_tmds, .clk_pix(clk), .locked);
    reset_sync u_rst (.clk, .rst_async_in(!locked || btn_s1), .rst_out(rst));

    // S2: synchronise, debounce (~10 ms), one pulse per press
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
        .thumb_raddr, .thumb_rdata,
        .sd_cke(O_sdram_cke), .sd_cs_n(O_sdram_cs_n), .sd_ras_n(O_sdram_ras_n),
        .sd_cas_n(O_sdram_cas_n), .sd_we_n(O_sdram_wen_n), .sd_ba(O_sdram_ba),
        .sd_addr(O_sdram_addr), .sd_dqm(O_sdram_dqm), .sd_dq_o, .sd_dq_oe, .sd_dq_i(IO_sdram_dq));

    logic       de, hsync, vsync;
    logic [7:0] r, g, b;
    npu_display #(.NPU_HZ(CLK_HZ)) u_disp (
        .clk, .rst, .booted, .boot_err, .busy, .res_valid, .res_class, .res_score, .res_cycles,
        .res_count, .thumb_raddr, .thumb_rdata, .de, .hsync, .vsync, .r, .g, .b);

    logic [9:0] tmds [3];
    dvi_tx u_dvi (.clk_pix(clk), .rst, .de, .hsync, .vsync, .r, .g, .b, .tmds);
    gw_dvi_phy u_phy (.clk_pix(clk), .clk_tmds, .rst, .tmds({tmds[2], tmds[1], tmds[0]}),
                      .tmds_clk_p, .tmds_clk_n, .tmds_d_p, .tmds_d_n);

    logic [24:0] hb;
    always_ff @(posedge clk) hb <= hb + 25'd1;

    logic person;
    assign person = res_valid && res_class == 8'd1;
    assign led_n = ~{hb[24], boot_err, booted, busy, res_valid && !person, person};

endmodule

`default_nettype wire
