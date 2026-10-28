`default_nettype none

module colorbars_top (
    input  wire logic       clk_27,
    input  wire logic       btn_s1,
    output      logic [5:0] led_n,
    output      logic       tmds_clk_p,
    output      logic       tmds_clk_n,
    output      logic [2:0] tmds_d_p,
    output      logic [2:0] tmds_d_n
);

    logic clk_tmds, clk_pix, locked, rst;

    gw_clk_gen u_clk (.clk_27, .clk_tmds, .clk_pix, .locked);
    reset_sync u_rst (.clk(clk_pix), .rst_async_in(!locked || btn_s1), .rst_out(rst));

    logic [10:0] x;
    logic [9:0]  y;
    logic        active, hsync, vsync, frame_start;

    video_timing u_vt (.clk(clk_pix), .rst, .x, .y, .active, .hsync, .vsync, .frame_start);

    logic [7:0] r, g, b;
    test_pattern u_tp (.clk(clk_pix), .rst, .frame_start, .x, .y, .r, .g, .b);

    logic [9:0] tmds [3];
    dvi_tx u_dvi (.clk_pix, .rst, .de(active), .hsync, .vsync, .r, .g, .b, .tmds);

    gw_dvi_phy u_phy (.clk_pix, .clk_tmds, .rst, .tmds({tmds[2], tmds[1], tmds[0]}), .tmds_clk_p, .tmds_clk_n, .tmds_d_p, .tmds_d_n);

    logic [23:0] heartbeat;
    always_ff @(posedge clk_pix) heartbeat <= heartbeat + 24'd1;

    assign led_n = ~{4'b0, locked, heartbeat[23]};

endmodule

`default_nettype wire
