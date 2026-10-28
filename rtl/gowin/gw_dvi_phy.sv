`default_nettype none

module gw_dvi_phy (
    input  wire logic        clk_pix,
    input  wire logic        clk_tmds,
    input  wire logic        rst,
    input  wire logic [29:0] tmds,
    output      logic       tmds_clk_p,
    output      logic       tmds_clk_n,
    output      logic [2:0] tmds_d_p,
    output      logic [2:0] tmds_d_n
);

    logic [2:0] ser;

    for (genvar ch = 0; ch < 3; ch++) begin : g_ser
        OSER10 u_oser (
            .Q    (ser[ch]),
            .D0   (tmds[10*ch+0]), .D1(tmds[10*ch+1]), .D2(tmds[10*ch+2]), .D3(tmds[10*ch+3]),
            .D4   (tmds[10*ch+4]), .D5(tmds[10*ch+5]), .D6(tmds[10*ch+6]), .D7(tmds[10*ch+7]),
            .D8   (tmds[10*ch+8]), .D9(tmds[10*ch+9]),
            .PCLK (clk_pix),
            .FCLK (clk_tmds),
            .RESET(rst)
        );

        TLVDS_OBUF u_obuf (.I(ser[ch]), .O(tmds_d_p[ch]), .OB(tmds_d_n[ch]));
    end

    TLVDS_OBUF u_obuf_clk (.I(clk_pix), .O(tmds_clk_p), .OB(tmds_clk_n));

endmodule

`default_nettype wire
