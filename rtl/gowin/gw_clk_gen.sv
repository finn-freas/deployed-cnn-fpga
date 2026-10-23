// gw_clk_gen: Gowin GW2A clocking for 640x480 DVI.
//   clk_27  -> rPLL -> clk_tmds = 27 * 14 / 3 = 126 MHz (5x pixel, OSER10 FCLK)
//   clk_tmds -> CLKDIV /5 -> clk_pix = 25.2 MHz
// Vendor primitives: synthesis only (not simulated).

`default_nettype none

module gw_clk_gen (
    input  wire logic clk_27,
    output      logic clk_tmds,
    output      logic clk_pix,
    output      logic locked
);

    rPLL #(
        .FCLKIN          ("27"),
        .DYN_IDIV_SEL    ("false"),
        .IDIV_SEL        (2),        // /3
        .DYN_FBDIV_SEL   ("false"),
        .FBDIV_SEL       (13),       // x14
        .DYN_ODIV_SEL    ("false"),
        .ODIV_SEL        (4),        // VCO = 504 MHz
        .PSDA_SEL        ("0000"),
        .DYN_DA_EN       ("true"),
        .DUTYDA_SEL      ("1000"),
        .CLKOUT_FT_DIR   (1'b1),
        .CLKOUTP_FT_DIR  (1'b1),
        .CLKOUT_DLY_STEP (0),
        .CLKOUTP_DLY_STEP(0),
        .CLKFB_SEL       ("internal"),
        .CLKOUT_BYPASS   ("false"),
        .CLKOUTP_BYPASS  ("false"),
        .CLKOUTD_BYPASS  ("false"),
        .DYN_SDIV_SEL    (2),
        .CLKOUTD_SRC     ("CLKOUT"),
        .CLKOUTD3_SRC    ("CLKOUT"),
        .DEVICE          ("GW2AR-18C")
    ) u_pll (
        .CLKOUT  (clk_tmds),
        .LOCK    (locked),
        .CLKOUTP (),
        .CLKOUTD (),
        .CLKOUTD3(),
        .RESET   (1'b0),
        .RESET_P (1'b0),
        .CLKIN   (clk_27),
        .CLKFB   (1'b0),
        .FBDSEL  (6'b0),
        .IDSEL   (6'b0),
        .ODSEL   (6'b0),
        .PSDA    (4'b0),
        .DUTYDA  (4'b0),
        .FDLY    (4'b0)
    );

    CLKDIV #(.DIV_MODE("5")) u_div (
        .CLKOUT(clk_pix),
        .HCLKIN(clk_tmds),
        .RESETN(locked),
        .CALIB (1'b0)
    );

endmodule

`default_nettype wire
