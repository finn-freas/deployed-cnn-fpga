// gw_pll_sys: the NPU/SDRAM system clock, clk_27 -> rPLL -> 27 * 2 = 54 MHz
// (VCO 864 MHz). Synthesis only.

`default_nettype none

module gw_pll_sys (
    input  wire logic clk_27,
    output      logic clk_sys,
    output      logic locked
);

    rPLL #(
        .FCLKIN          ("27"),
        .DYN_IDIV_SEL    ("false"),
        .IDIV_SEL        (0),        // /1
        .DYN_FBDIV_SEL   ("false"),
        .FBDIV_SEL       (1),        // x2
        .DYN_ODIV_SEL    ("false"),
        .ODIV_SEL        (16),       // VCO = 864 MHz
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
        .CLKOUT  (clk_sys),
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

endmodule

`default_nettype wire
