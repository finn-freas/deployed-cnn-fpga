// status_cdc: carries npu_system's status into the display's clock domain.
// The result fields are written together once per inference, with res_count incremented,
// so res_count[0] toggles per result; the destination copies the fields (by then long
// stable, the next result is milliseconds away) when the synchronised toggle changes.
// Single-bit flags go through two-flop synchronisers.

`default_nettype none

module status_cdc (
    input  wire logic        s_clk,
    input  wire logic        s_rst,
    input  wire logic        s_booted,
    input  wire logic        s_boot_err,
    input  wire logic        s_busy,
    input  wire logic        s_res_valid,
    input  wire logic [7:0]  s_res_class,
    input  wire logic [7:0]  s_res_score,
    input  wire logic [31:0] s_res_cycles,
    input  wire logic [15:0] s_res_count,
    input  wire logic        d_clk,
    input  wire logic        d_rst,
    output      logic        d_booted,
    output      logic        d_boot_err,
    output      logic        d_busy,
    output      logic        d_res_valid,
    output      logic [7:0]  d_res_class,
    output      logic [7:0]  d_res_score,
    output      logic [31:0] d_res_cycles,
    output      logic [15:0] d_res_count
);
    /* verilator lint_off UNUSEDSIGNAL */
    logic unused_src;
    assign unused_src = s_clk ^ s_rst;  // no source-side logic
    /* verilator lint_on UNUSEDSIGNAL */

    logic [2:0] tog, f_bt, f_be, f_bz;
    always_ff @(posedge d_clk) begin
        tog  <= {tog[1:0], s_res_count[0]};
        f_bt <= {f_bt[1:0], s_booted};
        f_be <= {f_be[1:0], s_boot_err};
        f_bz <= {f_bz[1:0], s_busy};
        if (d_rst) begin
            d_res_valid <= 1'b0;
            d_res_count <= '0;
        end else if (tog[2] != tog[1]) begin
            {d_res_valid, d_res_class, d_res_score, d_res_cycles, d_res_count} <=
                {s_res_valid, s_res_class, s_res_score, s_res_cycles, s_res_count};
        end
    end
    assign d_booted   = f_bt[2];
    assign d_boot_err = f_be[2];
    assign d_busy     = f_bz[2];

endmodule

`default_nettype wire
