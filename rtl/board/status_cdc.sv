// status_cdc: carries npu_system's status into the display's clock domain.
// The result fields change once per inference: the source domain snapshots them and flips
// a toggle; the destination copies the (by then long stable) snapshot when the
// synchronised toggle changes. Single-bit flags go through two-flop synchronisers.

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
    localparam int SW = 1 + 8 + 8 + 32 + 16;

    // ---- source: snapshot and toggle when a new result arrives
    logic [15:0]   s_seen;
    logic          s_tog;
    logic [SW-1:0] s_snap;
    always_ff @(posedge s_clk) begin
        if (s_rst) begin
            s_seen <= '0;
            s_tog  <= 1'b0;
        end else if (s_res_count != s_seen) begin
            s_seen <= s_res_count;
            s_snap <= {s_res_valid, s_res_class, s_res_score, s_res_cycles, s_res_count};
            s_tog  <= !s_tog;
        end
    end

    // ---- destination
    logic [2:0] tog, f_bt, f_be, f_bz;
    always_ff @(posedge d_clk) begin
        tog  <= {tog[1:0], s_tog};
        f_bt <= {f_bt[1:0], s_booted};
        f_be <= {f_be[1:0], s_boot_err};
        f_bz <= {f_bz[1:0], s_busy};
        if (d_rst) begin
            d_res_valid <= 1'b0;
            d_res_count <= '0;
        end else if (tog[2] != tog[1]) begin
            {d_res_valid, d_res_class, d_res_score, d_res_cycles, d_res_count} <= s_snap;
        end
    end
    assign d_booted   = f_bt[2];
    assign d_boot_err = f_be[2];
    assign d_busy     = f_bz[2];

endmodule

`default_nettype wire
