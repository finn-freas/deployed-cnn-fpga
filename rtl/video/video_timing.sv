// video_timing: raster counters and sync generation.
// Defaults are VESA 640x480@60 (25.175 MHz nominal; 25.2 MHz is within monitor tolerance).
// All outputs are registered and aligned to each other.

`default_nettype none

module video_timing #(
    parameter int H_ACTIVE = 640,
    parameter int H_FP     = 16,
    parameter int H_SYNC   = 96,
    parameter int H_BP     = 48,
    parameter int V_ACTIVE = 480,
    parameter int V_FP     = 10,
    parameter int V_SYNC   = 2,
    parameter int V_BP     = 33,
    parameter bit SYNC_NEG = 1'b1   // 640x480 uses negative sync polarity
) (
    input  wire logic        clk,
    input  wire logic        rst,
    output      logic [10:0] x,       // pixel column, valid when active
    output      logic [9:0]  y,       // pixel row, valid when active
    output      logic        active,  // inside the visible area
    output      logic        hsync,
    output      logic        vsync,
    output      logic        frame_start  // one-cycle pulse at x=0,y=0
);

    localparam int H_TOTAL = H_ACTIVE + H_FP + H_SYNC + H_BP;
    localparam int V_TOTAL = V_ACTIVE + V_FP + V_SYNC + V_BP;

    logic [10:0] hc;
    logic [9:0]  vc;

    always_ff @(posedge clk) begin
        if (rst) begin
            hc <= '0;
            vc <= '0;
        end else if (hc == 11'(H_TOTAL - 1)) begin
            hc <= '0;
            vc <= (vc == 10'(V_TOTAL - 1)) ? '0 : vc + 10'd1;
        end else begin
            hc <= hc + 11'd1;
        end
    end

    // Register outputs so downstream logic sees a clean, aligned set.
    always_ff @(posedge clk) begin
        if (rst) begin
            x           <= '0;
            y           <= '0;
            active      <= 1'b0;
            hsync       <= SYNC_NEG;
            vsync       <= SYNC_NEG;
            frame_start <= 1'b0;
        end else begin
            x           <= hc;
            y           <= vc;
            active      <= (hc < 11'(H_ACTIVE)) && (vc < 10'(V_ACTIVE));
            hsync       <= SYNC_NEG ^ ((hc >= 11'(H_ACTIVE + H_FP)) &&
                                       (hc <  11'(H_ACTIVE + H_FP + H_SYNC)));
            vsync       <= SYNC_NEG ^ ((vc >= 10'(V_ACTIVE + V_FP)) &&
                                       (vc <  10'(V_ACTIVE + V_FP + V_SYNC)));
            frame_start <= (hc == '0) && (vc == '0);
        end
    end

endmodule

`default_nettype wire
