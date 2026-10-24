// npu_requant: int32 accumulator -> int8, bit-exact with TFLite reference kernels.
//
//   x = acc + bias
//   rmode 0 (double rounding, CONV_2D / DEPTHWISE_CONV_2D):
//       y = RoundingDivideByPOT(SRDHM(x << max(shift,0), mult), max(-shift,0))
//   rmode 1 (single rounding, FULLY_CONNECTED):
//       y = (x * mult + 2^(30-shift)) >> (31-shift)
//   out = clamp(y + out_zp, act_min, act_max)
//
// Fully pipelined, one result per cycle, 4 cycles from inputs to `out`
// (npu_tile_engine's RQ_LAT must match).

`default_nettype none

module npu_requant
    import npu_pkg::*;
(
    input  wire logic                      clk,
    input  wire logic signed [ACC_W-1:0]   acc,
    input  wire logic signed [31:0]        bias,
    input  wire logic signed [31:0]        mult,
    input  wire logic signed [SHIFT_W-1:0] shift,
    input  wire logic                      rmode,
    input  wire logic signed [7:0]         out_zp,
    input  wire logic signed [7:0]         act_min,
    input  wire logic signed [7:0]         act_max,
    output      logic signed [7:0]         out
);

    localparam logic signed [31:0] I32_MIN = 32'sh8000_0000;
    localparam logic signed [31:0] I32_MAX = 32'sh7FFF_FFFF;

    // ---------------- stage 1: bias add, pre-shift
    logic signed [31:0] s1_x, s1_mult;
    logic [5:0]         s1_rs;          // right shift for double rounding
    logic [6:0]         s1_total;       // 31 - shift for single rounding
    logic               s1_rmode, s1_ovf;
    logic signed [7:0]  s1_zp, s1_min, s1_max;

    always_ff @(posedge clk) begin
        logic signed [31:0] x;
        logic [5:0] ls;
        x = acc + bias;                                   // int32 add, as in TFLite
        ls = (shift > 0) ? 6'(shift) : 6'd0;
        s1_rs    <= (shift > 0) ? 6'd0 : 6'(-shift);
        s1_total <= 7'(7'sd31 - 7'(shift));
        // Double rounding multiplies the int32-wrapped left-shifted value; single rounding
        // multiplies x itself (its left shift is folded into `total`).
        s1_x     <= rmode ? x : (x <<< ls);
        s1_ovf   <= !rmode && ((x <<< ls) == I32_MIN) && (mult == I32_MIN);
        s1_mult  <= mult;
        s1_rmode <= rmode;
        s1_zp    <= out_zp;
        s1_min   <= act_min;
        s1_max   <= act_max;
    end

    // ---------------- stage 2: 32 x 32 -> 64 multiply
    logic signed [63:0] s2_prod;
    logic [5:0]         s2_rs;
    logic [6:0]         s2_total;
    logic               s2_rmode, s2_ovf;
    logic signed [7:0]  s2_zp, s2_min, s2_max;

    always_ff @(posedge clk) begin
        s2_prod  <= 64'(s1_x) * 64'(s1_mult);
        s2_rs    <= s1_rs;
        s2_total <= s1_total;
        s2_rmode <= s1_rmode;
        s2_ovf   <= s1_ovf;
        s2_zp    <= s1_zp;
        s2_min   <= s1_min;
        s2_max   <= s1_max;
    end

    // ---------------- stage 3: rounding
    // 40 bits: single rounding with a left shift can exceed int32 (outside TFLite's valid
    // range); keeping the extra bits makes the final clamp saturate instead of wrapping.
    logic signed [39:0] s3_y;
    logic signed [7:0]  s3_zp, s3_min, s3_max;

    always_ff @(posedge clk) begin
        logic signed [63:0] s, mag;
        logic signed [31:0] hi, mask, rem, thr, q32;
        if (s2_rmode) begin
            // single rounding: round half up, arithmetic shift
            s3_y <= 40'((s2_prod + (64'sd1 <<< (s2_total - 7'd1))) >>> s2_total);
        end else begin
            // SaturatingRoundingDoublingHighMul: (prod + nudge) / 2^31, truncating toward zero
            s   = s2_prod + ((s2_prod >= 0) ? 64'sd1073741824 : -64'sd1073741823);
            mag = (s >= 0) ? s : -s;
            hi  = (s >= 0) ? 32'(mag >>> 31) : -32'(mag >>> 31);
            if (s2_ovf) hi = I32_MAX;
            // RoundingDivideByPOT: round half away from zero
            mask = (32'sd1 <<< s2_rs) - 32'sd1;
            rem  = hi & mask;
            thr  = (mask >>> 1) + ((hi < 0) ? 32'sd1 : 32'sd0);
            q32  = (hi >>> s2_rs) + ((rem > thr) ? 32'sd1 : 32'sd0);
            s3_y <= 40'(q32);
        end
        s3_zp  <= s2_zp;
        s3_min <= s2_min;
        s3_max <= s2_max;
    end

    // ---------------- stage 4: zero point + clamp
    always_ff @(posedge clk) begin
        logic signed [40:0] y;
        y = 41'(s3_y) + 41'(s3_zp);
        if (y < 41'(s3_min))      out <= s3_min;
        else if (y > 41'(s3_max)) out <= s3_max;
        else                      out <= 8'(y);
    end

endmodule

`default_nettype wire
