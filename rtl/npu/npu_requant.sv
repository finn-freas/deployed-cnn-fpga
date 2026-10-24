// npu_requant: int32 accumulator -> int8, bit-exact with TFLite reference kernels.
//
//   x = acc + bias
//   rmode 0 (double rounding, CONV_2D / DEPTHWISE_CONV_2D):
//       y = RoundingDivideByPOT(SRDHM(x << max(shift,0), mult), max(-shift,0))
//   rmode 1 (single rounding, FULLY_CONNECTED; shift <= 0, checked by the compiler):
//       y = (x * mult + 2^(30-shift)) >> (31-shift)
//   out = clamp(y + out_zp, act_min, act_max)
//
// Both modes share one rounding datapath. With p = x*mult, rs = max(-shift,0) and
// p30 = p >> 30 (floor):
//   SRDHM truncates (p + nudge) / 2^31 toward zero, which for either sign of p equals
//   (p + 2^30) >> 31 = (p30 + 1) >> 1; RoundingDivideByPOT(h, rs) = (h >> rs) + (rem > thr).
//   Single rounding: (p + 2^(30+rs)) >> (31+rs) = (p30 + 2^rs) >> (rs+1), since adding p's
//   dropped fraction to the numerator cannot change a floor division by 2^(rs+1) >= 2.
// So z = p30 + (single ? 2^rs : 1); y = (z >> (rs+1)) + (double ? rem > thr : 0), where
// rem is h's low rs bits, h = z >> 1.
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

    // ---------------- stage 1: bias add, pre-shift
    logic signed [31:0] s1_x, s1_mult;
    logic [4:0]         s1_rs;          // right shift (both modes)
    logic               s1_rmode, s1_ovf;
    logic signed [7:0]  s1_zp, s1_min, s1_max;

    always_ff @(posedge clk) begin
        logic signed [31:0] x;
        logic [5:0] ls;
        x = 32'(acc) + bias;                              // int32 add, as in TFLite
        ls = (shift > 0) ? 6'(shift) : 6'd0;
        s1_rs    <= (shift > 0) ? 5'd0 : 5'(-shift);
        // Double rounding multiplies the int32-wrapped left-shifted value.
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
    logic [4:0]         s2_rs;
    logic               s2_rmode, s2_ovf;
    logic signed [7:0]  s2_zp, s2_min, s2_max;

    always_ff @(posedge clk) begin
        s2_prod  <= 64'(s1_x) * 64'(s1_mult);
        s2_rs    <= s1_rs;
        s2_rmode <= s1_rmode;
        s2_ovf   <= s1_ovf;
        s2_zp    <= s1_zp;
        s2_min   <= s1_min;
        s2_max   <= s1_max;
    end

    // ---------------- stage 3: shared rounding
    logic signed [33:0] s3_q;           // z >> (rs+1)
    logic               s3_up;          // double rounding: round the remainder up
    logic signed [7:0]  s3_zp, s3_min, s3_max;

    always_ff @(posedge clk) begin
        logic signed [33:0] z;
        logic signed [31:0] h, mask, rem, thr;
        z = 34'(s2_prod >>> 30) + (s2_rmode ? 34'sd1 <<< s2_rs : 34'sd1);
        if (s2_ovf) z = 34'sh0_FFFF_FFFE;                 // SRDHM saturates to INT32_MAX
        h    = 32'(z >>> 1);
        mask = (32'sd1 <<< s2_rs) - 32'sd1;
        rem  = h & mask;
        thr  = (mask >>> 1) + ((h < 0) ? 32'sd1 : 32'sd0);
        s3_q   <= z >>> (6'(s2_rs) + 6'd1);
        s3_up  <= !s2_rmode && (rem > thr);
        s3_zp  <= s2_zp;
        s3_min <= s2_min;
        s3_max <= s2_max;
    end

    // ---------------- stage 4: zero point + clamp
    always_ff @(posedge clk) begin
        logic signed [34:0] y;
        y = 35'(s3_q) + (s3_up ? 35'sd1 : 35'sd0) + 35'(s3_zp);
        if (y < 35'(s3_min))      out <= s3_min;
        else if (y > 35'(s3_max)) out <= s3_max;
        else                      out <= 8'(y);
    end

endmodule

`default_nettype wire
