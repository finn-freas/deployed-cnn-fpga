// npu_display: 640x480 HDMI overlay for the person detector.
//
//   left:  the 48x48 gray thumbnail of the last input, scaled x8, framed green (person),
//          red (no person) or gray (no result yet)
//   right: text (5x7 font, x2): title, verdict, score, latency in microseconds, run count
//          and status
//
// Pipeline: video_timing outputs (cycle 0) -> thumbnail read and font lookup (cycle 1) ->
// colour (cycle 2); de/hsync/vsync are delayed to match. Latency = cycles * 1e6 / NPU_HZ,
// formed by shift-and-add when a new result arrives, then converted to decimal.

`default_nettype none

module npu_display #(
    parameter int NPU_HZ = 25_200_000
) (
    input  wire logic        clk,
    input  wire logic        rst,
    // status (same clock domain)
    input  wire logic        booted,
    input  wire logic        boot_err,
    input  wire logic        busy,
    input  wire logic        res_valid,
    input  wire logic [7:0]  res_class,
    input  wire logic signed [7:0] res_score,
    input  wire logic [31:0] res_cycles,
    input  wire logic [15:0] res_count,
    // thumbnail
    output      logic [11:0] thumb_raddr,
    input  wire logic [3:0]  thumb_rdata,
    // video out (to dvi_tx)
    output      logic        de,
    output      logic        hsync,
    output      logic        vsync,
    output      logic [7:0]  r,
    output      logic [7:0]  g,
    output      logic [7:0]  b
);

    // ------------------------------------------------------------------ numbers to text
    localparam int US_K = int'((64'd1 << 16) * 64'd1_000_000 / 64'(NPU_HZ));   // us per cycle, Q16
    logic [15:0] cnt_seen;
    logic        conv;                      // a new result: start the conversions
    logic [47:0] us_acc;                    // cycles * US_K
    logic [31:0] m_a;
    logic [15:0] m_b;
    logic [4:0]  m_n;
    logic        us_start;
    logic [23:0] us_bcd;
    logic [11:0] sc_bcd;
    logic [19:0] rn_bcd;
    logic        us_busy, sc_busy, rn_busy;
    logic [7:0]  sc_mag;

    always_ff @(posedge clk) begin
        conv     <= 1'b0;
        us_start <= 1'b0;
        if (rst) begin
            cnt_seen <= '0;
            m_n      <= '0;
        end else if (res_valid && res_count != cnt_seen) begin
            cnt_seen <= res_count;
            conv     <= 1'b1;
            us_acc   <= '0;
            m_a      <= res_cycles;
            m_b      <= 16'(US_K);
            m_n      <= 5'd16;
        end else if (m_n != 0) begin
            if (m_b[0]) us_acc <= us_acc + 48'(m_a);
            m_a <= m_a << 1;
            m_b <= m_b >> 1;
            m_n <= m_n - 5'd1;
            if (m_n == 5'd1) us_start <= 1'b1;
        end
    end
    assign sc_mag = res_score < 0 ? 8'(-res_score) : 8'(res_score);

    logic [31:0] us_int;
    assign us_int = 32'((us_acc + (48'd1 << 15)) >> 16);       // rounded microseconds
    bin2bcd #(.BITS(20), .DIGITS(6)) u_us (.clk, .rst, .start(us_start),
        .bin((us_int > 32'd999_999) ? 20'd999_999 : 20'(us_int)), .bcd(us_bcd), .busy(us_busy));
    bin2bcd #(.BITS(8), .DIGITS(3)) u_sc (.clk, .rst, .start(conv), .bin(sc_mag), .bcd(sc_bcd), .busy(sc_busy));
    bin2bcd #(.BITS(16), .DIGITS(5)) u_rn (.clk, .rst, .start(conv), .bin(res_count), .bcd(rn_bcd), .busy(rn_busy));

    // decimal digits as characters, leading zeros blank
    function automatic logic [7:0] dch(input logic [3:0] d, input logic blank);
        return blank ? 8'h20 : 8'h30 + 8'(d);
    endfunction

    // ------------------------------------------------------------------ text lines
    localparam int TX = 440, TY0 = 56, TDY = 40, NCH = 12, NL = 6;
    logic [7:0] txt [NL][NCH];
    logic       person;
    assign person = res_class == 8'd1;

    always_comb begin
        logic lz;
        for (int l = 0; l < NL; l++)
            for (int c = 0; c < NCH; c++) txt[l][c] = 8'h20;
        // "VWW ON FPGA"
        {txt[0][0], txt[0][1], txt[0][2], txt[0][4], txt[0][5]} = {"V", "W", "W", "O", "N"};
        {txt[0][7], txt[0][8], txt[0][9], txt[0][10]} = {"F", "P", "G", "A"};
        if (!res_valid) begin
            {txt[1][0], txt[1][1], txt[1][2], txt[1][3]} = {"W", "A", "I", "T"};
        end else if (person) begin
            {txt[1][0], txt[1][1], txt[1][2], txt[1][3], txt[1][4], txt[1][5]} = {"P", "E", "R", "S", "O", "N"};
        end else begin
            {txt[1][0], txt[1][1], txt[1][3], txt[1][4], txt[1][5], txt[1][6], txt[1][7], txt[1][8]} =
                {"N", "O", "P", "E", "R", "S", "O", "N"};
        end
        // "SCORE +nnn"
        {txt[2][0], txt[2][1], txt[2][2], txt[2][3], txt[2][4]} = {"S", "C", "O", "R", "E"};
        if (res_valid) begin
            txt[2][6] = res_score < 0 ? "-" : "+";
            txt[2][7] = dch(sc_bcd[11:8], sc_bcd[11:8] == 0);
            txt[2][8] = dch(sc_bcd[7:4], sc_bcd[11:4] == 0);
            txt[2][9] = dch(sc_bcd[3:0], 1'b0);
        end
        // "LAT nnnnnnUS"
        {txt[3][0], txt[3][1], txt[3][2]} = {"L", "A", "T"};
        {txt[3][10], txt[3][11]} = {"U", "S"};
        if (res_valid) begin
            lz = 1'b1;
            for (int d = 5; d >= 0; d--) begin
                lz = lz && (us_bcd[4*d +: 4] == 0) && d != 0;
                txt[3][4 + 5 - d] = dch(us_bcd[4*d +: 4], lz);
            end
        end
        // "RUNS nnnnn"
        {txt[4][0], txt[4][1], txt[4][2], txt[4][3]} = {"R", "U", "N", "S"};
        lz = 1'b1;
        for (int d = 4; d >= 0; d--) begin
            lz = lz && (rn_bcd[4*d +: 4] == 0) && d != 0;
            txt[4][5 + 4 - d] = dch(rn_bcd[4*d +: 4], lz);
        end
        // status
        if (boot_err)
            {txt[5][0], txt[5][1], txt[5][2], txt[5][3], txt[5][4], txt[5][6], txt[5][7], txt[5][8]} =
                {"F", "L", "A", "S", "H", "E", "R", "R"};
        else if (!booted)
            {txt[5][0], txt[5][1], txt[5][2], txt[5][3], txt[5][4]} = {"B", "O", "O", "T", "."};
        else if (busy)
            {txt[5][0], txt[5][1], txt[5][2], txt[5][3]} = {"B", "U", "S", "Y"};
        else
            {txt[5][0], txt[5][1], txt[5][2], txt[5][3], txt[5][4]} = {"R", "E", "A", "D", "Y"};
    end

    // ------------------------------------------------------------------ raster
    logic [10:0] x;
    logic [9:0]  y;
    logic        act0, hs0, vs0, fs0;
    video_timing u_vt (.clk, .rst, .x, .y, .active(act0), .hsync(hs0), .vsync(vs0), .frame_start(fs0));

    localparam int IX = 32, IY = 48, IS = 8 * 48, BW = 6;     // thumbnail origin, size, frame
    logic in_img0, in_frame0, in_txt0, txt_bit0, txt_hi0;
    logic in_img1, in_frame1, txt_bit1, txt_hi1, act1, hs1, vs1;
    logic [7:0] f_ch, f_bits;
    logic [2:0] f_row, f_col;

    always_comb begin
        logic [10:0] dx, tx;
        logic [9:0]  dy, ly;
        logic [5:0]  ty;
        logic [2:0]  li;
        logic        hit;
        in_img0   = x >= 11'(IX) && x < 11'(IX + IS) && y >= 10'(IY) && y < 10'(IY + IS);
        in_frame0 = x >= 11'(IX - BW) && x < 11'(IX + IS + BW) && y >= 10'(IY - BW) && y < 10'(IY + IS + BW);
        dx = x - 11'(IX);
        dy = y - 10'(IY);
        ty = 6'(dy >> 3);
        thumb_raddr = {1'b0, ty, 5'b0} + {2'b0, ty, 4'b0} + 12'(dx >> 3);
        // text: which line, which character, which glyph row/column
        li  = '0;
        hit = 1'b0;
        ly  = '0;
        for (int l = 0; l < NL; l++)
            if (y >= 10'(TY0 + TDY * l) && y < 10'(TY0 + TDY * l + 16)) begin
                li  = 3'(l);
                hit = 1'b1;
                ly  = y - 10'(TY0 + TDY * l);
            end
        tx      = x - 11'(TX);
        in_txt0 = hit && x >= 11'(TX) && x < 11'(TX + 16 * NCH);
        f_ch    = in_txt0 ? txt[li][4'(tx >> 4)] : 8'h20;
        f_row   = 3'(ly >> 1);
        f_col   = 3'(tx >> 1);
        txt_hi0 = (li == 3'd1);
    end

    font_rom u_font (.ch(f_ch), .row(f_row), .bits(f_bits));
    assign txt_bit0 = f_bits[3'd7 - f_col];

    always_ff @(posedge clk) begin
        in_img1   <= in_img0;
        in_frame1 <= in_frame0;
        txt_bit1  <= txt_bit0;
        txt_hi1   <= txt_hi0;
        act1 <= act0; hs1 <= hs0; vs1 <= vs0;
    end

    always_ff @(posedge clk) begin
        logic [7:0] rr, gg, bb, gray;
        gray = {thumb_rdata, thumb_rdata};
        {rr, gg, bb} = {8'd16, 8'd18, 8'd28};                       // background
        if (in_frame1 && !in_img1) begin
            if (!res_valid)  {rr, gg, bb} = {8'd90, 8'd90, 8'd90};
            else if (person) {rr, gg, bb} = {8'd40, 8'd210, 8'd70};
            else             {rr, gg, bb} = {8'd220, 8'd50, 8'd40};
        end
        if (in_img1) {rr, gg, bb} = {gray, gray, gray};
        if (txt_bit1) begin
            if (txt_hi1 && res_valid) {rr, gg, bb} = person ? {8'd60, 8'd255, 8'd90} : {8'd255, 8'd80, 8'd60};
            else                      {rr, gg, bb} = {8'd235, 8'd235, 8'd235};
        end
        {r, g, b} <= act1 ? {rr, gg, bb} : 24'd0;
        de    <= act1;
        hsync <= hs1;
        vsync <= vs1;
    end

    logic unused;
    assign unused = fs0 ^ us_busy ^ sc_busy ^ rn_busy;

endmodule

`default_nettype wire
