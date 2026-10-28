`default_nettype none

module npu_display #(
    parameter int NPU_HZ = 25_200_000
) (
    input  wire logic        clk,
    input  wire logic        rst,
    input  wire logic        booted,
    input  wire logic        boot_err,
    input  wire logic        busy,
    input  wire logic        res_valid,
    input  wire logic [7:0]  res_class,
    input  wire logic signed [7:0] res_score,
    input  wire logic [31:0] res_cycles,
    input  wire logic [15:0] res_count,
    output      logic [11:0] thumb_raddr,
    input  wire logic [3:0]  thumb_rdata,
    output      logic        de,
    output      logic        hsync,
    output      logic        vsync,
    output      logic [7:0]  r,
    output      logic [7:0]  g,
    output      logic [7:0]  b
);

    localparam int US_K = int'((64'd1 << 16) * 64'd1_000_000 / 64'(NPU_HZ));
    logic [15:0] cnt_seen;
    logic        conv;
    logic [47:0] us_acc;
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
    assign us_int = 32'((us_acc + (48'd1 << 15)) >> 16);
    bin2bcd #(.BITS(20), .DIGITS(6)) u_us (.clk, .rst, .start(us_start),
        .bin((us_int > 32'd999_999) ? 20'd999_999 : 20'(us_int)), .bcd(us_bcd), .busy(us_busy));
    bin2bcd #(.BITS(8), .DIGITS(3)) u_sc (.clk, .rst, .start(conv), .bin(sc_mag), .bcd(sc_bcd), .busy(sc_busy));
    bin2bcd #(.BITS(16), .DIGITS(5)) u_rn (.clk, .rst, .start(conv), .bin(res_count), .bcd(rn_bcd), .busy(rn_busy));

    localparam int NCH = 12, NL = 6;

    function automatic logic [6:0] dch(input logic [3:0] d, input logic blank);
        return blank ? 7'h20 : 7'h30 + 7'(d);
    endfunction

    function automatic logic [6:0] str(input logic [8 * NCH - 1:0] s, input int c);
        return (c >= 0 && c < NCH) ? 7'(s >> (8 * (NCH - 1 - c))) : 7'h20;
    endfunction

    logic person;
    assign person = res_class == 8'd1;

    logic [2:0] li;
    logic [3:0] col;
    logic [6:0] ch;
    always_comb begin
        int c;
        c  = int'(col);
        ch = 7'h20;
        unique case (li)
            3'd0: ch = str("VWW ON FPGA ", c);
            3'd1: ch = !res_valid ? str("WAITING     ", c) : person ? str("PERSON      ", c) : str("NO PERSON   ", c);
            3'd2: begin
                ch = str("SCORE       ", c);
                if (res_valid) begin
                    if (c == 6) ch = res_score < 0 ? 7'h2D : 7'h2B;
                    if (c == 7) ch = dch(sc_bcd[11:8], sc_bcd[11:8] == 0);
                    if (c == 8) ch = dch(sc_bcd[7:4], sc_bcd[11:4] == 0);
                    if (c == 9) ch = dch(sc_bcd[3:0], 1'b0);
                end
            end
            3'd3: begin
                ch = (c >= 10) ? str("US          ", c - 10) : str("LAT         ", c);
                if (res_valid) begin
                    if (c == 4) ch = dch(us_bcd[23:20], us_bcd[23:20] == 0);
                    if (c == 5) ch = dch(us_bcd[19:16], us_bcd[23:16] == 0);
                    if (c == 6) ch = dch(us_bcd[15:12], us_bcd[23:12] == 0);
                    if (c == 7) ch = dch(us_bcd[11:8], us_bcd[23:8] == 0);
                    if (c == 8) ch = dch(us_bcd[7:4], us_bcd[23:4] == 0);
                    if (c == 9) ch = dch(us_bcd[3:0], 1'b0);
                end
            end
            3'd4: begin
                ch = str("RUNS        ", c);
                if (c == 5) ch = dch(rn_bcd[19:16], rn_bcd[19:16] == 0);
                if (c == 6) ch = dch(rn_bcd[15:12], rn_bcd[19:12] == 0);
                if (c == 7) ch = dch(rn_bcd[11:8], rn_bcd[19:8] == 0);
                if (c == 8) ch = dch(rn_bcd[7:4], rn_bcd[19:4] == 0);
                if (c == 9) ch = dch(rn_bcd[3:0], 1'b0);
            end
            3'd5: ch = boot_err ? str("FLASH ERR   ", c) : !booted ? str("BOOT.       ", c) :
                       busy ? str("BUSY        ", c) : str("READY       ", c);
            default: ;
        endcase
    end

    logic [10:0] x;
    logic [9:0]  y;
    logic        act0, hs0, vs0, fs0;
    video_timing u_vt (.clk, .rst, .x, .y, .active(act0), .hsync(hs0), .vsync(vs0), .frame_start(fs0));

    localparam int HT = 800;
    localparam int IX = 32, IY = 48, IS = 8 * 48, BW = 6;
    localparam int TX = 440, TY0 = 56, TDY = 40;

    logic img_x, img_y, frm_x, frm_y, txt_x, txt_y;
    logic [2:0]  sx, sy;
    logic [5:0]  tcol;
    logic [11:0] trow;
    logic [3:0]  cx;
    logic [5:0]  lp;
    logic        eol;
    assign eol = (x == 11'(HT - 1));

    always_ff @(posedge clk) begin
        if (rst) begin
            {img_x, img_y, frm_x, frm_y, txt_x, txt_y} <= '0;
            li <= '0;
            lp <= '0;
        end else begin
            if (x == 11'(IX - BW - 1))      frm_x <= 1'b1;
            if (x == 11'(IX + IS + BW - 1)) frm_x <= 1'b0;
            if (x == 11'(IX - 1)) begin
                img_x <= 1'b1;
                sx    <= '0;
                tcol  <= '0;
            end else if (img_x) begin
                sx <= sx + 3'd1;
                if (sx == 3'd7) tcol <= tcol + 6'd1;
            end
            if (x == 11'(IX + IS - 1)) img_x <= 1'b0;
            if (x == 11'(TX - 1)) begin
                txt_x <= 1'b1;
                cx    <= '0;
                col   <= '0;
            end else if (txt_x) begin
                cx <= cx + 4'd1;
                if (cx == 4'd15) col <= col + 4'd1;
            end
            if (x == 11'(TX + 16 * NCH - 1)) txt_x <= 1'b0;
            if (eol) begin
                if (y == 10'(IY - BW - 1))      frm_y <= 1'b1;
                if (y == 10'(IY + IS + BW - 1)) frm_y <= 1'b0;
                if (y == 10'(IY - 1)) begin
                    img_y <= 1'b1;
                    sy    <= '0;
                    trow  <= '0;
                end else if (img_y) begin
                    sy <= sy + 3'd1;
                    if (sy == 3'd7) trow <= trow + 12'd48;
                end
                if (y == 10'(IY + IS - 1)) img_y <= 1'b0;
                if (y == 10'(TY0 - 1)) begin
                    txt_y <= 1'b1;
                    lp    <= '0;
                    li    <= '0;
                end else if (txt_y) begin
                    lp <= lp + 6'd1;
                    if (lp == 6'(TDY - 1)) begin
                        lp <= '0;
                        li <= li + 3'd1;
                        if (li == 3'(NL - 1)) txt_y <= 1'b0;
                    end
                end
            end
        end
    end
    assign thumb_raddr = trow + 12'(tcol);

    logic [7:0] f_bits;
    font_rom u_font (.clk, .ch, .row(lp[3:1]), .bits(f_bits));

    logic in_img1, in_frame1, in_txt1, txt_hi1, act1, hs1, vs1;
    logic [2:0] gcol1;
    always_ff @(posedge clk) begin
        in_img1   <= img_x && img_y;
        in_frame1 <= frm_x && frm_y;
        in_txt1   <= txt_x && txt_y && lp < 6'd16;
        txt_hi1   <= (li == 3'd1);
        gcol1     <= cx[3:1];
        act1 <= act0; hs1 <= hs0; vs1 <= vs0;
    end

    always_ff @(posedge clk) begin
        logic [7:0] rr, gg, bb, gray;
        gray = {thumb_rdata, thumb_rdata};
        {rr, gg, bb} = {8'd16, 8'd18, 8'd28};
        if (in_frame1 && !in_img1) begin
            if (!res_valid)  {rr, gg, bb} = {8'd90, 8'd90, 8'd90};
            else if (person) {rr, gg, bb} = {8'd40, 8'd210, 8'd70};
            else             {rr, gg, bb} = {8'd220, 8'd50, 8'd40};
        end
        if (in_img1) {rr, gg, bb} = {gray, gray, gray};
        if (in_txt1 && f_bits[3'd7 - gcol1]) begin
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
