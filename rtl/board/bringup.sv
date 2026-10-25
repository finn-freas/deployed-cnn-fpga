// bringup: board bring-up tests for the Tang Nano 20K, driven over the UART.
//
// Checks, one at a time, the parts the person detector depends on, with the same
// controllers, parameters and pins as npu_board_top:
//   * at reset: a banner, then the SDRAM's initialisation and read-latency calibration
//   * 'p'  ping                            -> "pong"
//   * 's'  the SDRAM status line again
//   * 'm'  SDRAM: writes MEM_WORDS 64-bit words (the region the NPU uses) and reads them
//          back as streamed bursts, twice with inverted patterns; first error reported
//   * 'f'  flash: reads the model image header at BOOT_W and sums the weights/params image
//          (sum of all 32-bit halves, mod 2^32); sw/board/board_check.py compares it
//   * 'a'  (or the `run_all` pulse): m then f
// Replies are text lines (any terminal at 1 Mbaud works). `status`: {fail, pass, cal_err,
// sdram ready, busy} for the LEDs.

`default_nettype none

module bringup #(
    parameter int          CLK_HZ     = 54_000_000,
    parameter int          BAUD       = 1_000_000,
    parameter logic [23:0] FLASH_BASE = 24'h40_0000,
    parameter int          BOOT_W     = 'h10000,
    parameter int          SPI_DIV    = 1,
    parameter int          MEM_WORDS  = 1 << 17
) (
    input  wire logic        clk,
    input  wire logic        rst,
    input  wire logic        run_all,
    input  wire logic        uart_rx,
    output      logic        uart_tx,
    output      logic [4:0]  status,
    // configuration flash (used as I/O after configuration)
    output      logic        spi_cs_n,
    output      logic        spi_clk,
    output      logic        spi_mosi,
    input  wire logic        spi_miso,
    // SDRAM pins
    output      logic        sd_cke,
    output      logic        sd_cs_n,
    output      logic        sd_ras_n,
    output      logic        sd_cas_n,
    output      logic        sd_we_n,
    output      logic [1:0]  sd_ba,
    output      logic [10:0] sd_addr,
    output      logic [3:0]  sd_dqm,
    output      logic [31:0] sd_dq_o,
    output      logic        sd_dq_oe,
    input  wire logic [31:0] sd_dq_i
);
    localparam int RD_CHUNK = 4096;                 // words per streamed read request
    localparam int MSGW     = 8 * 96;               // longest reply

    // ---------------------------------------------------------------- UART
    localparam int DIV = (CLK_HZ + BAUD / 2) / BAUD;
    logic       rx_v, tx_v, tx_rdy;
    logic [7:0] rx_d, tx_d;
    uart_rx #(.DIV(DIV)) u_rx (.clk, .rst, .rx(uart_rx), .valid(rx_v), .data(rx_d));
    uart_tx #(.DIV(DIV)) u_tx (.clk, .rst, .valid(tx_v), .data(tx_d), .ready(tx_rdy), .tx(uart_tx));

    // text replies: the message is left-aligned in `msg` and shifted out byte by byte
    logic [MSGW-1:0] msg;
    logic [6:0]      msg_len;
    assign tx_v = (msg_len != 0);
    assign tx_d = msg[MSGW-1 -: 8];

    function automatic logic [7:0] hd(input logic [3:0] n);
        return (n < 4'd10) ? 8'h30 + 8'(n) : 8'h57 + 8'(n);        // 0-9, a-f
    endfunction
    function automatic logic [39:0] hex5(input logic [19:0] v);
        for (int i = 0; i < 5; i++) hex5[8 * i +: 8] = hd(v[4 * i +: 4]);
    endfunction
    function automatic logic [63:0] hex8(input logic [31:0] v);
        for (int i = 0; i < 8; i++) hex8[8 * i +: 8] = hd(v[4 * i +: 4]);
    endfunction
    function automatic logic [127:0] hex16(input logic [63:0] v);
        for (int i = 0; i < 16; i++) hex16[8 * i +: 8] = hd(v[4 * i +: 4]);
    endfunction
    `define SAY(X) begin msg <= {X, {(MSGW - $bits(X)){1'b0}}}; msg_len <= 7'($bits(X) / 8); end

    // ---------------------------------------------------------------- SDRAM
    logic        sd_ready, cal_err, sd_idle, rd_req, rd_ack, rvalid, wr_req, wr_ack;
    logic [1:0]  cal_rl;
    logic [16:0] rd_addr, wr_addr;
    logic [15:0] rd_len;
    logic [63:0] rdata, wr_data;
    sdram_ctrl #(.CLK_HZ(CLK_HZ)) u_sdram (
        .clk, .rst, .ready(sd_ready), .cal_err, .cal_rl, .idle(sd_idle),
        .rd_req, .rd_addr, .rd_len, .rd_ack, .rvalid, .rdata,
        .wr_req, .wr_addr, .wr_data, .wr_ack,
        .sd_cke, .sd_cs_n, .sd_ras_n, .sd_cas_n, .sd_we_n, .sd_ba, .sd_addr, .sd_dqm,
        .sd_dq_o, .sd_dq_oe, .sd_dq_i);

    // test pattern: an invertible scramble of the address in both halves (every data bit
    // toggles across addresses), inverted on the second pass
    function automatic logic [63:0] pat(input logic [16:0] a, input logic inv);
        logic [31:0] x;
        x = 32'(a);
        x = x ^ (x << 13);
        x = x ^ (x >> 17);
        x = x ^ (x << 5);
        return {x ^ 32'hC3A5_5A3C, ~x} ^ {64{inv}};
    endfunction

    // ---------------------------------------------------------------- flash
    logic        f_req, f_ack, f_rvalid;
    logic [16:0] f_addr;
    logic [15:0] f_len;
    logic [63:0] f_rdata;
    spi_flash_reader #(.AW(17), .FLASH_BASE(FLASH_BASE), .CLK_DIV(SPI_DIV)) u_flash (
        .clk, .rst, .req(f_req), .addr(f_addr), .len(f_len), .ack(f_ack),
        .rvalid(f_rvalid), .rdata(f_rdata),
        .spi_cs_n, .spi_clk, .spi_mosi, .spi_miso);

    // ---------------------------------------------------------------- tests
    typedef enum logic [3:0] {
        T_BANNER, T_SDWAIT, T_IDLE, T_MW, T_MR, T_MDONE, T_FHDR, T_FSUM, T_FBAD, T_FDONE
    } tstate_e;
    tstate_e     ts;
    logic        seq;               // 'a': flash test after the memory test
    logic        pass_n;            // memory test pass (0, 1)
    logic [16:0] ma, mr;            // write address; read: next word expected
    logic [16:0] mreq;              // read: next chunk to request
    logic [31:0] errs;
    logic [16:0] e_addr;
    logic [63:0] e_got, e_exp;
    logic        f_hdr_seen;
    logic [63:0] f_hdr;
    logic [16:0] f_n, f_i, f_next;  // image words, words summed, next to request
    logic [31:0] f_sum;
    logic        ok_flag, bad_flag;
    logic [24:0] sd_timeout;

    assign status = {bad_flag, ok_flag, cal_err, sd_ready, (ts != T_IDLE)};

    always_ff @(posedge clk) begin
        if (msg_len != 0 && tx_rdy) begin
            msg     <= msg << 8;
            msg_len <= msg_len - 7'd1;
        end

        if (rst) begin
            ts       <= T_BANNER;
            msg_len  <= '0;
            wr_req   <= 1'b0;
            rd_req   <= 1'b0;
            f_req    <= 1'b0;
            seq      <= 1'b0;
            ok_flag  <= 1'b0;
            bad_flag <= 1'b0;
        end else begin
            unique case (ts)
                T_BANNER: begin
                    `SAY({"\r\nNPU bring-up (54 MHz, 1 Mbaud). Commands: p ping, s status, m SDRAM, f flash, a all\r\n"})
                    sd_timeout <= '0;
                    ts <= T_SDWAIT;
                end

                // SDRAM initialisation and calibration take well under a millisecond
                T_SDWAIT: if (msg_len == 0) begin
                    sd_timeout <= sd_timeout + 25'd1;
                    if (cal_err) begin
                        `SAY({"SDRAM: calibration FAILED (no read latency returned the pattern)\r\n"})
                        bad_flag <= 1'b1;
                        ts <= T_IDLE;
                    end else if (sd_ready) begin
                        `SAY({"SDRAM: ready, read latency CAS-1+", hd(4'(cal_rl)), "\r\n"})
                        ts <= T_IDLE;
                    end else if (&sd_timeout) begin
                        `SAY({"SDRAM: not ready (initialisation hung)\r\n"})
                        bad_flag <= 1'b1;
                        ts <= T_IDLE;
                    end
                end

                T_IDLE: if (msg_len == 0) begin
                    logic go_m, go_f;
                    go_m = run_all || (rx_v && (rx_d == "m" || rx_d == "a"));
                    go_f = rx_v && rx_d == "f";
                    if (go_m || go_f) begin
                        ok_flag  <= 1'b0;
                        bad_flag <= 1'b0;
                    end
                    if (go_m) begin
                        seq    <= run_all || rx_d == "a";
                        pass_n <= 1'b0;
                        ma     <= '0;
                        errs   <= '0;
                        if (!sd_ready) begin
                            `SAY({"mem: SDRAM not ready\r\n"})
                            bad_flag <= 1'b1;
                        end else begin
                            ts <= T_MW;
                        end
                    end else if (go_f) begin
                        seq        <= 1'b0;
                        f_hdr_seen <= 1'b0;
                        f_req      <= 1'b1;
                        f_addr     <= 17'(BOOT_W);
                        f_len      <= 16'd2;
                        ts         <= T_FHDR;
                    end else if (rx_v && rx_d == "s") begin
                        sd_timeout <= '0;
                        ts <= T_SDWAIT;
                    end else if (rx_v && rx_d == "p") begin
                        `SAY({"pong\r\n"})
                    end else if (rx_v && rx_d != "\r" && rx_d != "\n") begin
                        `SAY({"? commands: p ping, s status, m SDRAM, f flash, a all\r\n"})
                    end
                end

                // ---- SDRAM: write every word ...
                T_MW: begin
                    if (wr_req && wr_ack) begin
                        wr_req <= 1'b0;
                        ma     <= ma + 17'd1;
                        if (ma == 17'(MEM_WORDS - 1)) begin
                            mr   <= '0;
                            mreq <= '0;
                            ts   <= T_MR;
                        end
                    end else if (!wr_req && sd_idle) begin     // the last write has finished
                        wr_req  <= 1'b1;
                        wr_addr <= ma;
                        wr_data <= pat(ma, pass_n);
                    end
                end

                // ... then read it back in streamed chunks
                T_MR: begin
                    if (rd_req && rd_ack) rd_req <= 1'b0;
                    else if (!rd_req && mreq == mr && 32'(mreq) < 32'(MEM_WORDS)) begin
                        rd_req  <= 1'b1;
                        rd_addr <= mreq;
                        rd_len  <= 16'(RD_CHUNK);
                        mreq    <= mreq + 17'(RD_CHUNK);
                    end
                    if (rvalid) begin
                        if (rdata != pat(mr, pass_n)) begin
                            if (errs == 0) begin
                                e_addr <= mr;
                                e_got  <= rdata;
                                e_exp  <= pat(mr, pass_n);
                            end
                            errs <= errs + 32'd1;
                        end
                        mr <= mr + 17'd1;
                        if (mr == 17'(MEM_WORDS - 1)) begin
                            if (!pass_n) begin
                                pass_n <= 1'b1;
                                ma     <= '0;
                                ts     <= T_MW;
                            end else begin
                                ts <= T_MDONE;
                            end
                        end
                    end
                end

                T_MDONE: begin
                    if (errs == 0) begin
                        `SAY({"mem: PASS, 0x", hex5(20'(MEM_WORDS)), " words x 2 patterns\r\n"})
                        ok_flag <= 1'b1;
                    end else begin
                        `SAY({"mem: FAIL, 0x", hex8(errs), " errors, first @", hex5(20'(e_addr)),
                              " got ", hex16(e_got), " exp ", hex16(e_exp), "\r\n"})
                        bad_flag <= 1'b1;
                    end
                    if (seq) begin
                        f_hdr_seen <= 1'b0;
                        f_req      <= 1'b1;
                        f_addr     <= 17'(BOOT_W);
                        f_len      <= 16'd2;
                        ts         <= T_FHDR;
                    end else begin
                        ts <= T_IDLE;
                    end
                end

                // ---- flash: the boot header (word 0: magic, word 1: image length) ...
                T_FHDR: begin
                    if (f_req && f_ack) f_req <= 1'b0;
                    if (f_rvalid) begin
                        if (!f_hdr_seen) begin
                            f_hdr      <= f_rdata;
                            f_hdr_seen <= 1'b1;
                        end else if (f_hdr[63:58] != 6'b101101 || f_rdata[31:17] != 0 || f_rdata[16:0] == 0) begin
                            ts <= T_FBAD;
                        end else begin
                            f_n    <= f_rdata[16:0];
                            f_i    <= '0;
                            f_next <= '0;
                            f_sum  <= '0;
                            ts     <= T_FSUM;
                        end
                    end
                end

                // ... and the sum of the weights/params image
                T_FSUM: begin
                    if (f_req && f_ack) f_req <= 1'b0;
                    else if (!f_req && f_next == f_i && f_next < f_n) begin
                        f_req  <= 1'b1;
                        f_addr <= f_next;
                        f_len  <= (f_n - f_next > 17'(RD_CHUNK)) ? 16'(RD_CHUNK) : 16'(f_n - f_next);
                        f_next <= (f_n - f_next > 17'(RD_CHUNK)) ? f_next + 17'(RD_CHUNK) : f_n;
                    end
                    if (f_rvalid) begin
                        f_sum <= f_sum + f_rdata[31:0] + f_rdata[63:32];
                        f_i   <= f_i + 17'd1;
                        if (f_i == f_n - 17'd1) ts <= T_FDONE;
                    end
                end

                T_FBAD: if (msg_len == 0) begin
                    `SAY({"flash: no model image at 0x", hex8(32'(FLASH_BASE) + 32'(BOOT_W * 8)),
                          " (header ", hex16(f_hdr), ")\r\n"})
                    bad_flag <= 1'b1;
                    ts <= T_IDLE;
                end

                T_FDONE: if (msg_len == 0) begin
                    `SAY({"flash: image found, header ", hex16(f_hdr), ", 0x", hex5(20'(f_n)),
                          " words, sum ", hex8(f_sum), "\r\n"})
                    if (!bad_flag) ok_flag <= 1'b1;
                    ts <= T_IDLE;
                end

                default: ts <= T_IDLE;
            endcase
        end
    end
    `undef SAY

endmodule

`default_nettype wire
