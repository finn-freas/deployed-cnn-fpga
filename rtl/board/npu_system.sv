// npu_system: the NPU as a standalone device on a board with an SPI flash and a UART.
//
// Flash (64-bit words from FLASH_BASE; the core's external-store port reads words
// 0..EXT_DEPTH-1 directly, so the weights/params image sits at the start):
//   word BOOT_W     header {magic 6'b101101, img_c[2:0], img_w[6:0], in_words[15:0],
//                           in_base[15:0], n_beats[15:0]}
//   word BOOT_W+1   ext_words: length of the weights/params image
//   then            the program: n_beats 32-bit instruction beats, two per word (low first)
//   then            a demo image: in_words words of packed uint8 HWC bytes
// At reset the boot loader copies the program into instruction memory and the demo image
// into activation memory at in_base, then runs one inference. Button `run`: reload the demo
// image and run again (the program overwrites its input while it runs).
//
// UART (8N1): send 0x49 ('I') then in_words*8 image bytes; the image is written to
// activation memory, inference runs, and the reply is 0xA5, class, score, cycles (4 bytes,
// little-endian). Send the next image only after the reply.
//
// EXT_SDRAM = 1: at boot the weights/params image is first copied from flash into the
// in-package SDRAM, and the core's external-store port reads the SDRAM (one word per two
// cycles instead of one per ~130 over 1-bit SPI).
//
// While an image streams in, every second pixel of every second row (from 96x96) becomes a
// 4-bit gray pixel of a 48x48 thumbnail, which the display reads through `thumb_*`.

`default_nettype none

module npu_system
    import npu_pkg::*;
#(
    parameter int          CLK_HZ     = 25_200_000,
    parameter int          BAUD       = 1_000_000,
    parameter logic [23:0] FLASH_BASE = 24'h40_0000,
    parameter int          BOOT_W     = 'h10000,
    parameter int          SPI_DIV    = 1,
    parameter bit          EXT_SDRAM  = 1'b0
) (
    input  wire logic        clk,
    input  wire logic        rst,
    input  wire logic        run,           // pulse: reload the demo image and run
    input  wire logic        uart_rx,
    output      logic        uart_tx,
    output      logic        spi_cs_n,
    output      logic        spi_clk,
    output      logic        spi_mosi,
    input  wire logic        spi_miso,
    // status
    output      logic        booted,
    output      logic        boot_err,      // bad header magic
    output      logic        busy,
    output      logic        res_valid,     // a result is available
    output      logic [7:0]  res_class,
    output      logic signed [7:0] res_score,
    output      logic [31:0] res_cycles,
    output      logic [15:0] res_count,     // inferences run
    // thumbnail read port (1-cycle latency)
    input  wire logic [11:0] thumb_raddr,
    output      logic [3:0]  thumb_rdata,
    // SDRAM (EXT_SDRAM; dq split for simulation)
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
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire logic [31:0] sd_dq_i
    /* verilator lint_on UNUSEDSIGNAL */
);

    localparam int RAW = 17;                // reader word address width
    localparam int TN  = 48;                // thumbnail side

    // ------------------------------------------------------------------ NPU core
    logic              c_start, c_busy, c_done;
    logic [7:0]        c_class;
    logic signed [7:0] c_score;
    logic [31:0]       c_cycles;
    logic              h_we;
    logic [2:0]        h_sel;
    logic [15:0]       h_addr;
    logic [WORD_W-1:0] h_wdata;
    logic              e_req, e_ack, e_rvalid;
    logic [EXT_AW-1:0] e_addr;
    logic [15:0]       e_len;
    logic [WORD_W-1:0] e_rdata;
    /* verilator lint_off UNUSEDSIGNAL */
    logic              c_paused;
    logic [IMEM_AW-1:0] c_pc;
    logic [31:0]       c_icycles;
    logic [WORD_W-1:0] c_dbg;
    /* verilator lint_on UNUSEDSIGNAL */

    npu_core u_core (
        .clk, .rst, .start(c_start), .step_mode(1'b0), .resume(1'b0),
        .busy(c_busy), .paused(c_paused), .done(c_done), .pc(c_pc),
        .result_class(c_class), .result_score(c_score), .cycles(c_cycles), .instr_cycles(c_icycles),
        .host_we(h_we), .host_sel(h_sel), .host_addr(h_addr), .host_wdata(h_wdata),
        .dbg_addr('0), .dbg_rdata(c_dbg),
        .ext_req(e_req), .ext_addr(e_addr), .ext_len(e_len), .ext_ack(e_ack),
        .ext_rvalid(e_rvalid), .ext_rdata(e_rdata));

    // ------------------------------------------------------------------ flash reader
    // owned by the boot loader until the program is loaded, then by the core
    logic             r_req, r_ack, r_rvalid;
    logic [RAW-1:0]   r_addr;
    logic [15:0]      r_len;
    logic [63:0]      r_rdata;
    logic             b_req;                // boot loader request
    logic [RAW-1:0]   b_addr;
    logic [15:0]      b_len;
    logic             ld_own;               // the loader owns the reader

    spi_flash_reader #(.AW(RAW), .FLASH_BASE(FLASH_BASE), .CLK_DIV(SPI_DIV)) u_reader (
        .clk, .rst, .req(r_req), .addr(r_addr), .len(r_len), .ack(r_ack),
        .rvalid(r_rvalid), .rdata(r_rdata),
        .spi_cs_n, .spi_clk, .spi_mosi, .spi_miso);

    // SDRAM: written by the boot copy, read by the core
    /* verilator lint_off UNUSEDSIGNAL */  // half of these are unused in each configuration
    logic        m_ready, m_cal_err, m_rd_ack, m_rvalid, m_wr_req, m_wr_ack;
    logic [63:0] m_rdata, m_wr_data;
    logic [16:0] m_wr_addr;
    logic [31:0] ext_words;
    /* verilator lint_on UNUSEDSIGNAL */

    if (EXT_SDRAM) begin : g_sdram
        sdram_ctrl #(.CLK_HZ(CLK_HZ)) u_sdram (
            .clk, .rst, .ready(m_ready), .cal_err(m_cal_err),
            .rd_req(e_req && !ld_own), .rd_addr(17'(e_addr)), .rd_len(e_len), .rd_ack(m_rd_ack),
            .rvalid(m_rvalid), .rdata(m_rdata),
            .wr_req(m_wr_req), .wr_addr(m_wr_addr), .wr_data(m_wr_data), .wr_ack(m_wr_ack),
            .sd_cke, .sd_cs_n, .sd_ras_n, .sd_cas_n, .sd_we_n, .sd_ba, .sd_addr, .sd_dqm,
            .sd_dq_o, .sd_dq_oe, .sd_dq_i);
        always_comb begin
            r_req    = b_req;
            r_addr   = b_addr;
            r_len    = b_len;
            e_ack    = m_rd_ack;
            e_rvalid = m_rvalid;
            e_rdata  = m_rdata;
        end
    end else begin : g_flash
        assign {m_ready, m_cal_err, m_rd_ack, m_rvalid, m_wr_ack, m_rdata} = '0;
        assign {sd_cke, sd_cs_n, sd_ras_n, sd_cas_n, sd_we_n, sd_ba, sd_addr, sd_dqm} = '1;
        assign {sd_dq_o, sd_dq_oe} = '0;
        always_comb begin
            r_req    = ld_own ? b_req : e_req;
            r_addr   = ld_own ? b_addr : RAW'(e_addr);
            r_len    = ld_own ? b_len : e_len;
            e_ack    = !ld_own && r_ack;
            e_rvalid = !ld_own && r_rvalid;
            e_rdata  = r_rdata;
        end
    end

    // ------------------------------------------------------------------ UART
    localparam int DIV = (CLK_HZ + BAUD / 2) / BAUD;
    logic       rx_v, tx_v, tx_rdy;
    logic [7:0] rx_d, tx_d;
    uart_rx #(.DIV(DIV)) u_rx (.clk, .rst, .rx(uart_rx), .valid(rx_v), .data(rx_d));
    uart_tx #(.DIV(DIV)) u_tx (.clk, .rst, .valid(tx_v), .data(tx_d), .ready(tx_rdy), .tx(uart_tx));

    // ------------------------------------------------------------------ boot header
    logic [15:0] n_beats, in_base, in_words;
    logic [6:0]  img_w;
    logic [2:0]  img_c;

    // ------------------------------------------------------------------ image byte sink
    // bytes in HWC order -> activation words at in_base, plus the thumbnail
    logic        s_v;                       // a byte arrives
    logic [7:0]  s_d;
    logic [2:0]  s_byte;                    // byte within the word
    logic [15:0] s_word;                    // words written
    logic [55:0] s_buf;                     // bytes 0..6 of the current word
    logic        s_wr;                      // write the completed word this cycle
    logic [WORD_W-1:0] s_wdata;
    logic [1:0]  t_c;                       // channel of this byte
    logic [6:0]  t_x, t_y;                  // pixel of this byte
    logic [9:0]  t_acc;                     // gray accumulator: r + 2g + b
    logic [11:0] t_ptr;                     // next thumbnail pixel
    logic        t_we;
    logic [3:0]  t_wdata;
    logic        s_done;                    // in_words words written (pulse)

    sdpram #(.W(4), .D(TN * TN)) u_thumb (
        .clk, .we(t_we), .waddr(t_ptr), .wdata(t_wdata), .raddr(thumb_raddr), .rdata(thumb_rdata));

    always_comb begin
        logic [9:0] acc;
        acc = (t_c == 0) ? 10'(s_d) : t_acc + ((t_c == 1) ? {1'b0, s_d, 1'b0} : 10'(s_d));
        t_wdata = (img_c == 3'd1) ? s_d[7:4] : 4'(acc >> 6);
        t_we    = s_v && (t_c == 2'(img_c - 3'd1)) && !t_x[0] && !t_y[0] && t_x < 7'(2 * TN)
                  && t_y < 7'(2 * TN);
        s_wr    = s_v && s_byte == 3'd7;
        s_wdata = {s_d, s_buf};
    end

    logic s_reset;                          // start a new image
    always_ff @(posedge clk) begin
        s_done <= 1'b0;
        if (s_reset) begin
            s_byte <= '0; s_word <= '0;
            t_c <= '0; t_x <= '0; t_y <= '0; t_ptr <= '0;
        end else if (s_v) begin
            s_byte <= s_byte + 3'd1;
            if (s_byte != 3'd7) s_buf[8 * s_byte +: 8] <= s_d;
            if (s_wr) begin
                s_word <= s_word + 16'd1;
                if (s_word + 16'd1 == in_words) s_done <= 1'b1;
            end
            t_acc <= (t_c == 0) ? 10'(s_d) : t_acc + ((t_c == 1) ? {1'b0, s_d, 1'b0} : 10'(s_d));
            if (t_we) t_ptr <= t_ptr + 12'd1;
            if (t_c == 2'(img_c - 3'd1)) begin
                t_c <= '0;
                if (t_x == img_w - 7'd1) begin
                    t_x <= '0;
                    t_y <= t_y + 7'd1;
                end else begin
                    t_x <= t_x + 7'd1;
                end
            end else begin
                t_c <= t_c + 2'd1;
            end
        end
    end

    // ------------------------------------------------------------------ control
    typedef enum logic [3:0] {
        B_HDR, B_HDR_WAIT, B_COPY_REQ, B_COPY, B_PROG_REQ, B_PROG, B_PROG_WR, B_IMG,
        B_IMG_BYTES, B_RUN, B_WAIT, B_IDLE, B_UART_IMG, B_ERR
    } bstate_e;
    bstate_e bs;
    logic [15:0] prog_words;
    logic        hdr2;                      // header word 1 is next
    logic [63:0] w_buf;                     // flash word being unpacked
    logic [2:0]  u_byte;                    // byte of w_buf to feed (flash image)
    logic        u_v;                       // w_buf holds bytes to feed
    logic [15:0] beat;                      // next instruction beat
    logic        hi_beat;                   // writing the word's second beat
    logic        reply;                     // send a UART reply when the run finishes
    logic [2:0]  tx_idx;                    // reply byte being sent (0 = none)
    logic [55:0] tx_msg;

    assign booted = (bs == B_RUN) || (bs == B_WAIT) || (bs == B_IDLE) || (bs == B_UART_IMG);
    assign boot_err = (bs == B_ERR) || m_cal_err;
    assign busy   = c_busy;
    assign ld_own = !booted;

    // host port: program beats during boot, image words from the sink
    always_comb begin
        h_we    = 1'b0;
        h_sel   = 3'd0;
        h_addr  = '0;
        h_wdata = '0;
        if (bs == B_PROG_WR) begin
            h_we    = 1'b1;
            h_sel   = 3'd0;
            h_addr  = beat;
            h_wdata = 64'(hi_beat ? w_buf[63:32] : w_buf[31:0]);
        end else if (s_wr) begin
            h_we    = 1'b1;
            h_sel   = 3'd2;
            h_addr  = in_base + s_word;
            h_wdata = s_wdata;
        end
        // the image byte stream: flash words byte by byte, or UART bytes
        s_v = (bs == B_IMG_BYTES && u_v) || (bs == B_UART_IMG && rx_v);
        s_d = (bs == B_UART_IMG) ? rx_d : w_buf[8 * u_byte +: 8];
        tx_v = (tx_idx != 0) && tx_rdy;
        tx_d = tx_msg[7:0];
    end

    always_ff @(posedge clk) begin
        c_start <= 1'b0;
        s_reset <= 1'b0;
        if (tx_v) begin
            tx_msg <= tx_msg >> 8;
            tx_idx <= tx_idx - 3'd1;
        end
        if (c_done) begin
            res_valid  <= 1'b1;
            res_class  <= c_class;
            res_score  <= c_score;
            res_cycles <= c_cycles;
            res_count  <= res_count + 16'd1;
            if (reply) begin
                reply  <= 1'b0;
                tx_msg <= {c_cycles, c_score, c_class, 8'hA5};
                tx_idx <= 3'd7;
            end
        end

        if (rst) begin
            bs        <= B_HDR;
            b_req     <= 1'b0;
            res_valid <= 1'b0;
            res_count <= '0;
            reply     <= 1'b0;
            tx_idx    <= '0;
            u_v       <= 1'b0;
            m_wr_req  <= 1'b0;
        end else begin
            unique case (bs)
                // ---- boot: header word
                B_HDR: begin
                    b_req  <= 1'b1;
                    b_addr <= RAW'(BOOT_W);
                    b_len  <= 16'd2;
                    hdr2   <= 1'b0;
                    if (b_req && r_ack) begin
                        b_req <= 1'b0;
                        bs    <= B_HDR_WAIT;
                    end
                end

                B_HDR_WAIT: if (r_rvalid && hdr2) begin
                    ext_words <= r_rdata[31:0];
                    bs        <= EXT_SDRAM ? B_COPY_REQ : B_PROG_REQ;
                end else if (r_rvalid) begin
                    hdr2       <= 1'b1;
                    n_beats    <= r_rdata[15:0];
                    in_base    <= r_rdata[31:16];
                    in_words   <= r_rdata[47:32];
                    img_w      <= r_rdata[54:48];
                    img_c      <= r_rdata[57:55];
                    prog_words <= 16'((r_rdata[15:0] + 16'd1) >> 1);
                    if (r_rdata[63:58] != 6'b101101) bs <= B_ERR;
                end

                // ---- (EXT_SDRAM) weights/params: flash -> SDRAM, one word at a time
                B_COPY_REQ: if (m_ready) begin
                    b_req     <= 1'b1;
                    b_addr    <= '0;
                    b_len     <= 16'(ext_words);
                    m_wr_addr <= '0;
                    bs        <= B_COPY;
                end

                B_COPY: begin
                    if (b_req && r_ack) b_req <= 1'b0;
                    if (r_rvalid) begin                 // flash words are >100 cycles apart
                        m_wr_req  <= 1'b1;
                        m_wr_data <= r_rdata;
                    end
                    if (m_wr_req && m_wr_ack) begin
                        m_wr_req  <= 1'b0;
                        m_wr_addr <= m_wr_addr + 17'd1;
                        if (32'(m_wr_addr) + 32'd1 == ext_words) bs <= B_PROG_REQ;
                    end
                end

                B_PROG_REQ: begin
                    b_req  <= 1'b1;
                    b_addr <= RAW'(BOOT_W + 2);
                    b_len  <= prog_words;
                    beat   <= '0;
                    bs     <= B_PROG;
                end

                // ---- program words: two instruction beats each
                B_PROG: begin
                    if (b_req && r_ack) b_req <= 1'b0;
                    if (r_rvalid) begin
                        w_buf   <= r_rdata;
                        hi_beat <= 1'b0;
                        bs      <= B_PROG_WR;
                    end
                end

                B_PROG_WR: begin
                    beat <= beat + 16'd1;
                    if (!hi_beat && beat + 16'd1 < n_beats) begin
                        hi_beat <= 1'b1;
                    end else if (beat + 16'd1 >= n_beats) begin
                        bs <= B_IMG;                    // program complete
                    end else begin
                        bs <= B_PROG;
                    end
                end

                // ---- demo image (also after `run`): words, fed to the sink byte by byte
                B_IMG: begin
                    b_req   <= 1'b1;
                    b_addr  <= RAW'(BOOT_W + 2) + RAW'(prog_words);
                    b_len   <= in_words;
                    s_reset <= 1'b1;
                    bs      <= B_IMG_BYTES;
                end

                B_IMG_BYTES: begin
                    if (b_req && r_ack) b_req <= 1'b0;
                    if (r_rvalid) begin
                        w_buf  <= r_rdata;
                        u_byte <= '0;
                        u_v    <= 1'b1;
                    end else if (u_v) begin
                        u_byte <= u_byte + 3'd1;
                        if (u_byte == 3'd7) u_v <= 1'b0;
                    end
                    if (s_done) bs <= B_RUN;
                end

                // ---- inference
                B_RUN: if (!c_busy) begin
                    c_start <= 1'b1;
                    bs      <= B_WAIT;
                end

                B_WAIT: if (c_done) bs <= B_IDLE;

                B_IDLE: begin
                    if (run) begin
                        bs <= B_IMG;                    // reloads the image through the loader
                    end else if (rx_v && rx_d == 8'h49 && tx_idx == 0) begin
                        s_reset <= 1'b1;
                        reply   <= 1'b1;
                        bs      <= B_UART_IMG;
                    end
                end

                B_UART_IMG: if (s_done) bs <= B_RUN;

                B_ERR: ;

                default: bs <= B_HDR;
            endcase
        end
    end


endmodule

`default_nettype wire
