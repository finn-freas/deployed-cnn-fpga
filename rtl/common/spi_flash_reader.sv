// spi_flash_reader: serves the NPU's external-store burst port from a standard SPI NOR
// flash (READ 0x03, SPI mode 0, 24-bit addresses).
//
// A request for `len` 64-bit words at word address `addr` becomes one READ command at
// byte address FLASH_BASE + addr*8; bytes arrive MSB-first and are packed little-endian
// (first byte = bits 7:0). SCLK = clk / (2*CLK_DIV). Mode 0: MOSI changes while SCLK is
// low (on our falling edge), the flash samples on the rising edge; the flash changes MISO
// after the falling edge and we sample it as we raise SCLK.

`default_nettype none

module spi_flash_reader #(
    parameter int          AW         = 16,
    parameter logic [23:0] FLASH_BASE = 24'h40_0000,   // 4 MiB: clear of the bitstream
    parameter int          CLK_DIV    = 1
) (
    input  wire logic          clk,
    input  wire logic          rst,
    // burst port
    input  wire logic          req,
    input  wire logic [AW-1:0] addr,
    input  wire logic [15:0]   len,
    output      logic          ack,
    output      logic          rvalid,
    output      logic [63:0]   rdata,
    // SPI
    output      logic          spi_cs_n,
    output      logic          spi_clk,
    output      logic          spi_mosi,
    input  wire logic          spi_miso
);

    typedef enum logic [1:0] {S_IDLE, S_CMD, S_DATA, S_END} state_e;
    state_e state;

    logic [31:0] shift_out;           // remaining command/address bits, MSB first
    logic [5:0]  bits_left;           // command bits not yet clocked
    logic [2:0]  bit_idx;             // bits received into the current byte
    logic [2:0]  byte_idx;            // bytes received into the current word
    logic [15:0] words_left;
    logic [6:0]  in_bits;
    localparam int DW = (CLK_DIV > 1) ? $clog2(CLK_DIV) : 1;
    logic [DW-1:0] div;
    logic        tick;

    assign ack  = req && state == S_IDLE;
    assign tick = (div == '0);

    always_ff @(posedge clk) begin
        rvalid <= 1'b0;
        if (rst) begin
            state    <= S_IDLE;
            spi_cs_n <= 1'b1;
            spi_clk  <= 1'b0;
            spi_mosi <= 1'b0;
            div      <= '0;
        end else begin
            div <= tick ? DW'(CLK_DIV - 1) : div - 1'b1;
            unique case (state)
                S_IDLE: if (ack && len != 0) begin
                    // first command bit is presented now, with SCLK low
                    spi_cs_n   <= 1'b0;
                    spi_clk    <= 1'b0;
                    {spi_mosi, shift_out} <= {8'h03, FLASH_BASE + 24'({addr, 3'b000}), 1'b0};
                    bits_left  <= 6'd32;
                    words_left <= len;
                    bit_idx    <= '0;
                    byte_idx   <= '0;
                    div        <= DW'(CLK_DIV - 1);
                    state      <= S_CMD;
                end

                S_CMD: if (tick) begin
                    if (!spi_clk) begin
                        spi_clk   <= 1'b1;                 // flash samples MOSI
                        bits_left <= bits_left - 6'd1;
                    end else begin
                        spi_clk <= 1'b0;                   // falling edge
                        if (bits_left == 0) begin
                            state <= S_DATA;               // flash drives data bit 7 now
                        end else begin
                            spi_mosi  <= shift_out[31];
                            shift_out <= shift_out << 1;
                        end
                    end
                end

                S_DATA: if (tick) begin
                    if (!spi_clk) begin
                        spi_clk <= 1'b1;                   // sample MISO as SCLK rises
                        in_bits <= {in_bits[5:0], spi_miso};
                        bit_idx <= bit_idx + 3'd1;
                        if (bit_idx == 3'd7) begin
                            rdata[8 * byte_idx +: 8] <= {in_bits, spi_miso};
                            byte_idx <= byte_idx + 3'd1;
                            if (byte_idx == 3'd7) begin
                                rvalid     <= 1'b1;
                                words_left <= words_left - 16'd1;
                                if (words_left == 16'd1) state <= S_END;
                            end
                        end
                    end else begin
                        spi_clk <= 1'b0;
                    end
                end

                S_END: if (tick) begin
                    spi_clk  <= 1'b0;
                    spi_cs_n <= 1'b1;
                    state    <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
