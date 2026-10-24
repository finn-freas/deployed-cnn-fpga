// spi_flash_model: behavioural 8 MiB SPI NOR flash, READ (0x03) only, SPI mode 0
// (simulation). Contents are loaded 8 bytes at a time (little-endian) through a write
// port. Samples MOSI on SCLK rising edges and changes MISO after falling edges.

`default_nettype none
/* verilator lint_off SYNCASYNCNET */   // CS# is both an async reset and a sampled level here

module spi_flash_model (
    input  wire logic        cs_n,
    input  wire logic        sclk,
    input  wire logic        mosi,
    output      logic        miso,
    // contents loading (independent clock)
    input  wire logic        load_clk,
    input  wire logic        load_we,
    input  wire logic [22:0] load_addr,     // byte address, 8-aligned
    input  wire logic [63:0] load_data
);

    logic [7:0]  mem [1 << 23];
    logic [5:0]  nbits;               // bits received since CS# fell (saturates at 32)
    logic [31:0] cmd;                 // opcode + 24-bit address
    logic        started;             // data phase running
    logic [22:0] ptr;
    logic [2:0]  out_bit;

    always_ff @(posedge load_clk)
        if (load_we) for (int b = 0; b < 8; b++) mem[load_addr + 23'(b)] <= load_data[8 * b +: 8];

    always_ff @(posedge sclk or posedge cs_n) begin
        if (cs_n) begin
            nbits <= '0;
        end else if (nbits < 6'd32) begin
            cmd   <= {cmd[30:0], mosi};
            nbits <= nbits + 6'd1;
        end
    end

    // after the 32nd rising edge, a data bit comes out after each falling edge
    always_ff @(negedge sclk or posedge cs_n) begin
        if (cs_n) begin
            miso    <= 1'b0;
            started <= 1'b0;
            out_bit <= 3'd7;
        end else if (nbits == 6'd32) begin
            logic [22:0] a;
            a = started ? ptr : cmd[22:0];
            miso    <= mem[a][out_bit];
            started <= 1'b1;
            out_bit <= out_bit - 3'd1;
            ptr     <= (out_bit == 3'd0) ? a + 23'd1 : a;
        end
    end

endmodule

/* verilator lint_on SYNCASYNCNET */
`default_nettype wire
