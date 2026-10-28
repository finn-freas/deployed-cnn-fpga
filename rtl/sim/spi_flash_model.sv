`default_nettype none
/* verilator lint_off SYNCASYNCNET */

module spi_flash_model (
    input  wire logic        cs_n,
    input  wire logic        sclk,
    input  wire logic        mosi,
    output      logic        miso,
    input  wire logic        load_clk,
    input  wire logic        load_we,
    input  wire logic [22:0] load_addr,
    input  wire logic [63:0] load_data
);

    logic [7:0]  mem [1 << 23];
    logic [5:0]  nbits;
    logic [31:0] cmd;
    logic        started;
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

    always_ff @(negedge sclk or posedge cs_n) begin
        if (cs_n) begin
            miso    <= 1'b0;
            started <= 1'b0;
            out_bit <= 3'd7;
        end else if (nbits == 6'd32 && cmd[31:24] == 8'h03) begin
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
