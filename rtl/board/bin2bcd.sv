`default_nettype none

module bin2bcd #(
    parameter int BITS   = 20,
    parameter int DIGITS = 6
) (
    input  wire logic                  clk,
    input  wire logic                  rst,
    input  wire logic                  start,
    input  wire logic [BITS-1:0]       bin,
    output      logic [4*DIGITS-1:0]   bcd,
    output      logic                  busy
);
    logic [BITS-1:0]     sh;
    logic [4*DIGITS-1:0] acc;
    logic [$clog2(BITS+1)-1:0] n;

    assign busy = (n != 0);

    always_ff @(posedge clk) begin
        if (rst) begin
            n <= '0;
        end else if (start) begin
            sh  <= bin;
            acc <= '0;
            n   <= BITS[$clog2(BITS+1)-1:0];
        end else if (n != 0) begin
            logic [4*DIGITS-1:0] a;
            a = acc;
            for (int d = 0; d < DIGITS; d++)
                if (a[4*d +: 4] >= 4'd5) a[4*d +: 4] = a[4*d +: 4] + 4'd3;
            acc <= {a[4*DIGITS-2:0], sh[BITS-1]};
            sh  <= sh << 1;
            n   <= n - 1'b1;
            if (n == 1) bcd <= {a[4*DIGITS-2:0], sh[BITS-1]};
        end
    end
endmodule

`default_nettype wire
