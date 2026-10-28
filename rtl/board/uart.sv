`default_nettype none

module uart_rx #(
    parameter int DIV = 25
) (
    input  wire logic       clk,
    input  wire logic       rst,
    input  wire logic       rx,
    output      logic       valid,
    output      logic [7:0] data
);
    localparam int CW = $clog2(DIV * 3 / 2 + 1);
    logic [1:0]    sync;
    logic          busy;
    logic [CW-1:0] cnt;
    logic [3:0]    bitn;

    always_ff @(posedge clk) begin
        sync  <= {sync[0], rx};
        valid <= 1'b0;
        if (rst) begin
            busy <= 1'b0;
            sync <= 2'b11;
        end else if (!busy) begin
            if (!sync[1]) begin
                busy <= 1'b1;
                cnt  <= CW'(DIV * 3 / 2 - 1);
                bitn <= '0;
            end
        end else if (cnt != 0) begin
            cnt <= cnt - 1'b1;
        end else begin
            cnt <= CW'(DIV - 1);
            if (bitn == 4'd8) begin
                busy  <= 1'b0;
                valid <= sync[1];
            end else begin
                data <= {sync[1], data[7:1]};
                bitn <= bitn + 4'd1;
            end
        end
    end
endmodule

module uart_tx #(
    parameter int DIV = 25
) (
    input  wire logic       clk,
    input  wire logic       rst,
    input  wire logic       valid,
    input  wire logic [7:0] data,
    output      logic       ready,
    output      logic       tx
);
    localparam int CW = $clog2(DIV);
    logic [8:0]    sh;
    logic [3:0]    bitn;
    logic [CW-1:0] cnt;

    assign ready = (bitn == 0);

    always_ff @(posedge clk) begin
        if (rst) begin
            bitn <= '0;
            tx   <= 1'b1;
        end else if (bitn == 0) begin
            if (valid) begin
                sh   <= {data, 1'b0};
                bitn <= 4'd11;
                cnt  <= '0;
            end
        end else if (cnt != 0) begin
            cnt <= cnt - 1'b1;
        end else begin
            cnt  <= CW'(DIV - 1);
            if (bitn != 4'd1) begin
                tx <= sh[0];
                sh <= {1'b1, sh[8:1]};
            end
            bitn <= bitn - 4'd1;
        end
    end
endmodule

`default_nettype wire
