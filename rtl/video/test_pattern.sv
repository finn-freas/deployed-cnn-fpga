`default_nettype none

module test_pattern (
    input  wire logic        clk,
    input  wire logic        rst,
    input  wire logic        frame_start,
    input  wire logic [10:0] x,
    input  wire logic [9:0]  y,
    output      logic [7:0]  r,
    output      logic [7:0]  g,
    output      logic [7:0]  b
);

    localparam int W = 640, H = 480, SQ = 32;

    logic [10:0] sq_x;
    logic [9:0]  sq_y;
    logic        dir_x, dir_y;

    always_ff @(posedge clk) begin
        if (rst) begin
            sq_x <= 11'd100; sq_y <= 10'd100;
            dir_x <= 1'b1;   dir_y <= 1'b1;
        end else if (frame_start) begin
            sq_x <= dir_x ? sq_x + 11'd3 : sq_x - 11'd3;
            sq_y <= dir_y ? sq_y + 10'd2 : sq_y - 10'd2;
            if (dir_x && sq_x >= 11'(W - SQ - 3)) dir_x <= 1'b0;
            if (!dir_x && sq_x <= 11'd3)          dir_x <= 1'b1;
            if (dir_y && sq_y >= 10'(H - SQ - 2)) dir_y <= 1'b0;
            if (!dir_y && sq_y <= 10'd2)          dir_y <= 1'b1;
        end
    end

    logic in_sq;
    logic [2:0] bar;

    always_comb begin
        in_sq = (x >= sq_x) && (x < sq_x + 11'(SQ)) && (y >= sq_y) && (y < sq_y + 10'(SQ));
        unique case (x[9:0] / 10'd80)
            10'd0:   bar = 3'b111;
            10'd1:   bar = 3'b110;
            10'd2:   bar = 3'b011;
            10'd3:   bar = 3'b010;
            10'd4:   bar = 3'b101;
            10'd5:   bar = 3'b100;
            10'd6:   bar = 3'b001;
            default: bar = 3'b000;
        endcase
        if (in_sq) {r, g, b} = {8'hFF, 8'hFF, 8'hFF};
        else       {r, g, b} = {{8{bar[2]}} & 8'hC0, {8{bar[1]}} & 8'hC0, {8{bar[0]}} & 8'hC0};
    end

endmodule

`default_nettype wire
