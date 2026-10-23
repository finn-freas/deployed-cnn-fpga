// tmds_encoder: DVI 1.0 TMDS 8b/10b encoder for one channel.
// Video period (de=1): transition-minimized, DC-balanced data word.
// Blanking (de=0): one of four control tokens selected by {c1, c0}.
// Latency: 1 cycle. Bit 0 of `q` is transmitted first.

`default_nettype none

module tmds_encoder (
    input  wire logic       clk,
    input  wire logic       rst,
    input  wire logic       de,
    input  wire logic [7:0] d,
    input  wire logic       c0,
    input  wire logic       c1,
    output      logic [9:0] q
);

    // Stage 1: transition minimization (combinational)
    logic [3:0] n1_d;
    logic       use_xnor;
    logic [8:0] q_m;

    always_comb begin
        logic prev;
        n1_d = 4'($countones(d));
        use_xnor = (n1_d > 4'd4) || (n1_d == 4'd4 && !d[0]);
        prev = d[0];
        q_m[0] = prev;
        for (int i = 1; i < 8; i++) begin
            prev = use_xnor ? ~(prev ^ d[i]) : (prev ^ d[i]);
            q_m[i] = prev;
        end
        q_m[8] = ~use_xnor;
    end

    // Stage 2: DC balance using running disparity
    logic signed [4:0] cnt;        // running disparity (ones minus zeros), even, in [-8, 8]
    logic signed [4:0] cnt_next;
    logic signed [4:0] diff;       // ones(q_m[7:0]) - zeros(q_m[7:0])
    logic [9:0]        q_next;

    always_comb begin
        diff = 5'sd2 * $signed({1'b0, 4'($countones(q_m[7:0]))}) - 5'sd8;

        if (cnt == 0 || diff == 0) begin
            q_next = {~q_m[8], q_m[8], q_m[8] ? q_m[7:0] : ~q_m[7:0]};
            cnt_next = q_m[8] ? cnt + diff : cnt - diff;
        end else if ((cnt > 0 && diff > 0) || (cnt < 0 && diff < 0)) begin
            q_next = {1'b1, q_m[8], ~q_m[7:0]};
            cnt_next = cnt + (q_m[8] ? 5'sd2 : 5'sd0) - diff;
        end else begin
            q_next = {1'b0, q_m[8], q_m[7:0]};
            cnt_next = cnt - (q_m[8] ? 5'sd0 : 5'sd2) + diff;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            q   <= 10'b1101010100;
            cnt <= '0;
        end else if (de) begin
            q   <= q_next;
            cnt <= cnt_next;
        end else begin
            cnt <= '0;
            unique case ({c1, c0})
                2'b00: q <= 10'b1101010100;
                2'b01: q <= 10'b0010101011;
                2'b10: q <= 10'b0101010100;
                2'b11: q <= 10'b1010101011;
            endcase
        end
    end

endmodule

`default_nettype wire
