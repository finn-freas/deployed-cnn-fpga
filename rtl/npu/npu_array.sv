// npu_array: N x N output-stationary systolic array with input skew and shadow outputs.
//
// All rows' A values and all columns' B values for reduction index k are presented in
// the same cycle; internally row r is delayed r cycles and column c is delayed c cycles,
// so A[r][k] and B[k][c] meet in PE(r,c). Valid and last travel with A along each row.
// A value injected at cycle t is processed by PE(r,c) in cycle t+r+c, so a group's last
// value moves each PE's finished sum into its shadow register as it passes, and the next
// group can follow right behind it: no flush is needed.
//
// Depthwise mode (dw = 1; PE(r,c) = output pixel r x channel lane c): only the row skew
// remains. Valid/last reach every PE of row r at once (r cycles after injection), B moves
// down without the column skew, and every PE takes byte c of the one activation word dw_a
// in that cycle, so row r works r cycles after row 0 on whatever word is then on dw_a.

`default_nettype none

module npu_array
    import npu_pkg::*;
(
    input  wire logic                     clk,
    input  wire logic                     rst,
    input  wire logic                     valid,
    input  wire logic                     last,
    input  wire logic signed [DATA_W-1:0] a [N],   // a[r]: row r operand
    input  wire logic signed [DATA_W-1:0] b [N],   // b[c]: column c operand
    input  wire logic                     dw,
    input  wire logic [WORD_W-1:0]        dw_a,      // depthwise: activation word, all rows
    output      logic signed [ACC_W-1:0]  shadow [N][N]
);

    // ---- input skew: row r / column c delayed by r / c cycles
    logic signed [DATA_W-1:0] a_sk [N];
    logic signed [DATA_W-1:0] b_sk [N];
    logic                     v_sk [N];
    logic                     l_sk [N];

    for (genvar i = 0; i < N; i++) begin : g_skew
        if (i == 0) begin : g_direct
            assign a_sk[i] = a[i];
            assign b_sk[i] = b[i];
            assign v_sk[i] = valid;
            assign l_sk[i] = last;
        end else begin : g_delay
            logic signed [DATA_W-1:0] a_d [i];
            logic signed [DATA_W-1:0] b_d [i];
            logic                     v_d [i];
            logic                     l_d [i];
            always_ff @(posedge clk) begin
                a_d[0] <= a[i];
                b_d[0] <= b[i];
                v_d[0] <= valid;
                l_d[0] <= last;
                for (int s = 1; s < i; s++) begin
                    a_d[s] <= a_d[s-1];
                    b_d[s] <= b_d[s-1];
                    v_d[s] <= v_d[s-1];
                    l_d[s] <= l_d[s-1];
                end
            end
            assign a_sk[i] = a_d[i-1];
            assign b_sk[i] = dw ? b[i] : b_d[i-1];      // depthwise: no column skew
            assign v_sk[i] = v_d[i-1];
            assign l_sk[i] = l_d[i-1];
        end
    end

    // ---- PE grid: a_h/v_h/l_h[r][c] enter PE(r,c) from the left, b_v[r][c] from above
    logic signed [DATA_W-1:0] a_h [N][N+1];
    logic                     v_h [N][N+1];
    logic                     l_h [N][N+1];
    logic signed [DATA_W-1:0] b_v [N+1][N];

    for (genvar r = 0; r < N; r++) begin : g_row
        assign a_h[r][0] = a_sk[r];
        assign v_h[r][0] = v_sk[r];
        assign l_h[r][0] = l_sk[r];
    end
    for (genvar c = 0; c < N; c++) begin : g_col
        assign b_v[0][c] = b_sk[c];
    end

    for (genvar r = 0; r < N; r++) begin : g_r
        for (genvar c = 0; c < N; c++) begin : g_c
            npu_pe u_pe (
                .clk, .rst,
                .v_in  (dw ? v_h[r][0] : v_h[r][c]),
                .l_in  (dw ? l_h[r][0] : l_h[r][c]),
                .a_in  (dw ? DATA_W'(dw_a >> (DATA_W * c)) : a_h[r][c]),
                .b_in  (b_v[r][c]),
                .v_out (v_h[r][c+1]),
                .l_out (l_h[r][c+1]),
                .a_out (a_h[r][c+1]),
                .b_out (b_v[r+1][c]),
                .shadow(shadow[r][c])
            );
        end
    end

    // Outputs off the right/bottom edges are unused.
    logic unused;
    always_comb begin
        unused = 1'b0;
        for (int i = 0; i < N; i++)
            unused = unused ^ v_h[i][N] ^ l_h[i][N] ^ (^a_h[i][N]) ^ (^b_v[N][i]);
    end

endmodule

`default_nettype wire
