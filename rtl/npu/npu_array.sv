// npu_array: N x N output-stationary systolic array with input skew.
//
// All rows' A values and all columns' B values for reduction index k are presented in
// the same cycle; internally row r is delayed r cycles and column c is delayed c cycles,
// so A[r][k] and B[k][c] meet in PE(r,c). Valid travels with A along each row.
// The last product reaches PE(N-1,N-1) 3*(N-1)+1 cycles after the last input
// (npu_tile_engine's FLUSH_N must cover this).

`default_nettype none

module npu_array
    import npu_pkg::*;
(
    input  wire logic                     clk,
    input  wire logic                     clear,
    input  wire logic                     valid,
    input  wire logic signed [DATA_W-1:0] a [N],   // a[r]: row r operand
    input  wire logic signed [DATA_W-1:0] b [N],   // b[c]: column c operand
    output      logic signed [ACC_W-1:0]  acc [N][N]
);

    // ---- input skew: row r / column c delayed by r / c cycles
    logic signed [DATA_W-1:0] a_sk [N];
    logic signed [DATA_W-1:0] b_sk [N];
    logic                     v_sk [N];

    for (genvar i = 0; i < N; i++) begin : g_skew
        if (i == 0) begin : g_direct
            assign a_sk[i] = a[i];
            assign b_sk[i] = b[i];
            assign v_sk[i] = valid;
        end else begin : g_delay
            logic signed [DATA_W-1:0] a_d [i];
            logic signed [DATA_W-1:0] b_d [i];
            logic                     v_d [i];
            always_ff @(posedge clk) begin
                a_d[0] <= a[i];
                b_d[0] <= b[i];
                v_d[0] <= valid;
                for (int s = 1; s < i; s++) begin
                    a_d[s] <= a_d[s-1];
                    b_d[s] <= b_d[s-1];
                    v_d[s] <= v_d[s-1];
                end
            end
            assign a_sk[i] = a_d[i-1];
            assign b_sk[i] = b_d[i-1];
            assign v_sk[i] = v_d[i-1];
        end
    end

    // ---- PE grid: a_h[r][c] / v_h[r][c] enter PE(r,c) from the left, b_v[r][c] from above
    logic signed [DATA_W-1:0] a_h [N][N+1];
    logic                     v_h [N][N+1];
    logic signed [DATA_W-1:0] b_v [N+1][N];

    for (genvar r = 0; r < N; r++) begin : g_row
        assign a_h[r][0] = a_sk[r];
        assign v_h[r][0] = v_sk[r];
    end
    for (genvar c = 0; c < N; c++) begin : g_col
        assign b_v[0][c] = b_sk[c];
    end

    for (genvar r = 0; r < N; r++) begin : g_r
        for (genvar c = 0; c < N; c++) begin : g_c
            npu_pe u_pe (
                .clk,
                .clear,
                .v_in (v_h[r][c]),
                .a_in (a_h[r][c]),
                .b_in (b_v[r][c]),
                .v_out(v_h[r][c+1]),
                .a_out(a_h[r][c+1]),
                .b_out(b_v[r+1][c]),
                .acc  (acc[r][c])
            );
        end
    end

    // Outputs off the right/bottom edges are unused.
    logic unused;
    always_comb begin
        unused = 1'b0;
        for (int i = 0; i < N; i++) unused = unused ^ v_h[i][N] ^ (^a_h[i][N]) ^ (^b_v[N][i]);
    end

endmodule

`default_nettype wire
