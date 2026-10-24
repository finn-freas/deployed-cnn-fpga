// npu_array: N x N output-stationary MAC array with row skew and shadow outputs.
//
// All rows' A values and all columns' B values for reduction index k are presented in the
// same cycle. Row r is delayed r cycles: its A value and valid/last reach all N PEs of the
// row at once, while B moves down the columns one row per cycle, so A[r][k] and B[k][c]
// meet in PE(r,c) in cycle t+r. A group's last value moves each PE's finished sum into its
// shadow register as it passes, and the next group can follow right behind it.
//
// Depthwise mode (dw = 1; PE(r,c) = output pixel r x channel lane c): instead of A[r], every
// PE takes byte c of the one activation word dw_a, so row r works r cycles after row 0 on
// whatever word is then on dw_a.

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

    // ---- row skew: row r's A, valid and last delayed by r cycles
    logic signed [DATA_W-1:0] a_sk [N];
    logic                     v_sk [N];
    logic                     l_sk [N];

    for (genvar i = 0; i < N; i++) begin : g_skew
        if (i == 0) begin : g_direct
            assign a_sk[i] = a[i];
            assign v_sk[i] = valid;
            assign l_sk[i] = last;
        end else begin : g_delay
            logic signed [DATA_W-1:0] a_d [i];
            logic                     v_d [i];
            logic                     l_d [i];
            always_ff @(posedge clk) begin
                a_d[0] <= a[i];
                v_d[0] <= valid;
                l_d[0] <= last;
                for (int s = 1; s < i; s++) begin
                    a_d[s] <= a_d[s-1];
                    v_d[s] <= v_d[s-1];
                    l_d[s] <= l_d[s-1];
                end
            end
            assign a_sk[i] = a_d[i-1];
            assign v_sk[i] = v_d[i-1];
            assign l_sk[i] = l_d[i-1];
        end
    end

    // ---- PE grid: b_v[r][c] enters PE(r,c) from above
    logic signed [DATA_W-1:0] b_v [N+1][N];
    for (genvar c = 0; c < N; c++) begin : g_col
        assign b_v[0][c] = b[c];
    end

    for (genvar r = 0; r < N; r++) begin : g_r
        for (genvar c = 0; c < N; c++) begin : g_c
            npu_pe u_pe (
                .clk, .rst,
                .v_in  (v_sk[r]),
                .l_in  (l_sk[r]),
                .a_in  (dw ? DATA_W'(dw_a >> (DATA_W * c)) : a_sk[r]),
                .b_in  (b_v[r][c]),
                .b_out (b_v[r+1][c]),
                .shadow(shadow[r][c])
            );
        end
    end

    // Outputs off the bottom edge are unused.
    logic unused;
    always_comb begin
        unused = 1'b0;
        for (int i = 0; i < N; i++) unused = unused ^ (^b_v[N][i]);
    end

endmodule

`default_nettype wire
