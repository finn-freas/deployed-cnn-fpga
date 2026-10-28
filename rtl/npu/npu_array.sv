`default_nettype none

module npu_array
    import npu_pkg::*;
(
    input  wire logic                     clk,
    input  wire logic                     rst,
    input  wire logic                     valid,
    input  wire logic                     last,
    input  wire logic signed [DATA_W-1:0] a [N],
    input  wire logic signed [DATA_W-1:0] b [N],
    input  wire logic                     dw,
    input  wire logic [WORD_W-1:0]        dw_a,
    output      logic signed [ACC_W-1:0]  shadow [N][N]
);

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

    logic unused;
    always_comb begin
        unused = 1'b0;
        for (int i = 0; i < N; i++) unused = unused ^ (^b_v[N][i]);
    end

endmodule

`default_nettype wire
