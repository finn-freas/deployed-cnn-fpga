`default_nettype none

module reset_sync #(
    parameter int STAGES = 3
) (
    input  wire logic clk,
    input  wire logic rst_async_in,
    output      logic rst_out
);

    logic [STAGES-1:0] sync;

    always_ff @(posedge clk or posedge rst_async_in) begin
        if (rst_async_in) sync <= '1;
        else              sync <= {sync[STAGES-2:0], 1'b0};
    end

    assign rst_out = sync[STAGES-1];

endmodule

`default_nettype wire
