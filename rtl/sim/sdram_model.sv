`default_nettype none

module sdram_model (
    input  wire logic        clk,
    input  wire logic        cke,
    input  wire logic        cs_n,
    input  wire logic        ras_n,
    input  wire logic        cas_n,
    input  wire logic        we_n,
    input  wire logic [1:0]  ba,
    input  wire logic [10:0] addr,
    input  wire logic [3:0]  dqm,
    input  wire logic [31:0] dq_i,
    output      logic [31:0] dq_o,
    output      logic        dq_oe
);
    logic [31:0] mem [1 << 21];
    logic [10:0] row [4];
    logic [3:0]  open_;
    logic [2:0]  cas_l;
    typedef struct packed { logic v; logic [1:0] b; logic [10:0] r; logic [7:0] c; } beat_t;
    beat_t q [16];

    always_ff @(posedge clk) begin
        for (int i = 0; i < 15; i++) q[i] <= q[i + 1];
        q[15] <= '0;
        dq_oe <= q[0].v;
        if (q[0].v) dq_o <= mem[{q[0].b, q[0].r, q[0].c}];
        if (cke && !cs_n) begin
            unique case ({ras_n, cas_n, we_n})
                3'b011: begin
                    row[ba]   <= addr;
                    open_[ba] <= 1'b1;
                end
                3'b101: begin
                    if (!open_[ba]) $error("sdram_model: READ to closed bank %0d", ba);
                    for (int i = 0; i < 8; i++)
                        q[int'(cas_l) - 2 + i] <= '{1'b1, ba, row[ba], {addr[7:3], 3'(i)}};
                    if (addr[10]) open_[ba] <= 1'b0;
                end
                3'b100: begin
                    if (!open_[ba]) $error("sdram_model: WRITE to closed bank %0d", ba);
                    for (int k = 0; k < 4; k++)
                        if (!dqm[k]) mem[{ba, row[ba], addr[7:0]}][8 * k +: 8] <= dq_i[8 * k +: 8];
                    if (addr[10]) open_[ba] <= 1'b0;
                end
                3'b010: begin
                    if (addr[10]) open_ <= '0;
                    else          open_[ba] <= 1'b0;
                end
                3'b000: cas_l <= addr[6:4];
                default: ;
            endcase
        end
    end
endmodule

`default_nettype wire
