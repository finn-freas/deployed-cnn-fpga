// sdram_model: behavioural 64 Mbit x32 SDR SDRAM (4 banks x 2K rows x 256 columns), for
// simulation. Commands are sampled on the SDRAM clock's rising edge. MODE REGISTER sets the
// CAS latency; reads are bursts of 8, writes single-location (as sdram_ctrl programs it).
// Read beat i is driven at edge READ+CAS-1+i, so it is valid at edge READ+CAS+i.
// Accessing a bank without an open row is an error.

`default_nettype none

module sdram_model (
    input  wire logic        clk,           // SDRAM clock
    input  wire logic        cke,
    input  wire logic        cs_n,
    input  wire logic        ras_n,
    input  wire logic        cas_n,
    input  wire logic        we_n,
    input  wire logic [1:0]  ba,
    input  wire logic [10:0] addr,
    input  wire logic [3:0]  dqm,
    input  wire logic [31:0] dq_i,          // driven by the controller
    output      logic [31:0] dq_o,
    output      logic        dq_oe
);
    logic [31:0] mem [1 << 21];
    logic [10:0] row [4];
    logic [3:0]  open_;
    logic [2:0]  cas_l;              // set by MODE REGISTER before any read
    // pending read beats: q[k] is driven k+1 edges from now
    typedef struct packed { logic v; logic [1:0] b; logic [10:0] r; logic [7:0] c; } beat_t;
    beat_t q [16];

    always_ff @(posedge clk) begin
        for (int i = 0; i < 15; i++) q[i] <= q[i + 1];
        q[15] <= '0;
        dq_oe <= q[0].v;
        if (q[0].v) dq_o <= mem[{q[0].b, q[0].r, q[0].c}];
        if (cke && !cs_n) begin
            unique case ({ras_n, cas_n, we_n})
                3'b011: begin                                   // ACTIVATE
                    row[ba]   <= addr;
                    open_[ba] <= 1'b1;
                end
                3'b101: begin                                   // READ (burst of 8)
                    if (!open_[ba]) $error("sdram_model: READ to closed bank %0d", ba);
                    for (int i = 0; i < 8; i++)
                        q[int'(cas_l) - 2 + i] <= '{1'b1, ba, row[ba], {addr[7:3], 3'(i)}};
                    if (addr[10]) open_[ba] <= 1'b0;
                end
                3'b100: begin                                   // WRITE (single)
                    if (!open_[ba]) $error("sdram_model: WRITE to closed bank %0d", ba);
                    for (int k = 0; k < 4; k++)
                        if (!dqm[k]) mem[{ba, row[ba], addr[7:0]}][8 * k +: 8] <= dq_i[8 * k +: 8];
                    if (addr[10]) open_[ba] <= 1'b0;
                end
                3'b010: begin                                   // PRECHARGE
                    if (addr[10]) open_ <= '0;
                    else          open_[ba] <= 1'b0;
                end
                3'b000: cas_l <= addr[6:4];                     // MODE REGISTER
                default: ;                                      // NOP, REFRESH
            endcase
        end
    end
endmodule

`default_nettype wire
