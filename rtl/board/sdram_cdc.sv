`default_nettype none

module sdram_cdc (
    input  wire logic        clk_s,
    input  wire logic        rst_s,
    output      logic        ready,
    output      logic        cal_err,
    output      logic [15:0] diag,
    output      logic        idle,
    input  wire logic        rd_req,
    input  wire logic [19:0] rd_addr,
    input  wire logic [15:0] rd_len,
    output      logic        rd_ack,
    output      logic        rvalid,
    output      logic [63:0] rdata,
    input  wire logic        wr_req,
    input  wire logic [19:0] wr_addr,
    input  wire logic [63:0] wr_data,
    output      logic        wr_ack,

    input  wire logic        clk_f,
    input  wire logic        rst_f,
    input  wire logic        f_ready,
    input  wire logic        f_cal_err,
    input  wire logic [15:0] f_diag,
    input  wire logic        f_idle,
    output      logic        f_rd_req,
    output      logic [19:0] f_rd_addr,
    output      logic [15:0] f_rd_len,
    input  wire logic        f_rd_ack,
    input  wire logic        f_rvalid,
    input  wire logic [63:0] f_rdata,
    output      logic        f_wr_req,
    output      logic [19:0] f_wr_addr,
    output      logic [63:0] f_wr_data,
    input  wire logic        f_wr_ack
);
    logic        s_tog, s_ack3;
    logic [2:0]  s_ack_sy;
    logic        s_wr;
    logic [19:0] s_addr;
    logic [15:0] s_len;
    logic        f_ack_tog;
    logic        f_idle_r;

    (* async_reg = "true" *) logic [2:0] s_rdy_sy, s_err_sy, s_idle_sy;
    always_ff @(posedge clk_s) begin
        s_ack_sy  <= {s_ack_sy[1:0], f_ack_tog};
        s_rdy_sy  <= {s_rdy_sy[1:0], f_ready};
        s_err_sy  <= {s_err_sy[1:0], f_cal_err};
        s_idle_sy <= {s_idle_sy[1:0], f_idle_r};
    end
    assign s_ack3  = s_ack_sy[2];
    assign ready   = s_rdy_sy[2];
    assign cal_err = s_err_sy[2];
    assign diag    = f_diag;

    wire s_free = (s_tog == s_ack3);
    assign wr_ack = s_free && ready && wr_req;
    assign rd_ack = s_free && ready && rd_req && !wr_req && rd_len != 0;

    logic [1:0] s_quiet;
    logic       r_empty;
    always_ff @(posedge clk_s) begin
        if (rst_s) begin
            s_tog   <= 1'b0;
            s_quiet <= '0;
            idle    <= 1'b0;
        end else begin
            if (wr_ack || rd_ack) begin
                s_tog  <= ~s_tog;
                s_wr   <= wr_ack;
                s_addr <= wr_ack ? wr_addr : rd_addr;
                s_len  <= rd_len;
            end
            if (wr_ack || rd_ack || !s_free) s_quiet <= '0;
            else if (s_quiet != 2'd3)        s_quiet <= s_quiet + 2'd1;
            idle <= s_free && !(wr_ack || rd_ack) && s_quiet == 2'd3 && s_idle_sy[2] && r_empty && !rvalid;
        end
    end

    (* async_reg = "true" *) logic [2:0] f_req_sy;
    logic        f_seen, f_pend, f_wr;
    always_ff @(posedge clk_f) begin
        f_req_sy <= {f_req_sy[1:0], s_tog};
        if (rst_f) begin
            f_seen    <= 1'b0;
            f_ack_tog <= 1'b0;
            f_pend    <= 1'b0;
            f_idle_r  <= 1'b0;
        end else begin
            if (!f_pend && f_req_sy[2] != f_seen) begin
                f_seen    <= f_req_sy[2];
                f_pend    <= 1'b1;
                f_wr      <= s_wr;
                f_rd_addr <= s_addr;
                f_rd_len  <= s_len;
            end
            if ((f_rd_req && f_rd_ack) || (f_wr_req && f_wr_ack)) begin
                f_pend    <= 1'b0;
                f_ack_tog <= ~f_ack_tog;
            end
            f_idle_r <= f_idle && !f_pend && f_req_sy[2] == f_seen && !(f_rd_req && f_rd_ack)
                        && !(f_wr_req && f_wr_ack);
        end
    end
    assign f_wr_addr = f_rd_addr;
    assign f_wr_data = wr_data;
    assign f_rd_req = f_pend && !f_wr;
    assign f_wr_req = f_pend && f_wr;

    localparam int AW = 4;
    (* ram_style = "block" *) logic [63:0] fifo [1 << AW];
    logic [AW:0] w_bin, w_gray, r_bin, r_gray;
    (* async_reg = "true" *) logic [AW:0] w_gray_s1, w_gray_s2;
    (* async_reg = "true" *) logic [AW:0] r_gray_f1, r_gray_f2;

    function automatic logic [AW:0] g2b(input logic [AW:0] g);
        logic [AW:0] b;
        b[AW] = g[AW];
        for (int i = AW - 1; i >= 0; i--) b[i] = b[i + 1] ^ g[i];
        return b;
    endfunction

    always_ff @(posedge clk_f) begin
        r_gray_f1 <= r_gray;
        r_gray_f2 <= r_gray_f1;
        if (f_rvalid) fifo[w_bin[AW-1:0]] <= f_rdata;
        if (rst_f) begin
            w_bin  <= '0;
            w_gray <= '0;
        end else if (f_rvalid) begin
            w_bin  <= w_bin + 1'b1;
            w_gray <= (w_bin + 1'b1) ^ ((w_bin + 1'b1) >> 1);
        end
    end
`ifndef SYNTHESIS
    always_ff @(posedge clk_f)
        if (!rst_f && f_rvalid && (w_bin - g2b(r_gray_f2)) == (AW+1)'(1 << AW))
            $error("sdram_cdc: read FIFO overflow");
`endif

    always_ff @(posedge clk_s) begin
        w_gray_s1 <= w_gray;
        w_gray_s2 <= w_gray_s1;
        rvalid    <= 1'b0;
        if (rst_s) begin
            r_bin  <= '0;
            r_gray <= '0;
        end else if (!r_empty) begin
            rvalid <= 1'b1;
            rdata  <= fifo[r_bin[AW-1:0]];
            r_bin  <= r_bin + 1'b1;
            r_gray <= (r_bin + 1'b1) ^ ((r_bin + 1'b1) >> 1);
        end
    end
    assign r_empty = (r_gray == w_gray_s2);
endmodule

`default_nettype wire
