`default_nettype none

module host_link #(
    parameter int CLK_HZ    = 54_000_000,
    parameter int BAUD      = 1_000_000
) (
    input  wire logic        clk,
    input  wire logic        rst,
    input  wire logic        uart_rx,
    output      logic        uart_tx,
    output      logic        busy,

    input  wire logic        sd_ready,
    input  wire logic        cal_err,
    input  wire logic [15:0] sd_diag,
    input  wire logic        sd_idle,
    output      logic        rd_req,
    output      logic [19:0] rd_addr,
    output      logic [15:0] rd_len,
    input  wire logic        rd_ack,
    input  wire logic        rvalid,
    input  wire logic [63:0] rdata,
    output      logic        wr_req,
    output      logic [19:0] wr_addr,
    output      logic [63:0] wr_data,
    input  wire logic        wr_ack,

    output      logic        e_init,
    output      logic        e_go,
    output      logic [11:0] e_tok,
    output      logic        e_logits,
    output      logic        e_pos_we,
    output      logic [9:0]  e_pos_d,
    output      logic        e_best_clr,
    output      logic        e_cand_end,
    output      logic        e_cand_keep,
    input  wire logic        e_cmp_busy,
    input  wire logic [23:0] e_s_score,
    input  wire logic [7:0]  e_best_k,
    input  wire logic [31:0] e_best_s,
    input  wire logic [7:0]  e_best_n,
    input  wire logic        e_busy,
    input  wire logic [9:0]  e_pos,
    output      logic        e_seed_we,
    output      logic [31:0] e_seed,
    output      logic [11:0] e_inv_t,
    output      logic [15:0] e_cut,
    output      logic [15:0] e_gain,
    output      logic        e_emo_we,
    output      logic [6:0]  e_emo_addr,
    output      logic        e_emo_mask,
    output      logic [15:0] e_emo_bias,
    input  wire logic [11:0] e_s_tok,
    input  wire logic [47:0] e_s_em,
    input  wire logic [47:0] e_s_total
);
    localparam logic [7:0] VERSION = 8'd2;
    localparam int DIV   = (CLK_HZ + BAUD / 2) / BAUD;
    localparam int CHUNK = 16384;

    logic       rx_v, tx_v, tx_rdy;
    logic [7:0] rx_d, tx_d;
    uart_rx #(.DIV(DIV)) u_rx (.clk, .rst, .rx(uart_rx), .valid(rx_v), .data(rx_d));
    uart_tx #(.DIV(DIV)) u_tx (.clk, .rst, .valid(tx_v), .data(tx_d), .ready(tx_rdy), .tx(uart_tx));

    typedef enum logic [4:0] {
        S_CMD, S_ARG, S_WDATA, S_WLAST, S_RREQ, S_RWAIT, S_SUM, S_REPLY,
        S_EARG, S_ETAB, S_EWAIT, S_FREPLY, S_FREPLY2, S_YWAIT
    } state_e;
    state_e      st;
    logic [7:0]  cmd;
    logic [2:0]  argn;
    logic [47:0] args;
    logic [19:0] addr;
    logic [23:0] n_left;
    logic [2:0]  bytei;
    logic [23:0] got;
    logic [23:0] asked;
    logic [31:0] sum;

    logic [63:0] out_sh;
    logic [3:0]  out_n;

    assign busy = (st != S_CMD);

    logic [2:0] e_need;
    logic [1:0] etab_ph;
    logic [6:0] etab_a;
    logic       e_seen;

    always_ff @(posedge clk) begin
        tx_v        <= 1'b0;
        e_init      <= 1'b0;
        e_go        <= 1'b0;
        e_pos_we    <= 1'b0;
        e_best_clr  <= 1'b0;
        e_cand_end  <= 1'b0;
        e_seed_we   <= 1'b0;
        e_emo_we    <= 1'b0;
        if (out_n != 0 && tx_rdy && !tx_v) begin
            tx_v   <= 1'b1;
            tx_d   <= out_sh[7:0];
            out_sh <= out_sh >> 8;
            out_n  <= out_n - 4'd1;
        end
        if (rd_req && rd_ack) rd_req <= 1'b0;
        if (wr_req && wr_ack) wr_req <= 1'b0;

        if (rst) begin
            st     <= S_CMD;
            out_n  <= '0;
            rd_req <= 1'b0;
            wr_req <= 1'b0;
        end else begin
            unique case (st)
                S_CMD: if (rx_v) begin
                    cmd  <= rx_d;
                    argn <= '0;
                    unique case (rx_d)
                        "P": begin
                            out_sh <= 64'({6'b0, cal_err, sd_ready, VERSION, 8'("K")});
                            out_n  <= 4'd3;
                        end
                        "D": begin
                            out_sh <= 64'({sd_diag, 8'("d")});
                            out_n  <= 4'd3;
                        end
                        "W", "R", "C": st <= S_ARG;
                        "I": begin
                            e_init <= 1'b1;
                            e_seen <= 1'b0;
                            st     <= S_EWAIT;
                        end
                        "Z": begin
                            e_pos_we <= 1'b1;
                            e_pos_d  <= '0;
                            out_sh <= 64'("z");
                            out_n  <= 4'd1;
                        end
                        "S", "G", "F", "J", "Y": begin
                            e_need <= (rx_d == "S") ? 3'd4 : (rx_d == "G") ? 3'd6 : (rx_d == "F") ? 3'd3 :
                                      (rx_d == "J") ? 3'd2 : 3'd1;
                            st     <= S_EARG;
                        end
                        "E": begin
                            etab_ph <= '0;
                            etab_a  <= '0;
                            st      <= S_ETAB;
                        end
                        default: begin
                            out_sh <= 64'("?");
                            out_n  <= 4'd1;
                        end
                    endcase
                end

                S_ARG: if (rx_v) begin
                    args <= {rx_d, args[47:8]};
                    argn <= argn + 3'd1;
                    if (argn == 3'd5) begin
                        addr   <= args[27:8];
                        n_left <= {rx_d, args[47:32]};
                        bytei  <= '0;
                        got    <= '0;
                        asked  <= '0;
                        sum    <= '0;
                        unique case (cmd)
                            "W":     st <= ({rx_d, args[47:32]} == 0) ? S_REPLY : S_WDATA;
                            "R":     st <= ({rx_d, args[47:32]} == 0) ? S_CMD : S_RREQ;
                            default: st <= S_SUM;
                        endcase
                    end
                end

                S_WDATA: if (rx_v) begin
                    wr_data <= {rx_d, wr_data[63:8]};
                    bytei   <= bytei + 3'd1;
                    if (bytei == 3'd7) begin
                        wr_req  <= 1'b1;
                        wr_addr <= addr;
                        addr    <= addr + 20'd1;
                        n_left  <= n_left - 24'd1;
                        if (n_left == 24'd1) st <= S_WLAST;
                    end
                end
                S_WLAST: if (!wr_req) st <= S_REPLY;

                S_RREQ: if (out_n == 0 && !rd_req) begin
                    rd_req  <= 1'b1;
                    rd_addr <= addr;
                    rd_len  <= 16'd1;
                    st      <= S_RWAIT;
                end
                S_RWAIT: if (rvalid) begin
                    out_sh <= rdata;
                    out_n  <= 4'd8;
                    addr   <= addr + 20'd1;
                    n_left <= n_left - 24'd1;
                    st     <= (n_left == 24'd1) ? S_CMD : S_RREQ;
                end

                S_SUM: begin
                    if (!rd_req && asked == got && asked != n_left) begin
                        rd_req  <= 1'b1;
                        rd_addr <= addr + 20'(asked);
                        rd_len  <= (n_left - asked > 24'(CHUNK)) ? 16'(CHUNK) : 16'(n_left - asked);
                        asked   <= (n_left - asked > 24'(CHUNK)) ? asked + 24'(CHUNK) : n_left;
                    end
                    if (rvalid) begin
                        sum <= sum + rdata[31:0] + rdata[63:32];
                        got <= got + 24'd1;
                    end
                    if (got == n_left && asked == n_left && !rd_req) begin
                        out_sh <= 64'(sum);
                        out_n  <= 4'd4;
                        st     <= S_CMD;
                    end
                end

                S_EARG: if (rx_v) begin
                    args <= {rx_d, args[47:8]};
                    argn <= argn + 3'd1;
                    if (argn + 3'd1 == e_need) begin
                        unique case (cmd)
                            "S": begin
                                e_seed_we <= 1'b1;
                                e_seed    <= {rx_d, args[47:24]};
                                out_sh    <= 64'("s");
                                out_n     <= 4'd1;
                                st        <= S_CMD;
                            end
                            "G": begin
                                e_inv_t <= args[19:8];
                                e_cut   <= args[39:24];
                                e_gain  <= {rx_d, args[47:40]};
                                out_sh  <= 64'("g");
                                out_n   <= 4'd1;
                                st      <= S_CMD;
                            end
                            "J": begin
                                e_pos_we <= 1'b1;
                                e_pos_d  <= {rx_d[1:0], args[47:40]};
                                out_sh   <= 64'("j");
                                out_n    <= 4'd1;
                                st       <= S_CMD;
                            end
                            "Y": begin
                                e_best_clr  <= rx_d[0];
                                e_cand_end  <= rx_d[1] && !rx_d[0];
                                e_cand_keep <= rx_d[2];
                                e_seen      <= (rx_d[1:0] == 2'b00);
                                st          <= S_YWAIT;
                            end
                            default: begin
                                e_logits <= args[32];
                                e_tok    <= {rx_d[3:0], args[47:40]};
                                e_go     <= 1'b1;
                                e_seen   <= 1'b0;
                                st       <= S_EWAIT;
                            end
                        endcase
                    end
                end
                S_ETAB: if (rx_v) begin
                    etab_ph <= (etab_ph == 2'd2) ? 2'd0 : etab_ph + 2'd1;
                    unique case (etab_ph)
                        2'd0: e_emo_mask <= rx_d[0];
                        2'd1: e_emo_bias[7:0] <= rx_d;
                        default: begin
                            e_emo_bias[15:8] <= rx_d;
                            e_emo_addr <= etab_a;
                            etab_a     <= etab_a + 7'd1;
                            e_emo_we   <= 1'b1;
                        end
                    endcase
                    if (etab_a == 7'd127 && etab_ph == 2'd2) begin
                        out_sh <= 64'("e");
                        out_n  <= 4'd1;
                        st     <= S_CMD;
                    end
                end
                S_EWAIT: begin
                    if (e_busy) e_seen <= 1'b1;
                    else if (e_seen && out_n == 0) begin
                        if (cmd == "I") begin
                            out_sh <= 64'("i");
                            out_n  <= 4'd1;
                            st     <= S_CMD;
                        end else if (!e_logits) begin
                            out_sh <= 64'({6'b0, e_pos, 8'("a")});
                            out_n  <= 4'd3;
                            st     <= S_CMD;
                        end else begin
                            out_sh <= {e_s_em[39:0], 4'b0, e_s_tok, 8'("f")};
                            out_n  <= 4'd8;
                            st     <= S_FREPLY;
                        end
                    end
                end
                S_FREPLY: if (out_n == 0 && !tx_v) begin
                    out_sh <= {e_s_score[7:0], e_s_total, e_s_em[47:40]};
                    out_n  <= 4'd8;
                    st     <= S_FREPLY2;
                end
                S_FREPLY2: if (out_n == 0 && !tx_v) begin
                    out_sh <= 64'(e_s_score[23:8]);
                    out_n  <= 4'd2;
                    st     <= S_CMD;
                end
                S_YWAIT: begin
                    if (e_cmp_busy) e_seen <= 1'b1;
                    else if (e_seen && out_n == 0) begin
                        out_sh <= {8'b0, e_best_n, e_best_s, e_best_k, 8'("y")};
                        out_n  <= 4'd7;
                        st     <= S_CMD;
                    end
                end

                S_REPLY: if (out_n == 0) begin
                    out_sh <= 64'("w");
                    out_n  <= 4'd1;
                    st     <= S_CMD;
                end

                default: st <= S_CMD;
            endcase
        end
    end
endmodule

`default_nettype wire
