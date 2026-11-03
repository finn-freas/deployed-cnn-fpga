`default_nettype none

module engine (
    input  wire logic         clk,
    input  wire logic         rst,

    input  wire logic         init,
    input  wire logic         go,
    input  wire logic [11:0]  tok,
    input  wire logic         logits,
    input  wire logic         pos_we,
    input  wire logic [9:0]   pos_d,
    output      logic         busy,
    output      logic         ready,
    output      logic [9:0]   pos,

    input  wire logic         seed_we,
    input  wire logic [31:0]  seed,
    input  wire logic [11:0]  inv_t,
    input  wire logic [15:0]  cut,
    input  wire logic [15:0]  gain,
    input  wire logic         emo_we,
    input  wire logic [6:0]   emo_addr,
    input  wire logic         emo_mask,
    input  wire logic [15:0]  emo_bias,
    output      logic [11:0]  s_tok,
    output      logic [47:0]  s_em,
    output      logic [47:0]  s_total,
    output      logic [23:0]  s_score,
    input  wire logic         best_clr,
    input  wire logic         cand_end,
    input  wire logic         cand_keep,
    output      logic         cmp_busy,
    output      logic [7:0]   best_k,
    output      logic [31:0]  best_s,
    output      logic [7:0]   best_n,

    input  wire logic [7:0]   rd_row,
    output      logic [255:0] rd_vm,
    input  wire logic [11:0]  rd_logit,
    output      logic [15:0]  rd_l,

    input  wire logic         step,
    input  wire logic         step_go,
    output      logic         stopped,
    output      logic [63:0]  op_word,

    output      logic         rd_req,
    output      logic [19:0]  rd_addr,
    output      logic [15:0]  rd_len,
    input  wire logic         rd_ack,
    input  wire logic         rvalid,
    input  wire logic [63:0]  rdata,
    output      logic         wr_req,
    output      logic [19:0]  wr_addr,
    output      logic [63:0]  wr_data,
    input  wire logic         wr_ack,
    input  wire logic         sd_idle
);
    localparam logic [3:0] O_END = 4'd0, O_EMBED = 4'd1, O_RMS = 4'd2, O_MV = 4'd3, O_ROPE = 4'd4,
                           O_KVW = 4'd5, O_ATTN = 4'd6, O_ADD = 4'd7, O_SILU = 4'd8, O_GAIN = 4'd9,
                           O_LOGITS = 4'd10, O_SAMPLE = 4'd11;
    localparam logic [3:0] R_X = 4'd0, R_K = 4'd3, R_V = 4'd4, R_H1 = 4'd7, R_H3 = 4'd8;

    function automatic logic [7:0] base(input logic [3:0] r);
        unique case (r)
            4'd0: return 8'd0;    4'd1: return 8'd16;   4'd2: return 8'd32;   4'd3: return 8'd48;
            4'd4: return 8'd52;   4'd5: return 8'd56;   4'd6: return 8'd72;   4'd7: return 8'd88;
            4'd8: return 8'd132;  4'd9: return 8'd176;  4'd10: return 8'd220;
            default: return 8'd0;
        endcase
    endfunction

    logic [11:0] c_dim, c_hidden, c_vocab;
    logic [9:0]  c_seq;
    logic [1:0]  c_nkv;
    logic [39:0] c_eps;
    logic [7:0]  c_tsh, c_hsh, c_ugsh;
    logic [4:0]  c_flog;
    logic [19:0] c_ucode, c_rope, c_kv, c_rsq, c_exp, c_sig, c_gum, c_wtab;
    logic [8:0]  c_nops;
    logic [19:0] c_kvhalf;
    logic [3:0]  c_seq_sh;

    logic [23:0] rs_mem [1024];
    logic [15:0] sg_mem [512];
    logic [15:0] exa_mem [256];
    logic [23:0] exs_mem [256];
    logic [15:0] gb_mem [1024];
    logic [15:0] w_mem [3072];
    logic [63:0] uc_mem [512];
    logic [9:0]  rs_a;  logic [23:0] rs_d;
    logic [8:0]  sg_a;  logic [15:0] sg_d;
    logic [7:0]  exa_a, exs_a; logic [15:0] exa_d; logic [23:0] exs_d;
    logic [9:0]  gb_a;  logic [15:0] gb_d;
    logic [11:0] w_a;   logic [15:0] w_d;
    logic [8:0]  pc;    logic [63:0] uc_d;

    logic        ld_we;
    logic [2:0]  ld_tab, ld_wtab;
    logic [11:0] ld_i;
    logic [63:0] ld_d;
    always_ff @(posedge clk) begin
        if (ld_we) unique case (ld_wtab)
            3'd0: rs_mem[ld_i[9:0]] <= ld_d[23:0];
            3'd1: begin exa_mem[ld_i[7:0]] <= ld_d[15:0]; exs_mem[ld_i[7:0]] <= ld_d[23:0]; end
            3'd2: sg_mem[ld_i[8:0]] <= ld_d[15:0];
            3'd3: gb_mem[ld_i[9:0]] <= ld_d[15:0];
            3'd4: w_mem[ld_i] <= ld_d[15:0];
            3'd5: uc_mem[ld_i[8:0]] <= ld_d;
            default: ;
        endcase
        rs_d  <= rs_mem[rs_a];
        sg_d  <= sg_mem[sg_a];
        exa_d <= exa_mem[exa_a];
        exs_d <= exs_mem[exs_a];
        gb_d  <= gb_mem[gb_a];
        w_d   <= w_mem[w_a];
        uc_d  <= uc_mem[pc];
    end

    logic [7:0]   vm_ra, vm_wa;
    logic [255:0] vm_rd, vm_wd;
    logic         vm_we;
    logic [15:0]  vm_wm;
    vmem u_vm (.clk, .raddr(vm_ra), .rdata(vm_rd), .we(vm_we), .waddr(vm_wa), .wmask(vm_wm), .wdata(vm_wd));
    assign rd_vm = vm_rd;

    logic        sr_start, s_valid, s_ready;
    logic [19:0] sr_addr, sr_len;
    logic [63:0] s_data;
    stream_rd u_sr (.clk, .rst, .start(sr_start), .addr(sr_addr), .len(sr_len),
                    .rd_req, .rd_addr, .rd_len, .rd_ack, .rvalid, .rdata, .s_valid, .s_data, .s_ready);

    typedef enum logic [2:0] { U_NONE, U_LOAD, U_VEC, U_MV, U_ATT } unit_e;
    unit_e un;

    logic [3:0]  f_op, f_src, f_dst;
    logic [4:0]  f_aux;
    logic [11:0] f_rows, f_cols;
    logic [19:0] f_addr;
    assign f_op   = op_word[3:0];
    assign f_src  = op_word[7:4];
    assign f_dst  = op_word[11:8];
    assign f_aux  = {op_word[60], op_word[15:12]};
    assign f_rows = op_word[27:16];
    assign f_cols = op_word[39:28];
    assign f_addr = op_word[59:40];

    logic [11:0] t_tok;
    logic        t_logits;
    logic [4:0]  c_kvl_sh;
    wire  [19:0] kv_l = c_kv + (20'(f_aux) << c_kvl_sh);

    logic        v_start, v_done, v_sr_start, v_s_ready, v_vm_we, v_wr_req;
    logic [7:0]  v_abase, v_bbase, v_obase, v_vm_ra, v_vm_wa;
    logic [5:0]  v_nrows;
    logic [19:0] v_addr, v_vaddr, v_sr_addr, v_sr_len, v_wr_addr;
    logic [7:0]  v_kvsh;
    logic [255:0] v_vm_wd;
    logic [63:0] v_wr_data;
    vec_unit u_vec (.clk, .rst, .start(v_start), .op(f_op), .abase(v_abase), .bbase(v_bbase), .obase(v_obase),
                    .nrows(v_nrows), .addr(v_addr), .vaddr(v_vaddr), .seq_sh(c_seq_sh), .kv_sh(v_kvsh), .done(v_done),
                    .eps(c_eps), .h_shift(c_hsh), .ug_shift(c_ugsh),
                    .sr_start(v_sr_start), .sr_addr(v_sr_addr), .sr_len(v_sr_len), .s_valid, .s_data,
                    .s_ready(v_s_ready), .vm_raddr(v_vm_ra), .vm_rdata(vm_rd), .vm_we(v_vm_we), .vm_waddr(v_vm_wa),
                    .vm_wdata(v_vm_wd), .rs_addr(rs_a), .rs_data(rs_d), .sg_addr(sg_a), .sg_data(sg_d),
                    .wr_req(v_wr_req), .wr_addr(v_wr_addr), .wr_data(v_wr_data), .wr_ack, .sd_idle);
    assign wr_req  = v_wr_req;
    assign wr_addr = v_wr_addr;
    assign wr_data = v_wr_data;

    logic        m_start, m_busy, m_s_ready, m_ov;
    logic [5:0]  m_xa;
    logic [15:0] m_od;
    logic [11:0] m_oi;
    wire         m_tab = (f_op == O_LOGITS);
    mv_unit u_mv (.clk, .rst, .start(m_start), .tab8(m_tab), .rows(f_rows), .cols(f_cols), .busy(m_busy),
                  .s_valid, .s_data, .s_ready(m_s_ready), .x_addr(m_xa), .x_rdata(vm_rd),
                  .o_valid(m_ov), .o_data(m_od));

    logic        a_start, a_done, a_sr_start, a_s_ready, a_vm_we;
    logic [7:0]  a_vm_ra, a_vm_wa;
    logic [15:0] a_vm_wm;
    logic [19:0] a_sr_addr, a_sr_len, a_kaddr;
    logic [5:0]  a_tsh;
    logic [3:0]  a_vsh;
    logic [15:0] a_vm_wd;
    attn_unit u_att (.clk, .rst, .start(a_start), .qbase(base(f_src)), .obase(base(f_dst)), .kaddr(a_kaddr),
                     .vaddr(a_kaddr + c_kvhalf), .npos(pos + 10'd1), .nkv(c_nkv), .seq_sh(c_seq_sh),
                     .t_shift(a_tsh), .v_shift(a_vsh),
                     .done(a_done), .sr_start(a_sr_start), .sr_addr(a_sr_addr), .sr_len(a_sr_len), .s_valid,
                     .s_data, .s_ready(a_s_ready), .vm_raddr(a_vm_ra), .vm_rdata(vm_rd), .vm_we(a_vm_we),
                     .vm_waddr(a_vm_wa), .vm_wmask(a_vm_wm), .vm_wdata(a_vm_wd), .ex_addr(exa_a), .ex_data(exa_d));

    logic        p_go, p_done, l_start;
    sampler u_smp (.clk, .rst, .seed_we, .seed, .emo_we, .emo_addr, .emo_mask, .emo_bias(emo_bias),
                   .l_start, .l_valid(m_ov && m_tab), .l_data(m_od), .go(p_go), .n(c_vocab), .inv_t, .cut, .gain,
                   .prev(t_tok), .f_logit(c_flog), .ex_addr(exs_a), .ex_data(exs_d), .gb_addr(gb_a), .gb_data(gb_d),
                   .w_addr(w_a), .w_data(w_d), .best_clr, .cand_end, .cand_keep, .cmp_busy, .score(s_score),
                   .best_k(best_k), .best_s(best_s), .best_n(best_n),
                   .dbg_addr(rd_logit), .dbg_data(rd_l), .done(p_done), .tok(s_tok), .em(s_em), .total(s_total));

    logic        e_sr_start;
    logic [19:0] e_sr_addr, e_sr_len;
    always_comb begin
        sr_start = e_sr_start | v_sr_start | a_sr_start;
        sr_addr  = e_sr_start ? e_sr_addr : v_sr_start ? v_sr_addr : a_sr_addr;
        sr_len   = e_sr_start ? e_sr_len  : v_sr_start ? v_sr_len  : a_sr_len;
        unique case (un)
            U_LOAD:  s_ready = s_valid;
            U_VEC:   s_ready = v_s_ready;
            U_MV:    s_ready = m_s_ready;
            U_ATT:   s_ready = a_s_ready;
            default: s_ready = 1'b0;
        endcase
        vm_we = 1'b0; vm_wa = '0; vm_wd = '0; vm_wm = '1;
        unique case (un)
            U_VEC: begin vm_ra = v_vm_ra; vm_we = v_vm_we; vm_wa = v_vm_wa; vm_wd = v_vm_wd; end
            U_MV: begin
                vm_ra = base(f_src) + 8'(m_xa);
                vm_we = m_ov && !m_tab;
                vm_wa = base(f_dst) + 8'(m_oi >> 4);
                vm_wd = {16{m_od}};
                vm_wm = 16'(1) << m_oi[3:0];
            end
            U_ATT: begin vm_ra = a_vm_ra; vm_we = a_vm_we; vm_wa = a_vm_wa; vm_wd = {16{a_vm_wd}}; vm_wm = a_vm_wm; end
            default: vm_ra = rd_row;
        endcase
    end

    typedef enum logic [3:0] {
        S_IDLE, S_CFG, S_LUT, S_FETCH, S_FETCH2, S_DISPATCH, S_WAIT, S_MVWAIT, S_STOP, S_NEXT
    } state_e;
    state_e st;
    logic [3:0]  cfg_i;
    logic [11:0] ld_cnt, ld_n;
    logic        mv_seen;

    function automatic logic [19:0] lut_addr(input logic [2:0] t);
        unique case (t)
            3'd0: return c_rsq;  3'd1: return c_exp;  3'd2: return c_sig;  3'd3: return c_gum;
            3'd4: return c_wtab;
            default: return c_ucode;
        endcase
    endfunction
    function automatic logic [11:0] lut_len(input logic [2:0] t);
        unique case (t)
            3'd0: return 12'd768;  3'd1: return 12'd256;  3'd2: return 12'd512;  3'd3: return 12'd1024;
            3'd4: return 12'd3072;
            default: return 12'(c_nops);
        endcase
    endfunction

    wire [7:0] mv_rw = 8'd1 + 8'((f_cols + 12'd255) >> 8) + 8'(f_cols >> 4);
    wire tok_end = (f_op == O_END) || (f_op == O_LOGITS && !t_logits);

    assign busy    = (st != S_IDLE);
    assign stopped = (st == S_STOP);

    always_ff @(posedge clk) begin
        e_sr_start <= 1'b0;
        v_start    <= 1'b0;
        m_start    <= 1'b0;
        a_start    <= 1'b0;
        ld_we      <= 1'b0;
        p_go       <= 1'b0;
        l_start    <= 1'b0;
        if (m_ov) m_oi <= m_oi + 12'd1;
        if (rst) begin
            st    <= S_IDLE;
            un    <= U_NONE;
            ready <= 1'b0;
            pos   <= '0;
        end else begin
            if (pos_we && st == S_IDLE) pos <= pos_d;
            unique case (st)
                S_IDLE: begin
                    un <= U_NONE;
                    if (init) begin
                        un         <= U_LOAD;
                        ready      <= 1'b0;
                        e_sr_start <= 1'b1;
                        e_sr_addr  <= '0;
                        e_sr_len   <= 20'd8;
                        cfg_i      <= '0;
                        st         <= S_CFG;
                    end else if (go && ready) begin
                        t_tok    <= tok;
                        t_logits <= logits;
                        pc       <= '0;
                        st       <= S_FETCH;
                    end
                end

                S_CFG: if (s_valid) begin
                    unique case (cfg_i)
                        4'd0: begin
                            c_dim    <= s_data[11:0];
                            c_hidden <= s_data[27:16];
                            c_nkv    <= s_data[49:48];
                        end
                        4'd1: begin
                            c_vocab <= s_data[11:0];
                            c_seq   <= s_data[25:16];
                        end
                        4'd2: c_eps <= s_data[39:0];
                        4'd3: begin
                            c_tsh  <= s_data[7:0];
                            c_hsh  <= s_data[15:8];
                            c_ugsh <= s_data[23:16];
                            c_flog <= s_data[28:24];
                            c_wtab <= s_data[51:32];
                        end
                        4'd4: begin c_ucode <= s_data[19:0]; c_nops <= s_data[40:32]; end
                        4'd5: begin c_rope <= s_data[19:0]; c_kv <= s_data[51:32]; end
                        4'd6: begin c_rsq <= s_data[19:0]; c_exp <= s_data[51:32]; end
                        default: begin
                            c_sig      <= s_data[19:0];
                            c_gum      <= s_data[51:32];
                            c_kvhalf   <= (c_nkv == 2'd2) ? 20'(c_seq) << 3 : 20'(c_seq) << 2;
                            c_seq_sh   <= c_seq[9] ? 4'd9 : c_seq[8] ? 4'd8 : c_seq[7] ? 4'd7 : c_seq[6] ? 4'd6 : 4'd5;
                            c_kvl_sh   <= 5'(c_seq[9] ? 9 : c_seq[8] ? 8 : c_seq[7] ? 7 : c_seq[6] ? 6 : 5)
                                          + ((c_nkv == 2'd2) ? 5'd4 : 5'd3);
                            ld_tab     <= '0;
                            ld_cnt     <= '0;
                            ld_n       <= 12'd768;
                            e_sr_start <= 1'b1;
                            e_sr_addr  <= c_rsq;
                            e_sr_len   <= 20'd768;
                            st         <= S_LUT;
                        end
                    endcase
                    cfg_i <= cfg_i + 4'd1;
                end

                S_LUT: if (s_valid) begin
                    ld_we   <= 1'b1;
                    ld_wtab <= ld_tab;
                    ld_i    <= ld_cnt;
                    ld_d   <= s_data;
                    ld_cnt <= ld_cnt + 12'd1;
                    if (ld_cnt + 12'd1 == ld_n) begin
                        if (ld_tab == 3'd5) begin
                            ready <= 1'b1;
                            un    <= U_NONE;
                            st    <= S_IDLE;
                        end else begin
                            ld_tab     <= ld_tab + 3'd1;
                            ld_cnt     <= '0;
                            ld_n       <= lut_len(ld_tab + 3'd1);
                            e_sr_start <= 1'b1;
                            e_sr_addr  <= lut_addr(ld_tab + 3'd1);
                            e_sr_len   <= 20'(lut_len(ld_tab + 3'd1));
                        end
                    end
                end

                S_FETCH:  st <= S_FETCH2;
                S_FETCH2: begin
                    op_word <= uc_d;
                    st      <= S_DISPATCH;
                end
                S_DISPATCH: begin
                    st <= S_WAIT;
                    unique case (f_op)
                        O_END: st <= S_NEXT;
                        O_MV, O_LOGITS: begin
                            if (f_op == O_LOGITS && !t_logits) st <= S_NEXT;
                            else begin
                                un         <= U_MV;
                                m_start    <= 1'b1;
                                l_start    <= (f_op == O_LOGITS);
                                m_oi       <= '0;
                                mv_seen    <= 1'b0;
                                e_sr_start <= 1'b1;
                                e_sr_addr  <= f_addr;
                                e_sr_len   <= (f_op == O_LOGITS) ? (20'(f_rows) << 5) + 20'(f_rows)
                                                                 : 20'(f_rows) * 20'(mv_rw);
                                st         <= S_MVWAIT;
                            end
                        end
                        O_SAMPLE: begin
                            un   <= U_NONE;
                            p_go <= 1'b1;
                        end
                        O_ATTN: begin
                            un      <= U_ATT;
                            a_start <= 1'b1;
                            a_kaddr <= kv_l;
                            a_tsh   <= c_tsh[5:0] - 6'(f_rows[3:0]);
                            a_vsh   <= f_rows[7:4];
                        end

                        default: begin
                            un      <= U_VEC;
                            v_start <= 1'b1;
                            v_vaddr <= kv_l + c_kvhalf + 20'(pos);
                            v_abase <= (f_op == O_SILU) ? base(R_H1) : base(f_src);
                            v_bbase <= (f_op == O_SILU) ? base(R_H3) : (f_op == O_KVW) ? base(R_V) : base(f_dst);
                            v_obase <= (f_op == O_EMBED) ? base(R_X) : base(f_dst);
                            v_nrows <= (f_op == O_EMBED) ? 6'(c_dim >> 4) : 6'(f_cols >> 4);
                            v_kvsh  <= f_rows[7:0];
                            unique case (f_op)
                                O_EMBED: v_addr <= f_addr + (20'(t_tok) << 5) + 20'(t_tok);
                                O_ROPE:  v_addr <= c_rope + (20'(pos) << 3);
                                O_KVW:   v_addr <= kv_l + ((c_nkv == 2'd2) ? 20'(pos) << 3 : 20'(pos) << 2);
                                default: v_addr <= f_addr;
                            endcase
                        end
                    endcase
                end
                S_MVWAIT: begin
                    if (m_busy) mv_seen <= 1'b1;
                    else if (mv_seen) begin
                        un <= U_NONE;
                        st <= step ? S_STOP : S_NEXT;
                    end
                end
                S_WAIT: if (v_done || a_done || p_done) begin
                    un <= U_NONE;
                    st <= step ? S_STOP : S_NEXT;
                end
                S_STOP: if (step_go) st <= S_NEXT;

                S_NEXT: begin
                    if (tok_end) begin
                        pos <= pos + 10'd1;
                        st  <= S_IDLE;
                    end else begin
                        pc <= pc + 9'd1;
                        st <= S_FETCH;
                    end
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
