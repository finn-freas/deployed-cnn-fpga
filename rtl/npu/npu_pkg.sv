package npu_pkg;

    /* verilator lint_off UNUSEDPARAM */
    localparam int N          = 8;
    localparam int DATA_W     = 8;
    localparam int ACC_W      = 26;
    localparam int WORD_W     = N * DATA_W;
    localparam int ACT_DEPTH  = 7168;
    localparam int EXT_DEPTH  = 65536;
    localparam int WBUF_DEPTH = 1024;
    localparam int PBUF_DEPTH = 256;
    localparam int PARAM_WORDS = 2;
    localparam int IMEM_DEPTH = 64;
    localparam int K_MAX      = 1024;
    localparam int ACT_AW     = 13;
    localparam int EXT_AW     = 16;
    localparam int WBUF_AW    = 10;
    localparam int PBUF_AW    = 8;
    localparam int IMEM_AW    = 6;
    localparam int K_W        = 11;
    localparam int INSTR_W    = 160;
    localparam int SHIFT_W    = 6;
    localparam int PARAM_W    = 70;

    localparam bit RMODE_DOUBLE = 1'b0;
    localparam bit RMODE_SINGLE = 1'b1;
    /* verilator lint_on UNUSEDPARAM */

    typedef enum logic [3:0] {
        OP_NOP        = 4'h0,
        OP_CONV       = 4'h1,
        OP_MAXPOOL    = 4'h2,
        OP_DWCONV     = 4'h3,
        OP_ARGMAX     = 4'h4,
        OP_AVGPOOL    = 4'h5,
        OP_END        = 4'hF
    } op_e;

    typedef struct packed {
        logic [8:0] rsvd;
        logic uniform;
        logic in_src;
        logic rmode;
        logic signed [7:0] act_max;
        logic signed [7:0] act_min;
        logic signed [7:0] out_zp;
        logic signed [7:0] in_zp;
        logic [2:0] pad;
        logic [1:0] stride;
        logic [2:0] ksize;
        logic [5:0] out_g;
        logic [6:0] out_w;
        logic [6:0] out_h;
        logic [5:0] in_cb;
        logic [6:0] in_w;
        logic [6:0] in_h;
        logic [15:0] p_base;
        logic [15:0] w_base;
        logic [15:0] out_base;
        logic [15:0] in_base;
        op_e op;
    } instr_t;

    typedef struct packed {
        logic signed [SHIFT_W-1:0] shift;
        logic signed [31:0]        mult;
        logic signed [31:0]        bias;
    } param_t;

endpackage
