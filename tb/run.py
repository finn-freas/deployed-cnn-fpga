import sys
from pathlib import Path

from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parents[1]
TESTS = {"mv": ("mv_tb_top", ["rtl/mv_unit.sv", "rtl/sim/mv_tb_top.sv"], "test_mv_unit"),
         "link": ("link_tb_top", ["rtl/board/uart.sv", "rtl/board/sdram_ctrl.sv", "rtl/board/sdram_cdc.sv", "rtl/board/reset_sync.sv",
                                  "rtl/sim/sdram_model.sv", "rtl/host_link.sv", "rtl/sim/link_tb_top.sv"], "test_link"),
         "engine": ("engine_tb_top", ["rtl/board/sdram_ctrl.sv", "rtl/board/sdram_cdc.sv", "rtl/board/reset_sync.sv", "rtl/sim/sdram_model.sv", "rtl/vmem.sv",
                                      "rtl/stream_rd.sv", "rtl/vec_unit.sv", "rtl/mv_unit.sv", "rtl/attn_unit.sv",
                                      "rtl/sampler.sv", "rtl/engine.sv",
                                      "rtl/sim/engine_tb_top.sv"], "test_engine")}


def main():
    top, srcs, mod = TESTS[sys.argv[1]]
    if sys.argv[1] == "engine":
        sys.path.insert(0, str(ROOT / "tb"))
        import engine_image
        engine_image.write()
    build = ROOT / "build" / "sim" / top
    r = get_runner("verilator")
    r.build(sources=[ROOT / s for s in srcs], hdl_toplevel=top, build_dir=build, always=True,
            build_args=["-Wall", "-Wno-DECLFILENAME", "-Wno-UNUSEDSIGNAL", "-Wno-UNUSEDPARAM", "-Wno-PINCONNECTEMPTY", "--x-assign", "unique",
                        "--x-initial", "unique"])
    r.test(hdl_toplevel=top, test_module=mod, test_dir=ROOT / "tb", build_dir=build)


if __name__ == "__main__":
    main()
