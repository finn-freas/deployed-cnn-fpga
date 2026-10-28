from pathlib import Path

from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parents[1]
TB_DIR = ROOT / "tb"
RTL = ROOT / "rtl"

def run(top: str, sources: list[str], test_module: str, parameters: dict | None = None,
        waves: bool = False, testcase: str | None = None, extra_env: dict | None = None):
    build_dir = ROOT / "build" / "sim" / (top + "".join(f"_{k}{v}" for k, v in (parameters or {}).items()))
    runner = get_runner("verilator")
    runner.build(
        sources=[RTL / s for s in sources],
        hdl_toplevel=top,
        build_dir=build_dir,
        parameters=parameters or {},
        build_args=["-Wall", "-Wno-DECLFILENAME", "--x-assign", "unique", "--x-initial", "unique"],
        waves=waves,
        always=True,
    )
    runner.test(
        hdl_toplevel=top,
        test_module=test_module,
        test_dir=TB_DIR,
        build_dir=build_dir,
        waves=waves,
        testcase=testcase,
        extra_env=extra_env or {},
    )
