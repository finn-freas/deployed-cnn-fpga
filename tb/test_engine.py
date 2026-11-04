import json
import os
import pickle
import sys
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, First, RisingEdge, Timer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import board as B
import engine as E
import pack as P
import engine_image

KEY = engine_image.KEY
TOKENS = [int(t) for t in os.environ.get("FB_TOKENS", "1,2,700").split(",")]
BASE = {"X": 0, "U": 16, "Q": 32, "K": 48, "V": 52, "ATT": 56, "T": 72, "H1": 88, "H3": 132, "HB": 176, "UG": 220}
WRITES = {"EMBED", "RMS", "MV", "ROPE", "ATTN", "ADD", "SILU", "GAIN"}


def make_rng(seed):
    x = seed & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= x >> 16
    return B.XorShift(x)


def emoji_tables(p):
    mask = np.zeros(p["cfg"].vocab_size, bool)
    for tid, s, _ in json.loads((ROOT / "data" / "tok" / "slots.json").read_text()):
        if not ("︀" <= s <= "️" or "\U0001F3FB" <= s <= "\U0001F3FF"):
            mask[tid] = True
    bias = np.zeros(p["cfg"].vocab_size, np.int64)
    for tid, nats in json.loads((ROOT / "models" / f"{KEY}_emoji.json").read_text()).get("bias", []):
        bias[tid] = round(nats * 2 ** p["fmt"]["logit"])
    return mask, bias


def sgn(v, bits):
    v = int(v)
    return v - ((v >> (bits - 1)) << bits)


async def best(dut, sig, keep=False):
    getattr(dut, sig).value, dut.cand_keep.value = 1, int(keep)
    await RisingEdge(dut.clk)
    getattr(dut, sig).value, dut.cand_keep.value = 0, 0
    await RisingEdge(dut.cmp_busy)
    await FallingEdge(dut.cmp_busy)
    await RisingEdge(dut.clk)
    return int(dut.best_k.value), sgn(dut.best_s.value, 32), int(dut.best_n.value)


async def read_rows(dut, base, n):
    out = []
    for r in range(n):
        dut.dbg_row.value = base + r
        await ClockCycles(dut.clk, 2)
        v = int(dut.dbg_vm.value)
        out += [((v >> (16 * k)) & 0xFFFF) - ((v >> (16 * k) & 0x8000) << 1) for k in range(16)]
    return np.array(out, np.int64)


async def read_logits(dut, n):
    out = []
    for i in range(n):
        dut.dbg_logit.value = i
        await ClockCycles(dut.clk, 2)
        out.append(dut.dbg_l.value.to_signed())
    return np.array(out, np.int64)


async def watchdog(dut, every_us=500):
    e = dut.u_eng
    last = None
    while True:
        await Timer(every_us, unit="us")
        word = int(e.op_word.value)
        now = (int(e.st.value), int(e.pc.value), int(e.u_vec.st.value), int(e.u_att.st.value),
               int(e.u_mv.st.value), int(e.u_smp.st.value), int(e.u_sr.left.value), int(e.u_sr.out_v.value))
        dut._log.info(f"watchdog: engine st {now[0]} pc {now[1]} op {P.OPS[word & 0xF]} vec {now[2]} att {now[3]} "
                      f"mv {now[4]} smp {now[5]} stream left {now[6]} out_v {now[7]}")
        if now == last:
            raise AssertionError("engine made no progress in a watchdog period")
        last = now


@cocotb.test(timeout_time=400, timeout_unit="ms")
async def test_engine(dut):
    p = engine_image.model()
    img, _ = P.build(p)
    ref = E.Engine(img.bytes())
    mask, bias = emoji_tables(p)
    inv_t, cut_q, gain_q = 320, 4 * 2 ** p["fmt"]["logit"], 474

    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    cocotb.start_soon(Clock(dut.clk_f, int(os.environ.get("FB_FCLK_PS", "5000")), unit="ps").start())
    for s in ["init", "go", "logits", "forced", "ftok", "wbase", "pos_we", "seed_we", "emo_we", "step", "step_go", "best_clr", "cand_end",
              "cand_keep"]:
        getattr(dut, s).value = 0
    dut.tok.value, dut.dbg_row.value, dut.dbg_logit.value, dut.pos_d.value = 0, 0, 0, 0
    dut.inv_t.value, dut.cut.value, dut.gain.value = inv_t, cut_q, gain_q
    dut.rst.value = 1
    await ClockCycles(dut.clk, 5)
    dut.rst.value = 0
    await RisingEdge(dut.sd_ready)
    for t in range(128):
        dut.emo_we.value, dut.emo_addr.value = 1, t
        dut.emo_mask.value, dut.emo_bias.value = int(mask[t]), int(bias[t]) & 0xFFFF
        await RisingEdge(dut.clk)
    dut.emo_we.value = 0
    dut.seed_we.value, dut.seed.value = 1, make_rng(1).x
    await RisingEdge(dut.clk)
    dut.seed_we.value = 0
    dut.init.value = 1
    await RisingEdge(dut.clk)
    dut.init.value = 0
    await RisingEdge(dut.ready)
    dut._log.info("engine initialised (config, LUTs, microcode loaded)")
    cocotb.start_soon(watchdog(dut))

    for pos, tok in enumerate(TOKENS):
        last = pos == len(TOKENS) - 1
        trace = []
        ref.run(tok, pos, logits=last, trace=lambda op, reg, v: trace.append((op, reg, np.array(v))))
        dut.tok.value, dut.logits.value, dut.step.value = tok, int(last), int(last)
        dut.go.value = 1
        await RisingEdge(dut.clk)
        dut.go.value = 0
        if not last:
            await FallingEdge(dut.busy)
            assert int(dut.pos.value) == pos + 1
            dut._log.info(f"token {tok} at pos {pos}: done")
            pc = {k: int(getattr(dut, "pc_" + k).value) for k in
                  ("busy", "mv", "mvstall", "vec", "vecwait", "att", "attwait")}
            dut._log.info(f"PERF token {pos}: " + " ".join(f"{k} {v}" for k, v in pc.items()))
            continue
        n_ops, bad = 0, []
        while True:
            await First(RisingEdge(dut.stopped), FallingEdge(dut.busy))
            if not dut.busy.value:
                break
            op = P.OPS[int(dut.op_word.value) & 0xF]
            if op in WRITES or op == "LOGITS":
                top, reg, want = trace.pop(0)
                assert top == op, (top, op)
                got = await (read_logits(dut, len(want)) if op == "LOGITS"
                             else read_rows(dut, BASE[reg], len(want) // 16))
                diff = np.flatnonzero(got != want)
                if len(diff):
                    bad.append(f"op {n_ops} {op} -> {reg}: {len(diff)}/{len(want)} differ, first at {diff[:4]}: "
                               f"got {got[diff[:4]]} want {want[diff[:4]]}")
                    dut._log.error(bad[-1])
                    if len(bad) >= 3:
                        break
            n_ops += 1
            dut.step_go.value = 1
            await RisingEdge(dut.clk)
            dut.step_go.value = 0
        assert not bad, "\n".join(bad)
        dut._log.info(f"token {tok} at pos {pos}: {n_ops} ops, every register and all logits bit-exact")
        lg = await read_logits(dut, p["cfg"].vocab_size)
        rng = make_rng(1)
        g = 256 if mask[tok] else gain_q
        em, total = B.class_mass(lg, mask, p["fmt"]["logit"], p["exp2"])
        want_tok = B.pick(lg, inv_t, cut_q, rng, p["gumbel"], mask, g, p["fmt"]["logit"], p["exp2"], bias)
        got = (int(dut.s_tok.value), int(dut.s_em.value), int(dut.s_total.value))
        assert got == (want_tok, em, total), f"sample: got {got}, want {(want_tok, em, total)}"
        w = p["score_w"]
        s1 = B.token_score(lg, want_tok, p["fmt"]["logit"], total, w)
        assert sgn(dut.s_score.value, 24) == s1, f"score: got {sgn(dut.s_score.value, 24)}, want {s1}"
        dut._log.info(f"sampled token {got[0]} (emoji mass {em}/{total}, score {s1}): bit-exact")

        assert await best(dut, "cand_end", keep=True) == (0, s1, 1)
        dut.pos_we.value, dut.pos_d.value = 1, pos
        await RisingEdge(dut.clk)
        dut.pos_we.value = 0
        await RisingEdge(dut.clk)
        assert int(dut.pos.value) == pos
        dut.tok.value, dut.logits.value, dut.step.value = tok, 1, 0
        dut.go.value = 1
        await RisingEdge(dut.clk)
        dut.go.value = 0
        await FallingEdge(dut.busy)
        want2 = B.pick(lg, inv_t, cut_q, rng, p["gumbel"], mask, g, p["fmt"]["logit"], p["exp2"], bias)
        s2 = B.token_score(lg, want2, p["fmt"]["logit"], total, w)
        assert (int(dut.s_tok.value), sgn(dut.s_score.value, 24)) == (want2, s2)
        k = 1 if B.better(s2, 1, s1, 1) else 0
        assert await best(dut, "cand_end", keep=True) == (k, max(s1, s2), 1)
        assert await best(dut, "best_clr") == (0, 0, 0)
        dut._log.info(f"second candidate {want2} (score {s2}): best of the two is candidate {k}, bit-exact")

        base = 1
        for q, (t2, ft) in enumerate([(5, 700), (9, int(np.flatnonzero(mask)[0])), (11, None)]):
            at = pos + 1 + q
            want = ref.run(t2, at, logits=True, base=base)
            dut.wbase.value, dut.tok.value, dut.logits.value = base, t2, 1
            dut.forced.value, dut.ftok.value = int(ft is not None), ft or 0
            dut.go.value = 1
            await RisingEdge(dut.clk)
            dut.go.value = 0
            await FallingEdge(dut.busy)
            got = await read_logits(dut, len(want))
            assert (got == want).all(), f"window {base}: {int((got != want).sum())} logits differ at pos {at}"
            em, total = B.class_mass(got, mask, p["fmt"]["logit"], p["exp2"])
            assert (int(dut.s_em.value), int(dut.s_total.value)) == (em, total)
            if ft is not None:
                sx = B.token_score(got, ft, p["fmt"]["logit"], total, w)
                assert (int(dut.s_tok.value), sgn(dut.s_score.value, 24)) == (ft, sx), "forced token score"
            else:
                g = 256 if mask[t2] else gain_q
                w3 = B.pick(got, inv_t, cut_q, rng, p["gumbel"], mask, g, p["fmt"]["logit"], p["exp2"], bias)
                assert int(dut.s_tok.value) == w3, f"after forced: sampled {int(dut.s_tok.value)}, want {w3}"
                s3 = B.token_score(got, w3, p["fmt"]["logit"], total, w)
            dut._log.info(f"window from {base}: token {t2} at pos {at}, all logits bit-exact, "
                          + (f"forced {ft} scored {sx}" if ft is not None else "sampled (the RNG skipped forced steps)"))
        assert await best(dut, "cand_end", keep=True) == (0, s3, 1)
        dut.wbase.value, dut.forced.value = 0, 0
        dut._log.info("forced scores left the candidate books alone")
