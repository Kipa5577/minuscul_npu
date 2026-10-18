"""
Functional test of the activation_buffer (specification: src/activation_buffer/interface.md).

  - host access to both banks (bit BAW of the address), latency 1, bank read latched for host_rdata
  - core read of the IN bank and core write of the OUT bank with a per-byte mask, in the same cycle
  - a single active side at a time, selected by busy; comparison with a reference model
  - D6 layout: X stored by the host, read back at the addresses emitted by the controller scheduler
  - complete datapath with its memories (weight_buffer, activation_buffer): the host loads W, X and b,
    the core computes, the host reads Y back from the OUT bank; then two layers chained by copying OUT → IN

This file also provides the "host driver" functions (load_*, read_*, run_layer) reused by the benchmark
and the animation.

Run: python test/activation_buffer/test.py
"""

import sys
import pathlib
import random

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import Bench, pack, unpack, load_test_module, run_tests
import layout

import numpy as np
from src.activation_buffer.activation_buffer import ActivationBuffer

OD = load_test_module('output_drain')
PP = load_test_module('post_processing')
N, DW, BAW, IW, AW = 4, 8, 10, 6, 32
IN, OUT = 0, 1


def bench():
    b = Bench(
        inputs={'busy': 1, 'host_addr': BAW + 1, 'host_wdata': N * DW, 'host_we': 1, 'host_re': 1,
                'core_rd_addr': BAW, 'core_rd_re': 1, 'core_wr_addr': BAW, 'core_wr_wdata': N * DW,
                'core_wr_we': 1, 'core_wr_wmask': N},
        outputs={'host_rdata': N * DW, 'core_rd_rdata': N * DW},
        build=lambda s, w: ActivationBuffer(s, 'abuf', w['busy'], w['host_addr'], w['host_wdata'], w['host_we'],
                                            w['host_re'], w['host_rdata'], w['core_rd_addr'], w['core_rd_re'],
                                            w['core_rd_rdata'], w['core_wr_addr'], w['core_wr_wdata'],
                                            w['core_wr_we'], w['core_wr_wmask']))
    b.set(busy=0)
    return b


def hw(b, bank, a, word):
    b.step(busy=0, host_addr=(bank << BAW) | a, host_wdata=word, host_we=1, host_re=0)


def hr(b, bank, a):
    b.step(busy=0, host_addr=(bank << BAW) | a, host_we=0, host_re=1)
    return b.get('host_rdata')


# --- Unit tests --------------------------------------------------------------------------------------------

def test_host_both_banks():
    b = bench()
    hw(b, IN, 12, 0x01020304)
    hw(b, OUT, 12, 0x0A0B0C0D)
    assert hr(b, IN, 12) == 0x01020304
    assert hr(b, OUT, 12) == 0x0A0B0C0D
    b.step(busy=0, host_re=0)
    assert b.get('host_rdata') == 0x0A0B0C0D, 'bank read latched, data held'


def test_core_reads_in_and_writes_out_in_the_same_cycle():
    b = bench()
    hw(b, IN, 3, 0x11111111)
    hw(b, OUT, 3, 0x22222222)
    b.step(busy=1, core_rd_addr=3, core_rd_re=1, core_wr_addr=3, core_wr_wdata=0x99999999, core_wr_we=1,
           core_wr_wmask=0b1111)
    assert b.get('core_rd_rdata') == 0x11111111, 'the IN bank is not affected by the write'
    b.step(busy=1, core_rd_re=0, core_wr_we=0)
    assert hr(b, OUT, 3) == 0x99999999 and hr(b, IN, 3) == 0x11111111


def test_per_byte_mask():
    b = bench()
    hw(b, OUT, 8, 0xAABBCCDD)
    for mask, expected in [(0b0001, 0xAABBCC11), (0b0110, 0xAA2222DD), (0b1000, 0x33BBCCDD), (0, 0xAABBCCDD)]:
        hw(b, OUT, 8, 0xAABBCCDD)
        val = {0b0001: 0x11111111, 0b0110: 0x22222222, 0b1000: 0x33333333, 0: 0x44444444}[mask]
        b.step(busy=1, core_wr_addr=8, core_wr_wdata=val, core_wr_we=1, core_wr_wmask=mask)
        b.step(busy=1, core_wr_we=0)
        assert hr(b, OUT, 8) == expected, f'mask {mask:04b}'


def test_active_side_selected_by_busy():
    b = bench()
    hw(b, IN, 1, 0x55)
    b.step(busy=1, host_addr=1, host_wdata=0x66, host_we=1, host_re=1)
    assert hr(b, IN, 1) == 0x55, 'host write ignored during busy'
    b.step(busy=0, core_wr_addr=1, core_wr_wdata=0x77, core_wr_we=1, core_wr_wmask=0b1111, core_rd_re=1,
           core_rd_addr=1, host_we=0, host_re=0)
    assert hr(b, OUT, 1) == 0, 'core write ignored outside busy'


def test_random():
    rng = random.Random(140)
    b = bench()
    mem = {IN: {}, OUT: {}}
    in_r = out_r = sel = 0
    for _ in range(3000):
        busy = rng.randint(0, 1)
        v = dict(busy=busy, host_addr=rng.randrange(1 << (BAW + 1)), host_wdata=rng.randrange(1 << 32),
                 host_we=rng.randint(0, 1), host_re=rng.randint(0, 1), core_rd_addr=rng.randrange(1 << BAW),
                 core_rd_re=rng.randint(0, 1), core_wr_addr=rng.randrange(1 << BAW),
                 core_wr_wdata=rng.randrange(1 << 32), core_wr_we=rng.randint(0, 1), core_wr_wmask=rng.randrange(16))
        b.step(**v)
        if busy:
            if v['core_rd_re']:
                in_r = mem[IN].get(v['core_rd_addr'], 0)
            if v['core_wr_we']:
                old = mem[OUT].get(v['core_wr_addr'], 0)
                m = sum(0xFF << (8 * l) for l in range(4) if (v['core_wr_wmask'] >> l) & 1)
                mem[OUT][v['core_wr_addr']] = (old & ~m) | (v['core_wr_wdata'] & m)
        else:
            bank, a = v['host_addr'] >> BAW, v['host_addr'] & ((1 << BAW) - 1)
            if v['host_re']:
                if bank:
                    out_r = mem[OUT].get(a, 0)
                else:
                    in_r = mem[IN].get(a, 0)
                sel = bank
            if v['host_we']:
                mem[bank][a] = v['host_wdata']
        assert b.get('core_rd_rdata') == in_r
        assert b.get('host_rdata') == (out_r if sel else in_r)


def test_x_layout_read_back_at_scheduler_addresses():
    rng = np.random.default_rng(141)
    for variant in ('OS', 'WS'):
        for M, K, Nd in [(4, 4, 4), (6, 9, 10), (5, 3, 7)]:
            W = rng.integers(-128, 128, (M, K))
            X = rng.integers(-128, 128, (K, Nd))
            b = bench()
            for a in range(1 << BAW):
                hw(b, IN, a, int(rng.integers(0, 1 << 32)))
            for a, lanes in layout.x_image(X, variant).items():
                hw(b, IN, a, pack(lanes, DW))
            for e in (OD.plan_os if variant == 'OS' else OD.plan_ws)(W, X, N, rng):
                if e['x_addr'] is not None:
                    b.step(busy=1, core_rd_addr=e['x_addr'], core_rd_re=1)
                    got = unpack(b.get('core_rd_rdata'), N, DW)
                    for l in range(N):
                        if (e['x_mask'] >> l) & 1:
                            assert got[l] == e['x'][l], (variant, M, K, Nd, e['x_addr'], l)


# --- Host driver of the datapath with memories -----------------------------------------------------------------

def datapath(variant):
    return OD.core(variant, N, mem=True)


def host_step(b, **v):
    """One host access cycle; busy is only forced if it is an input of the bench (without controller)."""
    if 'busy' in b.inputs:
        v['busy'] = 0
    b.step(**v)


def sig(b, name):
    """Value of a signal of the bench: input, output or internal wire."""
    return (b.w[name] if name in b.w else b.dut['w'][name]).get()


def load_wb(b, image):
    for a, lanes in image.items():
        host_step(b, wb_host_addr=a, wb_host_wdata=pack(lanes, DW), wb_host_we=1)
    host_step(b, wb_host_we=0)


def load_ab(b, bank, image):
    for a, lanes in image.items():
        host_step(b, ab_host_addr=(bank << BAW) | a, ab_host_wdata=pack(lanes, DW), ab_host_we=1)
    host_step(b, ab_host_we=0)


def read_ab(b, bank, addrs):
    words = {}
    for a in addrs:
        host_step(b, ab_host_addr=(bank << BAW) | a, ab_host_we=0, ab_host_re=1)
        words[a] = unpack(b.w['ab_host_rdata'].get(), N, DW)
    host_step(b, ab_host_re=0)
    return words


def load_bias(b, bias):
    for i, v in enumerate(bias):
        host_step(b, bias_host_addr=i, bias_host_wdata=int(v) & 0xFFFFFFFF, bias_host_we=1)
    host_step(b, bias_host_we=0)


def make_dirty(b, rng):
    """Fills the whole memory with non-zero values: out-of-bounds data must never count."""
    load_wb(b, {a: list(rng.integers(-128, 128, N)) for a in range(1 << BAW)})
    load_ab(b, IN, {a: list(rng.integers(-128, 128, N)) for a in range(1 << BAW)})


def run_layer(b, variant, W, X, bias, M, s, relu, load_x=True, on_cycle=None, log=None):
    """
    One layer Y = requant(ReLU(W·X + b)), driven as the host will drive it:
    loading of W (and of X if load_x), of the biases and of the configuration, GEMM (busy = 1), reading Y back.
    Returns (Y, number of cycles per phase).

    If `log` is a list, one record per cycle describing the memory accesses is appended to it:
      phase, wb_w (address written by the host into the weight_buffer), ab_w ((bank, address) written by the host),
      ab_r ((bank, address) read by the host), w_rd / x_rd (addresses read by the core), y_wr ((address, mask)
      written by the post_processing).
    """
    Mr, Nd = W.shape[0], X.shape[1]
    phase = ['loading']
    if log is not None:
        orig_step = b.step

        def step(**v):
            orig_step(**v)
            g = lambda n: b.w[n].get()
            log.append(dict(phase=phase[0], wb_w=g('wb_host_addr') if g('wb_host_we') else None,
                            ab_w=(g('ab_host_addr') >> BAW, g('ab_host_addr') & ((1 << BAW) - 1))
                            if g('ab_host_we') else None,
                            ab_r=(g('ab_host_addr') >> BAW, g('ab_host_addr') & ((1 << BAW) - 1))
                            if g('ab_host_re') else None,
                            w_rd=None, x_rd=None, y_wr=None))
        b.step = step

        def hook(c, bb):
            g = lambda n: sig(bb, n)
            log.append(dict(phase='compute', wb_w=None, ab_w=None, ab_r=None,
                            w_rd=g('w_addr') if g('w_re') else None, x_rd=g('x_addr') if g('x_re') else None,
                            y_wr=(g('y_addr'), g('y_wmask')) if g('y_we') else None))
            if on_cycle:
                on_cycle(c, bb)
    else:
        hook = on_cycle
    try:
        c0 = b.cycle
        load_wb(b, layout.w_image(W))
        if load_x:
            load_ab(b, IN, layout.x_image(X, variant))
        load_bias(b, list(bias)[:Mr])                       # only the biases of the useful rows
        c1 = b.cycle
        if b.ctrl:
            b.set(req_mult=M, req_shift=s, relu_en=relu)
            run_go(b, Mr, W.shape[1], Nd, on_cycle=hook)
        else:
            b.set(busy=1, req_mult=M, req_shift=s, relu_en=relu)
            plan = (OD.plan_os if variant == 'OS' else OD.plan_ws)(W, X, N, np.random.default_rng(0))
            OD.run_core(b, plan, on_cycle=hook)
        c2 = b.cycle
        phase[0] = 'readback'
        words = read_ab(b, OUT, layout.y_addresses(Mr, Nd, variant))
    finally:
        if log is not None:
            b.step = orig_step
    Y = np.array(layout.y_from_words(words, Mr, Nd, variant))
    return Y, dict(loading=c1 - c0, compute=c2 - c1, readback=b.cycle - c2)


def run_go(b, M, K, Nd, on_cycle=None, limit=100000):
    """
    With the controller: go pulse with the dimensions, then cycles up to the cycle where done is 1 (included).
    Returns the number of elapsed cycles, the go cycle included.
    """
    b.set(go=1, dim_m=M, dim_k=K, dim_n=Nd)
    for c in range(limit):
        b.sim.propagateAll()
        if on_cycle:
            on_cycle(c, b)
        finished = b.get('done')                                # done during this cycle (Moore output)
        b.sim.clk(1)
        b.cycle += 1
        b.set(go=0)
        if finished:
            return c + 1
    raise AssertionError('done never raised')


def copy_out_to_in(b, addrs):
    """Copy by the host of the words of Y (OUT bank) to the IN bank, for the next layer."""
    words = read_ab(b, OUT, addrs)
    load_ab(b, IN, words)
    return 3 * len(addrs) + 2


# --- End-to-end tests with memories --------------------------------------------------------------------------

def test_complete_datapath_with_memories():
    rng = np.random.default_rng(142)
    for variant in ('OS', 'WS'):
        b = datapath(variant)
        make_dirty(b, rng)
        for (Mr, K, Nd), relu in [((4, 4, 4), 1), ((9, 11, 6), 0), ((16, 16, 16), 1)]:
            W = rng.integers(-128, 128, (Mr, K))
            X = rng.integers(-128, 128, (K, Nd))
            bias = list(rng.integers(-20000, 20000, Mr))
            M, s = int(rng.integers(1, 1 << 16)), int(rng.integers(14, 24))
            Y, _ = run_layer(b, variant, W, X, bias, M, s, relu)
            assert (Y == PP.reference(W, X, bias + [0] * 64, M, s, relu)).all(), f'{variant} {Mr}×{K}×{Nd}'


def test_two_layers_chained_through_the_memory():
    """Y1 is copied as is from the OUT bank to the IN bank: it already has the X format of the variant (D6)."""
    rng = np.random.default_rng(143)
    X = rng.integers(-128, 128, (12, 6))
    W1, b1 = rng.integers(-128, 128, (10, 12)), list(rng.integers(-3000, 3000, 10))
    W2, b2 = rng.integers(-128, 128, (5, 10)), list(rng.integers(-3000, 3000, 5))
    Y1_ref = PP.reference(W1, X, b1 + [0] * 64, 300, 13, 1)
    Y2_ref = PP.reference(W2, Y1_ref, b2 + [0] * 64, 200, 12, 0)
    for variant in ('OS', 'WS'):
        b = datapath(variant)
        make_dirty(b, rng)
        Y1, _ = run_layer(b, variant, W1, X, b1, 300, 13, 1)
        assert (Y1 == Y1_ref).all(), variant
        copy_out_to_in(b, layout.y_addresses(10, 6, variant))
        Y2, _ = run_layer(b, variant, W2, Y1, b2, 200, 12, 0, load_x=False)
        assert (Y2 == Y2_ref).all(), variant


if __name__ == '__main__':
    sys.exit(run_tests(globals()))
