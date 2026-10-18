"""
Functional test of the data_feeder (specification: src/data_feeder/interface.md).

  - latencies: lane l of a skewed stream comes out 1 + l cycles after emission; W in WS and ctl after 1 cycle
  - masking: out-of-bounds lanes come out as 0, whatever the contents of the memory
  - cycle-by-cycle comparison with a reference model
  - data_feeder → pe_array chain: complete GEMM with dimensions that are not multiples of N and "dirty" memory

Timing convention: the controller emits (masks, ctl, address) at cycle e; the SRAM returns the word at cycle e + 1.
The outputs are read DURING each cycle (Bench.observe), because lane 0 is combinational.

Run: python test/data_feeder/test.py
"""

import sys
import pathlib
import random

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import Bench, pack, unpack, run_tests

import numpy as np
from src.data_feeder.data_feeder import DataFeeder_OS, DataFeeder_WS
from src.pe_array.pe_array import PE_Array_OS, PE_Array_WS

DW, AW = 8, 32
VARIANTS = {'OS': DataFeeder_OS, 'WS': DataFeeder_WS}


# --- Bench --------------------------------------------------------------------------------

def bench(variant='OS', N=4):
    cls = VARIANTS[variant]
    b = Bench(
        inputs={'rst_n': 1, 'w_rdata': N * DW, 'x_rdata': N * DW, 'w_mask': N, 'x_mask': N, 'ctl_in': 1},
        outputs={'w_arr': N * DW, 'x_arr': N * DW, 'ctl_arr': 1},
        build=lambda s, w: cls(s, 'feeder', w['rst_n'], w['w_rdata'], w['x_rdata'], w['w_mask'], w['x_mask'],
                               w['ctl_in'], w['w_arr'], w['x_arr'], w['ctl_arr']))
    b.N = N
    b.step(rst_n=0)
    b.set(rst_n=1)
    return b


def run(b, emissions, extra=8):
    """
    emissions[e] = dict(w=[...], x=[...], w_mask=int, x_mask=int, ctl=int) emitted at cycle e.
    The SRAM returns the word of emission e at cycle e + 1. Returns the outputs read during each cycle.
    """
    N = b.N
    idle = dict(w=[0] * N, x=[0] * N, w_mask=0, x_mask=0, ctl=0)
    hist = []
    prev = idle
    for e in range(len(emissions) + extra):
        cur = emissions[e] if e < len(emissions) else idle
        snap = b.observe(w_rdata=pack(prev['w'], DW), x_rdata=pack(prev['x'], DW),
                         w_mask=cur['w_mask'], x_mask=cur['x_mask'], ctl_in=cur['ctl'])
        hist.append(dict(w=unpack(snap['w_arr'], N, DW), x=unpack(snap['x_arr'], N, DW), ctl=snap['ctl_arr']))
        prev = cur
    return hist


def mask_of(n_valid, N):
    return (1 << max(0, min(n_valid, N))) - 1


# --- Reference model -------------------------------------------------------------------------

class RefFeeder:
    def __init__(self, N, skew_w):
        self.N, self.skew_w = N, skew_w
        self.wm = self.xm = self.ctl = 0
        self.wc = [[0] * l for l in range(N)]
        self.xc = [[0] * l for l in range(N)]

    def cycle(self, w_rdata, x_rdata, w_mask, x_mask, ctl):
        """Returns the outputs during the cycle, then updates the state (edge)."""
        N = self.N
        wmk = [w_rdata[l] if (self.wm >> l) & 1 else 0 for l in range(N)]
        xmk = [x_rdata[l] if (self.xm >> l) & 1 else 0 for l in range(N)]
        sk = lambda m, c: [m[0]] + [c[l][-1] for l in range(1, N)]
        out = dict(w=sk(wmk, self.wc) if self.skew_w else wmk, x=sk(xmk, self.xc), ctl=self.ctl)
        for l in range(1, N):
            if self.skew_w:
                self.wc[l] = [wmk[l]] + self.wc[l][:-1]
            self.xc[l] = [xmk[l]] + self.xc[l][:-1]
        self.wm, self.xm, self.ctl = w_mask, x_mask, ctl
        return out


# --- Tests ------------------------------------------------------------------------------------

def test_reset():
    for v in VARIANTS:
        b = bench(v)
        run(b, [dict(w=[9] * 4, x=[9] * 4, w_mask=15, x_mask=15, ctl=1)] * 3, extra=0)
        b.step(rst_n=0)
        snap = b.observe(rst_n=1, w_rdata=pack([5] * 4, DW), x_rdata=pack([5] * 4, DW))
        assert snap['w_arr'] == 0 and snap['x_arr'] == 0 and snap['ctl_arr'] == 0, v


def test_latency_per_lane():
    word = [11, -22, 33, -44]
    for v in VARIANTS:
        for N in (1, 2, 4, 8):
            w = (word * 2)[:N]
            b = bench(v, N)
            h = run(b, [dict(w=w, x=w, w_mask=mask_of(N, N), x_mask=mask_of(N, N), ctl=1)])
            for l in range(N):
                lat_x = 1 + l
                lat_w = 1 + l if v == 'OS' else 1
                for c, snap in enumerate(h):
                    assert snap['x'][l] == (w[l] if c == lat_x else 0), f'{v} N={N} X lane {l} cycle {c}'
                    assert snap['w'][l] == (w[l] if c == lat_w else 0), f'{v} N={N} W lane {l} cycle {c}'
            assert [s['ctl'] for s in h[:3]] == [0, 1, 0], v


def test_masking():
    garbage = [101, -102, 103, -104]
    for v in VARIANTS:
        for wm, xm in [(0b0101, 0b1010), (0b0111, 0b0001), (0, 0), (0b1111, 0b1111)]:
            b = bench(v)
            h = run(b, [dict(w=garbage, x=garbage, w_mask=wm, x_mask=xm, ctl=0)])
            for l in range(4):
                lat_w = 1 + l if v == 'OS' else 1
                exp_w = garbage[l] if (wm >> l) & 1 else 0
                exp_x = garbage[l] if (xm >> l) & 1 else 0
                assert h[lat_w]['w'][l] == exp_w, f'{v} W lane {l} mask {wm:04b}'
                assert h[1 + l]['x'][l] == exp_x, f'{v} X lane {l} mask {xm:04b}'


def test_idle_injects_zeros():
    for v in VARIANTS:
        b = bench(v)
        h = run(b, [dict(w=[7, 7, 7, 7], x=[7, 7, 7, 7], w_mask=0, x_mask=0, ctl=0)] * 10)
        assert all(s['w'] == [0] * 4 and s['x'] == [0] * 4 for s in h), v


def test_random_cycle_by_cycle():
    for v in VARIANTS:
        random.seed(30)
        N = 4
        b, ref = bench(v, N), RefFeeder(N, skew_w=(v == 'OS'))
        for _ in range(1000):
            w = [random.randint(-128, 127) for _ in range(N)]
            x = [random.randint(-128, 127) for _ in range(N)]
            wm, xm, ctl = random.randrange(16), random.randrange(16), random.randint(0, 1)
            snap = b.observe(w_rdata=pack(w, DW), x_rdata=pack(x, DW), w_mask=wm, x_mask=xm, ctl_in=ctl)
            exp = ref.cycle(w, x, wm, xm, ctl)
            got = dict(w=unpack(snap['w_arr'], N, DW), x=unpack(snap['x_arr'], N, DW), ctl=snap['ctl_arr'])
            assert got == exp, f'{v} cycle {b.cycle}: {got} ≠ {exp}'


# --- data_feeder → pe_array chain --------------------------------------------------------------

def chain(variant, N=4):
    feeder, array = (DataFeeder_OS, PE_Array_OS) if variant == 'OS' else (DataFeeder_WS, PE_Array_WS)
    out = {'acc': N * N * AW} if variant == 'OS' else {'psum_south': N * AW}

    def build(s, w):
        wa, xa, ca = s.wire('w_arr', N * DW), s.wire('x_arr', N * DW), s.wire('ctl_arr')
        feeder(s, 'feeder', w['rst_n'], w['w_rdata'], w['x_rdata'], w['w_mask'], w['x_mask'], w['ctl_in'],
               wa, xa, ca)
        if variant == 'OS':
            return PE_Array_OS(s, 'arr', w['rst_n'], wa, xa, ca, w['acc'], N=N)
        return PE_Array_WS(s, 'arr', w['rst_n'], xa, wa, ca, w['psum_south'], N=N)

    b = Bench(inputs={'rst_n': 1, 'w_rdata': N * DW, 'x_rdata': N * DW, 'w_mask': N, 'x_mask': N, 'ctl_in': 1},
              outputs=out, build=build)
    b.N = N
    b.step(rst_n=0)
    b.set(rst_n=1)
    return b


def run_chain(b, emissions, extra):
    N = b.N
    idle = dict(w=[0] * N, x=[0] * N, w_mask=0, x_mask=0, ctl=0)
    prev, hist = idle, []
    for e in range(len(emissions) + extra):
        cur = emissions[e] if e < len(emissions) else idle
        b.step(w_rdata=pack(prev['w'], DW), x_rdata=pack(prev['x'], DW),
               w_mask=cur['w_mask'], x_mask=cur['x_mask'], ctl_in=cur['ctl'])
        hist.append(b.w['acc' if 'acc' in b.w else 'psum_south'].get())
        prev = cur
    return hist


def test_chain_os_gemm_edges_and_dirty_memory():
    rng = np.random.default_rng(31)
    N = 4
    for M, K, Nd in [(4, 6, 4), (3, 5, 2), (1, 1, 1), (2, 9, 3)]:
        W = rng.integers(-128, 128, (M, K))
        X = rng.integers(-128, 128, (K, Nd))
        # memory words: out-of-bounds lanes filled with non-zero values ("dirty" memory)
        Wm = rng.integers(-128, 128, (N, K))
        Xm = rng.integers(-128, 128, (K, N))
        Wm[:M], Xm[:, :Nd] = W, X
        em = [dict(w=[0] * N, x=[0] * N, w_mask=0, x_mask=0, ctl=1)]                    # clear
        em += [dict(w=list(Wm[:, k]), x=list(Xm[k]), w_mask=mask_of(M, N), x_mask=mask_of(Nd, N), ctl=0)
               for k in range(K)]
        h = run_chain(chain('OS', N), em, extra=3 * N)
        acc = np.array(unpack(h[-1], N * N, AW)).reshape(N, N)
        assert (acc[:M, :Nd] == W @ X).all(), f'M={M} K={K} N={Nd}'
        assert not acc[M:].any() and not acc[:, Nd:].any(), 'out-of-bounds PEs must stay at 0'


def test_chain_ws_gemm_edges_and_dirty_memory():
    rng = np.random.default_rng(32)
    N = 4
    for M, K, J in [(4, 4, 6), (3, 2, 5), (1, 1, 1), (2, 3, 7)]:
        W = rng.integers(-128, 128, (M, K))
        X = rng.integers(-128, 128, (K, J))
        Wm = rng.integers(-128, 128, (N, N))
        Xm = rng.integers(-128, 128, (N, J))
        Wm[:M, :K], Xm[:K] = W, X
        # preload: array row k = N−1 is presented first
        em = [dict(w=list(Wm[:, k]), x=[0] * N, w_mask=mask_of(M, N) if k < K else 0, x_mask=0, ctl=1)
              for k in reversed(range(N))]
        em += [dict(w=[0] * N, x=list(Xm[:, j]), w_mask=0, x_mask=mask_of(K, N), ctl=0) for j in range(J)]
        h = run_chain(chain('WS', N), em, extra=3 * N)
        s0 = N                                    # emission cycle of column j = 0
        Y = np.array([[unpack(h[s0 + 1 + j + i + N - 1], N, AW)[i] for j in range(J)] for i in range(N)])
        assert (Y[:M] == W @ X).all(), f'M={M} K={K} J={J}'
        assert not Y[M:].any(), 'out-of-bounds array columns must output 0'


if __name__ == '__main__':
    sys.exit(run_tests(globals()))
