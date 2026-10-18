"""
Functional test of the pe_array (specification: src/pe_array/interface.md).

  - wiring: cycle-by-cycle comparison with an array of pe reference models
  - contract: complete GEMM with the staircase injection described in the specification,
    checking the result AND the exact cycle at which it becomes available

Run: python test/pe_array/test.py
"""

import sys
import pathlib
import random

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import Bench, pack, unpack, load_test_module, run_tests

import numpy as np
from src.pe_array.pe_array import PE_Array_OS, PE_Array_WS

pe_test = load_test_module('pe')
DW, AW = 8, 32


# --- Benches ---------------------------------------------------------------------

def bench_os(N=4):
    b = Bench(
        inputs={'rst_n': 1, 'w_west': N * DW, 'x_north': N * DW, 'clear': 1},
        outputs={'acc': N * N * AW},
        build=lambda s, w: PE_Array_OS(s, 'arr', w['rst_n'], w['w_west'], w['x_north'], w['clear'],
                                       w['acc'], N=N))
    b.N = N
    b.step(rst_n=0)
    b.set(rst_n=1)
    return b


def bench_ws(N=4):
    b = Bench(
        inputs={'rst_n': 1, 'x_west': N * DW, 'w_north': N * DW, 'load': 1},
        outputs={'psum_south': N * AW},
        build=lambda s, w: PE_Array_WS(s, 'arr', w['rst_n'], w['x_west'], w['w_north'], w['load'],
                                       w['psum_south'], N=N))
    b.N = N
    b.step(rst_n=0)
    b.set(rst_n=1)
    return b


def acc_matrix(b):
    return np.array(unpack(b.w['acc'].get(), b.N * b.N, AW)).reshape(b.N, b.N)


def psum_south(b):
    return unpack(b.w['psum_south'].get(), b.N, AW)


# --- Input scheduling (contracts of the specification) ---------------------------------

def os_lanes(W, X, c):
    """Cycle c (relative to t0): lane i of w_west = W[i][c−i], lane j of x_north = X[c−j][j]."""
    N, K = W.shape
    w = [W[i][c - i] if 0 <= c - i < K else 0 for i in range(N)]
    x = [X[c - j][j] if 0 <= c - j < K else 0 for j in range(N)]
    return pack(w, DW), pack(x, DW)


def run_os_tile(b, W, X):
    """Injects a tile; returns the list of acc matrices after each cycle."""
    N, K = W.shape
    hist = []
    for c in range(K + 2 * N):
        wl, xl = os_lanes(W, X, c)
        b.step(w_west=wl, x_north=xl, clear=0)
        hist.append(acc_matrix(b))
    return hist


def ws_preload(b, W):
    """W is M×K = N×N: array row k receives W[:, k]. k = N−1 is presented first."""
    N = b.N
    for k in reversed(range(N)):
        b.step(w_north=pack(W[:, k], DW), x_west=0, load=1)


def ws_x_lanes(X, c):
    """Cycle c (relative to t0): lane k of x_west = X[k][c−k]."""
    N, J = X.shape
    return pack([X[k][c - k] if 0 <= c - k < J else 0 for k in range(N)], DW)


def run_ws_stream(b, X, extra=0):
    """Sends the J columns of X; returns the list of psum_south after each cycle."""
    N, J = X.shape
    hist = []
    for c in range(J + 2 * N + extra):
        b.step(x_west=ws_x_lanes(X, c), w_north=0, load=0)
        hist.append(psum_south(b))
    return hist


def ws_outputs(hist, N, J):
    """Lane i carries Y[i][j] at cycle t0 + j + i + N, hence after the step of index j + i + N − 1."""
    return np.array([[hist[j + i + N - 1][i] for j in range(J)] for i in range(N)])


# --- Reference models (array of reference PEs) ----------------------------------------

class RefArray_OS:
    def __init__(self, N):
        self.N = N
        self.pe = [[pe_test.RefPE_OS() for _ in range(N)] for _ in range(N)]

    def step(self, rst_n, w_lanes, x_lanes, clear):
        N = self.N
        ins = [[(w_lanes[i] if j == 0 else self.pe[i][j - 1].w_r,
                 x_lanes[j] if i == 0 else self.pe[i - 1][j].x_r)
                for j in range(N)] for i in range(N)]
        for i in range(N):
            for j in range(N):
                self.pe[i][j].step(rst_n, *ins[i][j], clear)

    def acc(self):
        return np.array([[p.acc for p in row] for row in self.pe])


class RefArray_WS:
    def __init__(self, N):
        self.N = N
        self.pe = [[pe_test.RefPE_WS() for _ in range(N)] for _ in range(N)]

    def step(self, rst_n, x_lanes, w_lanes, load):
        N = self.N
        ins = [[(x_lanes[k] if i == 0 else self.pe[k][i - 1].x_r,
                 0 if k == 0 else self.pe[k - 1][i].psum,
                 w_lanes[i] if k == 0 else self.pe[k - 1][i].w_r)
                for i in range(N)] for k in range(N)]
        for k in range(N):
            for i in range(N):
                self.pe[k][i].step(rst_n, *ins[k][i], load)

    def south(self):
        return [p.psum for p in self.pe[self.N - 1]]


# --- OS tests ----------------------------------------------------------------------------

def test_os_reset():
    b = bench_os()
    b.step(w_west=pack([5] * 4, DW), x_north=pack([7] * 4, DW), clear=0)
    b.step()
    assert acc_matrix(b).any()
    b.step(rst_n=0)
    assert not acc_matrix(b).any()


def test_os_gemm_contract():
    rng = np.random.default_rng(10)
    for N in (1, 2, 3, 4, 8):
        for K in (1, 5, 16):
            W = rng.integers(-128, 128, (N, K))
            X = rng.integers(-128, 128, (K, N))
            W[N - 1][K - 1] = X[K - 1][N - 1] = 3          # last term non-zero
            b = bench_os(N)
            hist = run_os_tile(b, W, X)
            fin = K + 2 * N - 3                            # step after which acc is complete
            assert (hist[fin] == W @ X).all(), f'N={N} K={K}: wrong result'
            if fin > 0:
                assert not (hist[fin - 1] == W @ X).all(), f'N={N} K={K}: result available too early'
            assert (hist[-1] == W @ X).all(), f'N={N} K={K}: result not held'


def test_os_two_tiles_with_clear():
    rng = np.random.default_rng(11)
    N, K = 4, 8
    b = bench_os(N)
    for _ in range(2):
        W = rng.integers(-128, 128, (N, K))
        X = rng.integers(-128, 128, (K, N))
        b.step(w_west=0, x_north=0, clear=1)
        hist = run_os_tile(b, W, X)
        assert (hist[-1] == W @ X).all()


def test_os_extremes():
    N, K = 4, 64
    W = np.full((N, K), -128)
    X = np.full((K, N), -128)
    hist = run_os_tile(bench_os(N), W, X)
    assert (hist[-1] == K * 16384).all()


def test_os_random_cycle_by_cycle():
    random.seed(12)
    N = 4
    b, ref = bench_os(N), RefArray_OS(N)
    ref.step(0, [0] * N, [0] * N, 0)
    for _ in range(600):
        w = [random.randint(-128, 127) for _ in range(N)]
        x = [random.randint(-128, 127) for _ in range(N)]
        clear = int(random.random() < 0.05)
        b.step(w_west=pack(w, DW), x_north=pack(x, DW), clear=clear)
        ref.step(1, w, x, clear)
        assert (acc_matrix(b) == ref.acc()).all(), f'cycle {b.cycle}'


# --- WS tests ----------------------------------------------------------------------------

def test_ws_reset():
    b = bench_ws()
    ws_preload(b, np.ones((4, 4), dtype=int))
    b.step(x_west=pack([3] * 4, DW), load=0)
    b.step(rst_n=0)
    assert psum_south(b) == [0] * 4
    assert all(pe.w_r == 0 for row in b.dut.pes for pe in row)


def test_ws_preload():
    rng = np.random.default_rng(13)
    for N in (1, 2, 4, 8):
        W = rng.integers(-128, 128, (N, N))
        b = bench_ws(N)
        ws_preload(b, W)
        for k in range(N):
            for i in range(N):
                got = unpack(b.dut.pes[k][i].w_load_out.get(), 1, DW)[0]
                assert got == W[i][k], f'N={N} PE({k},{i}): {got} ≠ W[{i}][{k}] = {W[i][k]}'


def test_ws_gemm_contract():
    rng = np.random.default_rng(14)
    for N in (1, 2, 3, 4, 8):
        for J in (1, 5, 16):
            W = rng.integers(-128, 128, (N, N))
            X = rng.integers(-128, 128, (N, J))
            b = bench_ws(N)
            ws_preload(b, W)
            hist = run_ws_stream(b, X)
            assert (ws_outputs(hist, N, J) == W @ X).all(), f'N={N} J={J}'


def test_ws_minimum_gap_before_reload():
    """
    Between the last column of X (lane 0, cycle a) and the first load of the next preload,
    at least 2N − 2 cycles are needed (i.e. 2N − 3 cycles of FLUSH on the controller side).
    With 2N − 2 the result is correct; with one cycle less, the last column is wrong.
    """
    rng = np.random.default_rng(15)
    for N in (2, 3, 4, 8):
        J = 6
        W1 = rng.integers(1, 128, (N, N))
        W2 = rng.integers(-128, 0, (N, N))
        X1 = rng.integers(1, 128, (N, J))
        X2 = rng.integers(1, 128, (N, J))
        for gap, expect_correct in ((2 * N - 2, True), (2 * N - 3, False)):
            b = bench_ws(N)
            ws_preload(b, W1)
            # stream X1 then wait: lane 0 receives the last column at cycle a = J − 1
            hist = []
            for c in range(J - 1 + gap):
                b.step(x_west=ws_x_lanes(X1, c), w_north=0, load=0)
                hist.append(psum_south(b))
            # preload of W2 (lane k of x_west continues the staircase of X1 meanwhile)
            for n, k in enumerate(reversed(range(N))):
                c = J - 1 + gap + n
                b.step(x_west=ws_x_lanes(X1, c), w_north=pack(W2[:, k], DW), load=1)
                hist.append(psum_south(b))
            for _ in range(2 * N):
                b.step(x_west=0, w_north=0, load=0)
                hist.append(psum_south(b))
            correct = (ws_outputs(hist, N, J) == W1 @ X1).all()
            assert correct == expect_correct, f'N={N} gap={gap}: correct={correct}, expected {expect_correct}'
            # and the array does hold W2 afterwards
            hist2 = run_ws_stream(b, X2)
            assert (ws_outputs(hist2, N, J) == W2 @ X2).all()


def test_ws_random_cycle_by_cycle():
    random.seed(16)
    N = 4
    b, ref = bench_ws(N), RefArray_WS(N)
    ref.step(0, [0] * N, [0] * N, 0)
    for _ in range(600):
        x = [random.randint(-128, 127) for _ in range(N)]
        w = [random.randint(-128, 127) for _ in range(N)]
        load = int(random.random() < 0.15)
        b.step(x_west=pack(x, DW), w_north=pack(w, DW), load=load)
        ref.step(1, x, w, load)
        assert psum_south(b) == ref.south(), f'cycle {b.cycle}'


if __name__ == '__main__':
    sys.exit(run_tests(globals()))
