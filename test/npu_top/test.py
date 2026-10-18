"""
Functional test of npu_top: the complete chip, driven only through its 32-bit bus.

The class Npu plays the role of the host software: it knows only the memory map (architecture.md, D7)
and the data layout (D6, test/layout.py). Typical sequence (src/npu_top/interface.md):
  1. write W into WEIGHT, X into ACT_IN and b into BIAS;
  2. write DIM_M, DIM_K, DIM_N, REQ_MULT, REQ_SHIFT;
  3. write CTRL = GO | RELU_EN;
  4. wait for STATUS.DONE (or irq);
  5. read Y from ACT_OUT, and CYCLES.

Tests: ID register, end-to-end GEMM in both variants ("dirty" memory, dimensions that are not multiples of 4),
two chained layers, CYCLES and irq, invalid dimensions, accesses ignored during a GEMM, default
requantization, OS and WS identical.

Run: python test/npu_top/test.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import Bench, pack, unpack, load_test_module, run_tests
import layout

import numpy as np
from src.npu_top.npu_top import NpuTop

PP = load_test_module('post_processing')
CT = load_test_module('controller')

# memory map (addresses of 32-bit words)
ID, CTRL, STATUS, DIM_M, DIM_K, DIM_N, REQ_MULT, REQ_SHIFT, CYCLES = range(9)
WEIGHT, ACT_IN, ACT_OUT, BIAS = 0x0400, 0x0800, 0x0C00, 0x1000
N, DW = 4, 8


class Npu:
    """Host software: drives the chip through its bus (one access per cycle, read with latency 1)."""

    def __init__(self, dataflow='OS'):
        self.dataflow = dataflow
        self.b = Bench(inputs=dict(rst_n=1, bus_addr=13, bus_wdata=32, bus_we=1, bus_re=1),
                       outputs=dict(bus_rdata=32, irq=1),
                       build=lambda s, w: NpuTop(s, 'npu', w['rst_n'], w['bus_addr'], w['bus_wdata'], w['bus_we'],
                                                 w['bus_re'], w['bus_rdata'], w['irq'], dataflow=dataflow))
        self.b.step(rst_n=0)
        self.b.set(rst_n=1)

    @property
    def cycle(self):
        return self.b.cycle

    @property
    def irq(self):
        return self.b.get('irq')

    # --- bus access --------------------------------------------------------------------------

    def write(self, addr, value):
        self.b.step(bus_addr=addr, bus_wdata=value & 0xFFFFFFFF, bus_we=1, bus_re=0)
        self.b.set(bus_we=0)

    def read(self, addr):
        self.b.step(bus_addr=addr, bus_we=0, bus_re=1)
        self.b.set(bus_re=0)
        return self.b.get('bus_rdata')

    # --- driver -------------------------------------------------------------------------------

    def load_w(self, W):
        for a, lanes in layout.w_image(W).items():
            self.write(WEIGHT + a, pack(lanes, DW))

    def load_x(self, X):
        for a, lanes in layout.x_image(X, self.dataflow).items():
            self.write(ACT_IN + a, pack(lanes, DW))

    def load_bias(self, bias):
        for i, v in enumerate(bias):
            self.write(BIAS + i, int(v))

    def configure(self, M, K, Nd, mult, shift):
        for r, v in ((DIM_M, M), (DIM_K, K), (DIM_N, Nd), (REQ_MULT, mult), (REQ_SHIFT, shift)):
            self.write(r, v)

    def start(self, relu):
        self.write(CTRL, 1 | (int(relu) << 1))

    def wait_done(self, limit=200000):
        for _ in range(limit):
            if self.read(STATUS) & 0b10:
                return
        raise AssertionError('DONE never raised')

    def read_y(self, M, Nd):
        words = {a: unpack(self.read(ACT_OUT + a), N, DW) for a in layout.y_addresses(M, Nd, self.dataflow)}
        return np.array(layout.y_from_words(words, M, Nd, self.dataflow))

    def copy_out_to_in(self, M, Nd):
        for a in layout.y_addresses(M, Nd, self.dataflow):
            self.write(ACT_IN + a, self.read(ACT_OUT + a))

    def gemm(self, W, X, bias, mult=1, shift=0, relu=False, load_x=True):
        """One complete layer, as the host would do it. Returns (Y, CYCLES, bus cycles per phase)."""
        M, K = W.shape
        Nd = X.shape[1]
        c0 = self.cycle
        self.load_w(W)
        if load_x:
            self.load_x(X)
        self.load_bias(bias[:M])
        self.configure(M, K, Nd, mult, shift)
        c1 = self.cycle
        self.start(relu)
        self.wait_done()
        c2 = self.cycle
        cycles = self.read(CYCLES)
        Y = self.read_y(M, Nd)
        return Y, cycles, dict(loading=c1 - c0, compute=c2 - c1, readback=self.cycle - c2)

    def make_dirty(self, rng):
        """Fills the memory with "dirty" data: out-of-bounds data must never count."""
        for a in range(1 << 10):
            self.write(WEIGHT + a, int(rng.integers(0, 1 << 32)))
            self.write(ACT_IN + a, int(rng.integers(0, 1 << 32)))


def plan_len(variant, M, K, Nd):
    return len(CT.plan(variant, M, K, Nd))


# --- Tests --------------------------------------------------------------------------------------------------------

def test_id_register():
    for v, bit in (('OS', 0), ('WS', 1)):
        idv = Npu(v).read(ID)
        assert (idv & 0xF, (idv >> 4) & 1, (idv >> 8) & 0xFF, idv >> 24) == (4, bit, 64, 1), v


def test_gemm_end_to_end():
    rng = np.random.default_rng(190)
    for v in ('OS', 'WS'):
        npu = Npu(v)
        npu.make_dirty(rng)
        for (M, K, Nd), relu in [((4, 4, 4), 1), ((10, 7, 9), 0), ((16, 16, 16), 1), ((1, 33, 2), 0)]:
            W, X = rng.integers(-128, 128, (M, K)), rng.integers(-128, 128, (K, Nd))
            bias = list(rng.integers(-20000, 20000, M))
            mult, shift = int(rng.integers(1, 1 << 16)), int(rng.integers(14, 24))
            Y, _, _ = npu.gemm(W, X, bias, mult, shift, relu)
            assert (Y == PP.reference(W, X, bias + [0] * 64, mult, shift, relu)).all(), f'{v} {M}×{K}×{Nd}'


def test_two_chained_layers():
    rng = np.random.default_rng(191)
    X = rng.integers(-128, 128, (12, 6))
    W1, b1 = rng.integers(-128, 128, (10, 12)), list(rng.integers(-3000, 3000, 10))
    W2, b2 = rng.integers(-128, 128, (5, 10)), list(rng.integers(-3000, 3000, 5))
    Y1_ref = PP.reference(W1, X, b1 + [0] * 64, 300, 13, 1)
    Y2_ref = PP.reference(W2, Y1_ref, b2 + [0] * 64, 200, 12, 0)
    for v in ('OS', 'WS'):
        npu = Npu(v)
        Y1, _, _ = npu.gemm(W1, X, b1, 300, 13, True)
        npu.copy_out_to_in(10, 6)
        Y2, _, _ = npu.gemm(W2, Y1, b2, 200, 12, False, load_x=False)
        assert (Y1 == Y1_ref).all() and (Y2 == Y2_ref).all(), v


def test_cycles_and_irq():
    rng = np.random.default_rng(192)
    for v in ('OS', 'WS'):
        npu = Npu(v)
        W, X = rng.integers(-128, 128, (8, 8)), rng.integers(-128, 128, (8, 8))
        assert npu.irq == 0
        _, cycles, _ = npu.gemm(W, X, [0] * 8)
        assert cycles == plan_len(v, 8, 8, 8) + 1, (v, cycles)
        assert npu.irq == 1 and npu.read(STATUS) == 0b10
        npu.start(False)
        assert npu.irq == 0, 'GO clears DONE and irq'
        assert npu.read(STATUS) & 1, 'BUSY during the computation'
        npu.wait_done()
        assert npu.irq == 1


def test_invalid_dimensions():
    for v in ('OS', 'WS'):
        npu = Npu(v)
        npu.write(ACT_OUT + 0, 0x12345678)
        npu.configure(0, 4, 4, 1, 0)
        npu.start(False)
        npu.wait_done()
        assert npu.read(CYCLES) == 1, 'immediate DONE'
        assert npu.read(ACT_OUT + 0) == 0x12345678, 'nothing is written'


def test_accesses_ignored_during_a_gemm():
    rng = np.random.default_rng(193)
    for v in ('OS', 'WS'):
        npu = Npu(v)
        W, X = rng.integers(-128, 128, (8, 12)), rng.integers(-128, 128, (12, 8))
        npu.load_w(W)
        npu.load_x(X)
        npu.load_bias([0] * 8)
        npu.configure(8, 12, 8, 1, 6)
        npu.start(True)
        w0 = pack(layout.w_image(W)[0], DW)
        # during the computation: attempts to write the configuration and the memories
        npu.write(DIM_M, 1)
        npu.write(REQ_SHIFT, 0)
        npu.write(WEIGHT + 0, 0)
        npu.write(BIAS + 0, 99999)
        npu.wait_done()
        assert npu.read(DIM_M) == 8 and npu.read(REQ_SHIFT) == 6, 'configuration intact'
        assert npu.read(WEIGHT + 0) == w0, 'weight memory intact'
        assert (npu.read_y(8, 8) == PP.reference(W, X, [0] * 64, 1, 6, 1)).all(), 'correct result'


def test_default_requantization():
    rng = np.random.default_rng(194)
    W, X = rng.integers(-3, 4, (4, 4)), rng.integers(-3, 4, (4, 4))
    bias = list(rng.integers(-20, 20, 4))
    npu = Npu('OS')
    npu.load_w(W)
    npu.load_x(X)
    npu.load_bias(bias)
    for r, v in ((DIM_M, 4), (DIM_K, 4), (DIM_N, 4)):
        npu.write(r, v)
    npu.start(False)
    npu.wait_done()
    expected = np.clip(W @ X + np.array(bias)[:, None], -128, 127)
    assert (npu.read_y(4, 4) == expected).all(), 'reset: y = sat8(acc + b)'


def test_os_and_ws_identical():
    rng = np.random.default_rng(195)
    W, X = rng.integers(-128, 128, (9, 14)), rng.integers(-128, 128, (14, 11))
    bias = list(rng.integers(-5000, 5000, 9))
    Y = {v: Npu(v).gemm(W, X, bias, 777, 16, True)[0] for v in ('OS', 'WS')}
    assert (Y['OS'] == Y['WS']).all()


if __name__ == '__main__':
    sys.exit(run_tests(globals()))
