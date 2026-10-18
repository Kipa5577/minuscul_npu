"""
Functional test of the weight_buffer (specification: src/weight_buffer/interface.md).

  - write / read by the host, read latency 1, read data held as long as re = 0
  - a single active port at a time, selected by busy; contents kept when busy changes
  - comparison with a reference model on random accesses
  - D6 layout: W stored by the host (test/layout.py) then read back at the addresses emitted by the controller
    scheduler → each read returns exactly the word expected by the compute core

Run: python test/weight_buffer/test.py
"""

import sys
import pathlib
import random

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import Bench, pack, unpack, load_test_module, run_tests
import layout

import numpy as np
from src.weight_buffer.weight_buffer import WeightBuffer

OD = load_test_module('output_drain')
N, DW, BAW = 4, 8, 10


def bench():
    b = Bench(
        inputs={'busy': 1, 'host_addr': BAW, 'host_wdata': N * DW, 'host_we': 1, 'host_re': 1,
                'core_addr': BAW, 'core_re': 1},
        outputs={'host_rdata': N * DW, 'core_rdata': N * DW},
        build=lambda s, w: WeightBuffer(s, 'wbuf', w['busy'], w['host_addr'], w['host_wdata'], w['host_we'],
                                        w['host_re'], w['host_rdata'], w['core_addr'], w['core_re'], w['core_rdata']))
    b.set(busy=0)
    return b


def write(b, a, word):
    b.step(busy=0, host_addr=a, host_wdata=word, host_we=1, host_re=0)


def host_read(b, a):
    b.step(busy=0, host_addr=a, host_we=0, host_re=1)
    return b.get('host_rdata')


def core_read(b, a):
    b.step(busy=1, core_addr=a, core_re=1)
    return b.get('core_rdata')


# --- Tests ---------------------------------------------------------------------------------------------

def test_host_write_read():
    b = bench()
    write(b, 5, 0x11223344)
    write(b, 1023, 0xDEADBEEF)
    assert host_read(b, 5) == 0x11223344                       # visible in the cycle after the request
    assert host_read(b, 1023) == 0xDEADBEEF
    assert host_read(b, 0) == 0


def test_latency_1_and_hold():
    b = bench()
    write(b, 7, 0xCAFE)
    b.step(busy=0, host_addr=7, host_we=0, host_re=1)
    assert b.get('host_rdata') == 0xCAFE
    for a in (1, 2, 3):                                        # re = 0: the read data is held
        b.step(busy=0, host_addr=a, host_re=0)
        assert b.get('host_rdata') == 0xCAFE


def test_simultaneous_read_and_write():
    b = bench()
    write(b, 9, 111)
    b.step(busy=0, host_addr=9, host_wdata=222, host_we=1, host_re=1)
    assert b.get('host_rdata') == 111, 'the read returns the old value'
    assert host_read(b, 9) == 222


def test_active_port_selected_by_busy():
    b = bench()
    write(b, 3, 0xAAAA)
    # busy = 1: the host is ignored (write and read), the core reads
    b.step(busy=1, host_addr=3, host_wdata=0xBBBB, host_we=1, host_re=1, core_addr=3, core_re=0)
    assert b.get('host_rdata') == 0, 'host read ignored during busy'
    assert core_read(b, 3) == 0xAAAA, 'host write ignored during busy'
    # busy = 0: the core is ignored
    b.step(busy=0, host_we=0, host_re=0, core_addr=4, core_re=1)
    assert b.get('core_rdata') == 0xAAAA
    # the contents survive the changes of busy
    assert host_read(b, 3) == 0xAAAA


def test_random():
    rng = random.Random(130)
    b, ref, rdata = bench(), {}, 0
    for _ in range(3000):
        busy, a = rng.randint(0, 1), rng.randrange(1 << BAW)
        we, re, word = rng.randint(0, 1), rng.randint(0, 1), rng.randrange(1 << 32)
        b.step(busy=busy, host_addr=a, host_wdata=word, host_we=we, host_re=re, core_addr=a, core_re=re)
        if busy and re:
            rdata = ref.get(a, 0)
        elif not busy:
            if re:
                rdata = ref.get(a, 0)
            if we:
                ref[a] = word
        assert b.get('host_rdata') == b.get('core_rdata') == rdata


def test_w_layout_read_back_at_scheduler_addresses():
    """The controller (here its reference scheduler) reads exactly the words the core expects."""
    rng = np.random.default_rng(131)
    for variant in ('OS', 'WS'):
        for M, K, Nd in [(4, 4, 4), (10, 7, 3), (3, 13, 9)]:
            W = rng.integers(-128, 128, (M, K))
            X = rng.integers(-128, 128, (K, Nd))
            b = bench()
            for a in range(1 << BAW):                          # "dirty" memory
                write(b, a, int(rng.integers(0, 1 << 32)))
            for a, lanes in layout.w_image(W).items():
                write(b, a, pack(lanes, DW))
            plan = (OD.plan_os if variant == 'OS' else OD.plan_ws)(W, X, N, rng)
            reads = 0
            for e in plan:
                if e['w_addr'] is not None:
                    got = unpack(core_read(b, e['w_addr']), N, DW)
                    for l in range(N):
                        if (e['w_mask'] >> l) & 1:
                            assert got[l] == e['w'][l], (variant, M, K, Nd, e['w_addr'], l)
                    reads += 1
            assert reads > 0


if __name__ == '__main__':
    sys.exit(run_tests(globals()))
