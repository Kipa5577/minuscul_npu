"""
Functional test of the host_interface (specification: src/host_interface/interface.md).

The memories and the controller are simulated by the bench (busy, done, cycles, *_rdata driven by hand).

  - ID register, reset values, write / read-back of the registers and width of their fields
  - address decoding: each window controls only its own memory, unassigned addresses control nothing
  - one-cycle go pulse, GO reads back as 0; DONE latched, cleared to 0 by GO; irq = DONE
  - during busy, configuration writes and GO ignored, reads still possible
  - read latency 1 for all targets (registers and memories), read of an empty address = 0

Run: python test/host_interface/test.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import Bench, run_tests

from src.host_interface.host_interface import HostInterface

BAW, IW = 10, 6
REG, WEIGHT, ACT_IN, ACT_OUT, BIAS = 0x0000, 0x0400, 0x0800, 0x0C00, 0x1000
ID, CTRL, STATUS, DIM_M, DIM_K, DIM_N, REQ_MULT, REQ_SHIFT, CYCLES = range(9)


def bench(dataflow='OS'):
    ins = dict(rst_n=1, bus_addr=13, bus_wdata=32, bus_we=1, bus_re=1, busy=1, done=1, cycles=32,
               wb_rdata=32, ab_rdata=32, bias_rdata=32)
    outs = dict(bus_rdata=32, dim_m=7, dim_k=7, dim_n=7, req_mult=16, req_shift=5, relu_en=1, go=1, host_wdata=32,
                wb_addr=BAW, wb_we=1, wb_re=1, ab_addr=BAW + 1, ab_we=1, ab_re=1, bias_addr=IW, bias_we=1, bias_re=1,
                irq=1)
    order = ['rst_n', 'bus_addr', 'bus_wdata', 'bus_we', 'bus_re', 'bus_rdata', 'dim_m', 'dim_k', 'dim_n', 'req_mult',
             'req_shift', 'relu_en', 'go', 'busy', 'done', 'cycles', 'host_wdata', 'wb_addr', 'wb_we', 'wb_re',
             'wb_rdata', 'ab_addr', 'ab_we', 'ab_re', 'ab_rdata', 'bias_addr', 'bias_we', 'bias_re', 'bias_rdata', 'irq']
    b = Bench(inputs=ins, outputs=outs,
              build=lambda s, w: HostInterface(s, 'host', *[w[n] for n in order], dataflow=dataflow))
    b.step(rst_n=0)
    b.set(rst_n=1)
    return b


def write(b, addr, val):
    b.step(bus_addr=addr, bus_wdata=val, bus_we=1, bus_re=0)
    b.set(bus_we=0)


def read(b, addr):
    b.step(bus_addr=addr, bus_we=0, bus_re=1)
    b.set(bus_re=0)
    return b.get('bus_rdata')


# --- Tests ---------------------------------------------------------------------------------------------------

def test_id_and_reset_values():
    assert read(bench('OS'), ID) == 4 | (0 << 4) | (64 << 8) | (1 << 24)
    b = bench('WS')
    assert read(b, ID) == 4 | (1 << 4) | (64 << 8) | (1 << 24)
    assert [read(b, r) for r in (CTRL, STATUS, DIM_M, DIM_K, DIM_N, REQ_MULT, REQ_SHIFT)] == [0, 0, 0, 0, 0, 1, 0]
    assert (b.get('req_mult'), b.get('req_shift'), b.get('relu_en')) == (1, 0, 0), 'neutral requantization'


def test_configuration_registers():
    b = bench()
    for r, v, expected, output in [(DIM_M, 33, 33, 'dim_m'), (DIM_K, 0xFFFF, 0x7F, 'dim_k'), (DIM_N, 64, 64, 'dim_n'),
                                  (REQ_MULT, 0x12345, 0x2345, 'req_mult'), (REQ_SHIFT, 0x3F, 0x1F, 'req_shift')]:
        write(b, r, v)
        assert read(b, r) == expected, r
        assert b.get(output) == expected, output
    write(b, CTRL, 0b10)
    assert read(b, CTRL) == 0b10 and b.get('relu_en') == 1
    write(b, ID, 0)
    assert read(b, ID) >> 24 == 1, 'ID is read-only'


def test_address_decoding():
    b = bench()
    for base, active in [(WEIGHT + 5, 'wb'), (ACT_IN + 7, 'ab'), (ACT_OUT + 9, 'ab'), (BIAS + 3, 'bias'),
                        (REG + 4, None), (0x1400, None), (0x1C00, None)]:
        for op in ('we', 're'):
            snap = b.observe(bus_addr=base, bus_wdata=0xABCD, bus_we=int(op == 'we'), bus_re=int(op == 're'))
            for m in ('wb', 'ab', 'bias'):
                assert snap[f'{m}_{op}'] == int(m == active), (hex(base), op, m)
            assert snap['host_wdata'] == 0xABCD
    snap = b.observe(bus_addr=ACT_OUT + 9, bus_we=0, bus_re=0)
    assert snap['ab_addr'] == (1 << BAW) | 9, 'bit 10 selects the OUT bank'
    snap = b.observe(bus_addr=ACT_IN + 9)
    assert snap['ab_addr'] == 9
    assert b.observe(bus_addr=WEIGHT + 1023)['wb_addr'] == 1023
    assert b.observe(bus_addr=BIAS + 63)['bias_addr'] == 63


def test_go_pulse():
    b = bench()
    b.step(bus_addr=CTRL, bus_wdata=0b11, bus_we=1)
    assert b.get('go') == 1, 'go in the cycle following the write'
    b.step(bus_we=0)
    assert b.get('go') == 0, 'single-cycle pulse'
    assert read(b, CTRL) == 0b10, 'GO reads back as 0, RELU_EN stays'


def test_done_irq():
    b = bench()
    b.step(done=1)
    b.set(done=0)
    assert b.get('irq') == 1 and read(b, STATUS) == 0b10
    b.step()
    assert b.get('irq') == 1, 'DONE is latched'
    write(b, CTRL, 1)
    assert b.get('irq') == 0 and read(b, STATUS) == 0b01, 'GO clears DONE (and BUSY from the go pulse on)'


def test_go_has_priority_over_done_in_the_same_cycle():
    b = bench()
    b.step(bus_addr=CTRL, bus_wdata=1, bus_we=1, done=1)
    assert b.get('irq') == 0 and b.get('go') == 1


def test_busy_blocks_configuration_and_go():
    b = bench()
    write(b, DIM_M, 10)
    b.set(busy=1)
    write(b, DIM_M, 20)
    b.step(bus_addr=CTRL, bus_wdata=0b11, bus_we=1)
    assert b.get('go') == 0, 'GO ignored during busy'
    b.set(bus_we=0)
    assert read(b, DIM_M) == 10, 'write ignored during busy'
    assert read(b, CTRL) == 0
    assert read(b, STATUS) == 0b01, 'BUSY visible, reads possible during busy'
    b.set(busy=0)
    write(b, DIM_M, 20)
    assert read(b, DIM_M) == 20


def test_busy_from_the_go_pulse():
    """
    The controller only sets busy to 1 one cycle after the go pulse: during that cycle, the host_interface already
    considers itself busy, otherwise a host access right after GO would go through before the computation starts.
    """
    b = bench()
    write(b, DIM_M, 5)
    b.step(bus_addr=CTRL, bus_wdata=1, bus_we=1)              # GO; busy of the controller still at 0
    assert b.get('go') == 1
    # during the cycle of the pulse (in the next cycle, the real controller will have busy = 1)
    snap = b.observe(bus_addr=WEIGHT + 2, bus_wdata=7, bus_we=1, bus_re=1)
    assert (snap['wb_we'], snap['wb_re']) == (0, 0), 'memory accesses blocked during the go pulse'
    b.set(bus_we=0, bus_re=0)
    assert read(b, DIM_M) == 5


def test_read_latency_1_and_multiplexer():
    b = bench()
    for addr, src in [(WEIGHT + 3, 'wb_rdata'), (ACT_OUT + 2, 'ab_rdata'), (BIAS + 1, 'bias_rdata')]:
        b.step(bus_addr=addr, bus_re=1)
        # the memory returns its data in the next cycle: it is presented on its port, with noise on the others
        b.set(bus_re=0, bus_addr=REG + STATUS, wb_rdata=111, ab_rdata=222, bias_rdata=333)
        b.set(**{src: 0xC0FFEE})
        b.sim.propagateAll()
        assert b.get('bus_rdata') == 0xC0FFEE, src
    b.set(cycles=4242)
    assert read(b, CYCLES) == 4242
    assert read(b, 0x1400) == 0 and read(b, 0x1FFF) == 0, 'unassigned address'
    # the data remains available as long as no new read is requested
    write(b, DIM_N, 9)
    assert read(b, DIM_N) == 9
    b.step(bus_re=0, bus_addr=WEIGHT)
    assert b.get('bus_rdata') == 9


if __name__ == '__main__':
    sys.exit(run_tests(globals()))
