"""
Tools shared by the testbenches (test.py, benchmark.py, animation.py).

Usage from test/<block>/<file>.py:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from common import ...
"""

import sys
import pathlib
import inspect
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))          # allows `from src.<block>.<block> import ...`

import py4hw


# --- Two's complement --------------------------------------------------------

def to_c2(v: int, w: int) -> int:
    """Signed integer → w-bit pattern."""
    return v & ((1 << w) - 1)


def from_c2(v: int, w: int) -> int:
    """w-bit pattern → signed integer."""
    v &= (1 << w) - 1
    return v - (1 << w) if v >> (w - 1) else v


def wrap(v: int, w: int) -> int:
    """Reduces a signed integer modulo 2^w (behaviour of a w-bit adder)."""
    return from_c2(to_c2(v, w), w)


def pack(values, w: int) -> int:
    """List of signed values → packed bus (lane 0 = least significant bits)."""
    bus = 0
    for l, v in enumerate(values):
        bus |= to_c2(int(v), w) << (w * l)
    return bus


def unpack(bus: int, n: int, w: int) -> list:
    """Packed bus → list of n signed values."""
    return [from_c2(bus >> (w * l), w) for l in range(n)]


def load_test_module(block: str, filename: str = 'test.py'):
    """
    Loads test/<block>/<filename> as a module (to reuse its testbenches and reference models).
    A plain `import test` would pick up the `test` module of the standard library.
    """
    import importlib.util
    path = ROOT / 'test' / block / filename
    spec = importlib.util.spec_from_file_location(f'{block}_{path.stem}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def setup_matplotlib_fonts():
    """
    Matplotlib rereads the font file for every text it draws. Depending on the file system where it
    is installed, this access can become the limiting factor of the animations. When an identical copy
    of a font shipped with matplotlib exists among the system fonts, that copy is used instead.
    No effect if no system copy is found.
    """
    import os
    import dataclasses
    from matplotlib import font_manager
    bundled = os.path.join(os.path.dirname(font_manager.__file__), 'mpl-data')
    system = {}
    for f in font_manager.findSystemFonts():
        system.setdefault(os.path.basename(f), f)
    fm = font_manager.fontManager
    fm.ttflist = [dataclasses.replace(e, fname=system[os.path.basename(e.fname)])
                  if e.fname.startswith(bundled) and os.path.basename(e.fname) in system else e
                  for e in fm.ttflist]
    fm._findfont_cached.cache_clear()
    font_manager._get_font.cache_clear()


# --- Generic testbench ---------------------------------------------------------

class Bench:
    """
    Minimal py4hw system around a block.

    - `inputs` : {name: width} of the wires driven by the bench
    - `outputs`: {name: width} of the wires read by the bench
    - `build(sys, wires)` instantiates the block under test

    A call to `step(**values)` presents the inputs during one cycle, then
    advances by one clock edge; the outputs read afterwards are those after the edge.
    """

    def __init__(self, inputs: dict, outputs: dict, build, signed=()):
        self.sys = py4hw.HWSystem()
        self.widths = {**inputs, **outputs}
        self.signed = set(signed)
        self.w = {n: self.sys.wire(n, wd) for n, wd in self.widths.items()}
        self.inputs = set(inputs)
        self.dut = build(self.sys, self.w)
        self.sim = self.sys.getSimulator()
        self.cycle = 0

    def set(self, **values):
        for n, v in values.items():
            assert n in self.inputs, f'{n} is not an input'
            self.w[n].put(to_c2(v, self.widths[n]))

    def get(self, name: str) -> int:
        v = self.w[name].get()
        return from_c2(v, self.widths[name]) if name in self.signed else v

    def step(self, **values):
        self.set(**values)
        self.sim.clk(1)
        self.cycle += 1

    def observe(self, **values) -> dict:
        """
        Presents the inputs, propagates the combinational logic, reads all the outputs DURING the cycle,
        then advances by one edge. Useful for blocks some of whose outputs are combinational.
        """
        self.set(**values)
        self.sim.propagateAll()
        snap = self.snapshot()
        self.sim.clk(1)
        self.cycle += 1
        return snap

    def snapshot(self) -> dict:
        return {n: self.get(n) for n in self.widths}


# --- Mini test runner ----------------------------------------------------------

def run_tests(module_globals: dict) -> int:
    """Runs all the test_* functions of the module, prints the summary, returns the exit code."""
    tests = [(n, f) for n, f in module_globals.items()
             if n.startswith('test_') and inspect.isfunction(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f'  OK    {name}')
        except Exception:
            failed += 1
            print(f'  FAIL  {name}')
            traceback.print_exc(limit=3)
    print(f'\n{len(tests) - failed}/{len(tests)} tests passed')
    return 1 if failed else 0
