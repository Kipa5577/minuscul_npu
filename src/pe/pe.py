"""
pe — Processing Element (specification: src/pe/interface.md)

Two variants, one per dataflow:
  - PE_OS: output-stationary, the accumulator stays in the PE
  - PE_WS: weight-stationary, the weight stays in the PE

Behavioural description: all outputs are registers, updated in clock().
"""

from py4hw import Logic, Wire
from py4hw.helper import IntegerHelper


def _signed(wire: Wire) -> int:
    return IntegerHelper.c2_to_signed(wire.get(), wire.getWidth())


class PE_OS(Logic):
    """Output-stationary PE: acc ← clear ? 0 : acc + w_in·x_in ; w_in and x_in are passed on to the neighbour."""

    def __init__(self, parent, name: str,
                 rst_n: Wire, w_in: Wire, x_in: Wire, clear: Wire,
                 w_out: Wire, x_out: Wire, acc: Wire):
        super().__init__(parent, name)

        self.rst_n = self.addIn('rst_n', rst_n)
        self.w_in = self.addIn('w_in', w_in)
        self.x_in = self.addIn('x_in', x_in)
        self.clear = self.addIn('clear', clear)
        self.w_out = self.addOut('w_out', w_out)
        self.x_out = self.addOut('x_out', x_out)
        self.acc = self.addOut('acc', acc)

        DW = w_in.getWidth()
        assert x_in.getWidth() == DW and w_out.getWidth() == DW and x_out.getWidth() == DW
        assert acc.getWidth() >= 2 * DW

        self.w_r = 0
        self.x_r = 0
        self.acc_r = 0

    def clock(self):
        if not self.rst_n.get():
            self.w_r, self.x_r, self.acc_r = 0, 0, 0
        else:
            if self.clear.get():
                self.acc_r = 0                    # clear has priority: the product of this cycle is lost
            else:
                self.acc_r += _signed(self.w_in) * _signed(self.x_in)
            self.w_r = self.w_in.get()
            self.x_r = self.x_in.get()

        self.w_out.prepare(self.w_r)
        self.x_out.prepare(self.x_r)
        self.acc.prepare(self.acc_r)              # prepare() reduces modulo 2^AW
        self.acc_r &= (1 << self.acc.getWidth()) - 1


class PE_WS(Logic):
    """Weight-stationary PE: psum_out ← psum_in + w_r·x_in ; if load: w_r ← w_load_in."""

    def __init__(self, parent, name: str,
                 rst_n: Wire, x_in: Wire, psum_in: Wire, w_load_in: Wire, load: Wire,
                 x_out: Wire, psum_out: Wire, w_load_out: Wire):
        super().__init__(parent, name)

        self.rst_n = self.addIn('rst_n', rst_n)
        self.x_in = self.addIn('x_in', x_in)
        self.psum_in = self.addIn('psum_in', psum_in)
        self.w_load_in = self.addIn('w_load_in', w_load_in)
        self.load = self.addIn('load', load)
        self.x_out = self.addOut('x_out', x_out)
        self.psum_out = self.addOut('psum_out', psum_out)
        self.w_load_out = self.addOut('w_load_out', w_load_out)

        DW = x_in.getWidth()
        AW = psum_in.getWidth()
        assert w_load_in.getWidth() == DW and x_out.getWidth() == DW and w_load_out.getWidth() == DW
        assert psum_out.getWidth() == AW and AW >= 2 * DW

        self.w_r = 0
        self.x_r = 0
        self.psum_r = 0

    def clock(self):
        if not self.rst_n.get():
            self.w_r, self.x_r, self.psum_r = 0, 0, 0
        else:
            w = IntegerHelper.c2_to_signed(self.w_r, self.w_load_out.getWidth())
            self.psum_r = self.psum_in.get() + w * _signed(self.x_in)    # old weight
            if self.load.get():
                self.w_r = self.w_load_in.get()
            self.x_r = self.x_in.get()

        self.x_out.prepare(self.x_r)
        self.psum_out.prepare(self.psum_r)        # prepare() reduces modulo 2^AW
        self.w_load_out.prepare(self.w_r)
