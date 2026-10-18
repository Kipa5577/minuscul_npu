"""
data_feeder — Alignment, masking and skewing of the streams (specification: src/data_feeder/interface.md)

  1. Align: the masks and ctl emitted by the controller are delayed by one cycle,
            so that they arrive at the same time as the data from the SRAM (read latency = 1).
  2. Mask : out-of-bounds lanes are forced to 0.
  3. Skew : lane l of a skewed stream goes through l registers (systolic staircase).

Two variants:
  - DataFeeder_OS: W and X are skewed
  - DataFeeder_WS: W is not skewed (broadcast preload), X is skewed

Behavioural description:
  - clock()    : alignment registers and skew chains
  - propagate(): masking and outputs (lane 0 of a skewed stream has no register)
"""

from py4hw import Logic, Wire


class _DataFeeder(Logic):

    SKEW_W = True
    SKEW_X = True

    def __init__(self, parent, name: str, rst_n: Wire,
                 w_rdata: Wire, x_rdata: Wire, w_mask: Wire, x_mask: Wire, ctl_in: Wire,
                 w_arr: Wire, x_arr: Wire, ctl_arr: Wire):
        super().__init__(parent, name)

        self.rst_n = self.addIn('rst_n', rst_n)
        self.w_rdata = self.addIn('w_rdata', w_rdata)
        self.x_rdata = self.addIn('x_rdata', x_rdata)
        self.w_mask = self.addIn('w_mask', w_mask)
        self.x_mask = self.addIn('x_mask', x_mask)
        self.ctl_in = self.addIn('ctl_in', ctl_in)
        self.w_arr = self.addOut('w_arr', w_arr)
        self.x_arr = self.addOut('x_arr', x_arr)
        self.ctl_arr = self.addOut('ctl_arr', ctl_arr)

        self.N = w_mask.getWidth()
        self.DW = w_rdata.getWidth() // self.N
        assert x_mask.getWidth() == self.N
        assert x_rdata.getWidth() == w_rdata.getWidth() == w_arr.getWidth() == x_arr.getWidth()
        assert w_rdata.getWidth() == self.N * self.DW

        self._reset_state()

    def _reset_state(self):
        N = self.N
        self.w_mask_r = 0
        self.x_mask_r = 0
        self.ctl_r = 0
        # chain of lane l: l registers, element [-1] is the output of the lane
        self.w_chain = [[0] * l for l in range(N)]
        self.x_chain = [[0] * l for l in range(N)]

    # --- helpers --------------------------------------------------------------------

    def _lanes(self, bus: int) -> list:
        m = (1 << self.DW) - 1
        return [(bus >> (self.DW * l)) & m for l in range(self.N)]

    def _pack(self, lanes: list) -> int:
        bus = 0
        for l, v in enumerate(lanes):
            bus |= v << (self.DW * l)
        return bus

    def _masked(self, rdata: Wire, mask_r: int) -> list:
        return [v if (mask_r >> l) & 1 else 0 for l, v in enumerate(self._lanes(rdata.get()))]

    @staticmethod
    def _out(masked: list, chain: list, skew: bool) -> list:
        if not skew:
            return masked
        return [masked[0]] + [chain[l][-1] for l in range(1, len(masked))]

    @staticmethod
    def _shift(masked: list, chain: list):
        for l in range(1, len(masked)):
            chain[l] = [masked[l]] + chain[l][:-1]

    # --- behaviour ----------------------------------------------------------------------

    def clock(self):
        if not self.rst_n.get():
            self._reset_state()
            return
        # the chains advance with the masked values present during this cycle
        if self.SKEW_W:
            self._shift(self._masked(self.w_rdata, self.w_mask_r), self.w_chain)
        if self.SKEW_X:
            self._shift(self._masked(self.x_rdata, self.x_mask_r), self.x_chain)
        # alignment stage
        self.w_mask_r = self.w_mask.get()
        self.x_mask_r = self.x_mask.get()
        self.ctl_r = self.ctl_in.get()

    def propagate(self):
        w = self._out(self._masked(self.w_rdata, self.w_mask_r), self.w_chain, self.SKEW_W)
        x = self._out(self._masked(self.x_rdata, self.x_mask_r), self.x_chain, self.SKEW_X)
        self.w_arr.put(self._pack(w))
        self.x_arr.put(self._pack(x))
        self.ctl_arr.put(self.ctl_r)


class DataFeeder_OS(_DataFeeder):
    """W → w_west and X → x_north, both skewed; ctl → clear."""
    SKEW_W = True
    SKEW_X = True


class DataFeeder_WS(_DataFeeder):
    """W → w_north without skew (preload); X → x_west skewed; ctl → load."""
    SKEW_W = False
    SKEW_X = True
