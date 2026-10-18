"""
output_drain — Collecting the results from the array (specification: src/output_drain/interface.md)

Output common to both variants: N int32 lanes (`res`) + `res_valid` + `tag_out`.

  - OutputDrain_OS: reads one row of accumulators from the array per cycle (draining a tile in N cycles)
  - OutputDrain_WS: deskews the staircase output of the array, delays the controller sideband
                    by 2N cycles, and adds up the partial sums of the successive K tiles
                    in an accumulator memory (one entry per column j)

Behavioural description: all outputs are registers, updated in clock().
"""

from py4hw import Logic, Wire


class OutputDrain_OS(Logic):
    """res ← row drain_row of the accumulators; res_valid ← drain_en; tag_out ← tag_in (latency 1)."""

    def __init__(self, parent, name: str, rst_n: Wire, acc: Wire, drain_en: Wire, drain_row: Wire,
                 tag_in: Wire, res: Wire, res_valid: Wire, tag_out: Wire, N: int = 4):
        super().__init__(parent, name)

        self.rst_n = self.addIn('rst_n', rst_n)
        self.acc = self.addIn('acc', acc)
        self.drain_en = self.addIn('drain_en', drain_en)
        self.drain_row = self.addIn('drain_row', drain_row)
        self.tag_in = self.addIn('tag_in', tag_in)
        self.res = self.addOut('res', res)
        self.res_valid = self.addOut('res_valid', res_valid)
        self.tag_out = self.addOut('tag_out', tag_out)

        self.N = N
        assert (1 << drain_row.getWidth()) >= N
        self.row_w = res.getWidth()                       # N·AW bits: one row of the array
        assert acc.getWidth() == self.N * self.row_w
        assert tag_out.getWidth() == tag_in.getWidth()

        self.res_r = 0
        self.valid_r = 0
        self.tag_r = 0

    def clock(self):
        if not self.rst_n.get():
            self.res_r, self.valid_r, self.tag_r = 0, 0, 0
        else:
            self.valid_r = self.drain_en.get()
            if self.valid_r:
                # row r occupies bits [r·N·AW, (r+1)·N·AW) of the acc bus (element (i, j) at index i·N + j)
                r = self.drain_row.get()
                self.res_r = (self.acc.get() >> (r * self.row_w)) & ((1 << self.row_w) - 1)
                self.tag_r = self.tag_in.get()

        self.res.prepare(self.res_r)
        self.res_valid.prepare(self.valid_r)
        self.tag_out.prepare(self.tag_r)


class OutputDrain_WS(Logic):
    """
    Deskewing (lane i goes through N−1−i registers) + 2N-cycle delay line for the sideband
    + accumulator memory indexed by s_j:
        total = s_first ? psum : accmem[s_j] + psum
        s_last ? (res ← total, res_valid ← 1, tag_out ← tag) : accmem[s_j] ← total
    """

    def __init__(self, parent, name: str, rst_n: Wire, psum_south: Wire,
                 s_valid: Wire, s_j: Wire, s_first: Wire, s_last: Wire, tag_in: Wire,
                 res: Wire, res_valid: Wire, tag_out: Wire, N: int = 4):
        super().__init__(parent, name)

        self.rst_n = self.addIn('rst_n', rst_n)
        self.psum_south = self.addIn('psum_south', psum_south)
        self.s_valid = self.addIn('s_valid', s_valid)
        self.s_j = self.addIn('s_j', s_j)
        self.s_first = self.addIn('s_first', s_first)
        self.s_last = self.addIn('s_last', s_last)
        self.tag_in = self.addIn('tag_in', tag_in)
        self.res = self.addOut('res', res)
        self.res_valid = self.addOut('res_valid', res_valid)
        self.tag_out = self.addOut('tag_out', tag_out)

        self.N = N
        self.AW = psum_south.getWidth() // N
        assert psum_south.getWidth() == N * self.AW and res.getWidth() == N * self.AW
        self.mask = (1 << self.AW) - 1
        self.depth = 1 << s_j.getWidth()

        # accumulator memory: its contents are not reset by rst_n
        self.accmem = [[0] * N for _ in range(self.depth)]
        self._reset_state()

    def _reset_state(self):
        N = self.N
        self.deskew = [[0] * (N - 1 - i) for i in range(N)]      # lane i: N−1−i registers
        self.sideband = [None] * (2 * N)                          # 2N-cycle delay line
        self.res_r = [0] * N
        self.valid_r = 0
        self.tag_r = 0

    def _lanes(self, bus: int) -> list:
        return [(bus >> (self.AW * l)) & self.mask for l in range(self.N)]

    def _pack(self, lanes: list) -> int:
        bus = 0
        for l, v in enumerate(lanes):
            bus |= (v & self.mask) << (self.AW * l)
        return bus

    def clock(self):
        if not self.rst_n.get():
            self._reset_state()
        else:
            N = self.N
            south = self._lanes(self.psum_south.get())
            # values visible during this cycle (before the edge)
            aligned = [self.deskew[i][-1] if N - 1 - i else south[i] for i in range(N)]
            sb = self.sideband[-1]

            self.valid_r = 0
            if sb is not None and sb['valid']:
                j = sb['j']
                total = aligned if sb['first'] else [(a + p) & self.mask
                                                     for a, p in zip(self.accmem[j], aligned)]
                if sb['last']:
                    self.res_r, self.valid_r, self.tag_r = total, 1, sb['tag']
                else:
                    self.accmem[j] = total

            # advance the deskew registers and the delay line
            for i in range(N - 1):
                self.deskew[i] = [south[i]] + self.deskew[i][:-1]
            self.sideband = [dict(valid=self.s_valid.get(), j=self.s_j.get(), first=self.s_first.get(),
                                  last=self.s_last.get(), tag=self.tag_in.get())] + self.sideband[:-1]

        self.res.prepare(self._pack(self.res_r))
        self.res_valid.prepare(self.valid_r)
        self.tag_out.prepare(self.tag_r)
