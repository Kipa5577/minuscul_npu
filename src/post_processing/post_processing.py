"""
post_processing — Bias, ReLU, requantization (specification: src/post_processing/interface.md)

3-stage pipeline, N lanes in parallel, one input per cycle:
  S1: a = res + bias                           (int32, modulo 2^32)
  S2: r = relu_en ? max(0, a) : a ; p = r × M  (M unsigned, signed 48-bit product)
  S3: q = (p + 2^(s−1)) >> s ; y = sat(q, −128, 127)   (no rounding if s = 0)
The valid bit and the tag move along with the data; after S3, y_we, y_addr, y_wmask and y_wdata
drive the write into the OUT bank of activation_buffer.

The block also contains the bias memory (64 × int32), accessible by the host when busy = 0.
Choice of the bias for lane l (row comes from the tag):
  - PostProcessing_OS: the N lanes are one and the same row of Y → bias[row] for all of them
  - PostProcessing_WS: the N lanes are N consecutive rows        → bias[row + l]

Behavioural description: all outputs are registers, updated in clock().
"""

from py4hw import Logic, Wire


def _signed(v: int, w: int) -> int:
    v &= (1 << w) - 1
    return v - (1 << w) if v >> (w - 1) else v


class _PostProcessing(Logic):

    BIAS_PER_LANE = False

    def __init__(self, parent, name: str, rst_n: Wire, busy: Wire,
                 res: Wire, res_valid: Wire, tag_in: Wire,
                 relu_en: Wire, req_mult: Wire, req_shift: Wire,
                 bias_host_addr: Wire, bias_host_wdata: Wire, bias_host_we: Wire, bias_host_re: Wire,
                 bias_host_rdata: Wire,
                 y_addr: Wire, y_wdata: Wire, y_we: Wire, y_wmask: Wire):
        super().__init__(parent, name)

        self.rst_n = self.addIn('rst_n', rst_n)
        self.busy = self.addIn('busy', busy)
        self.res = self.addIn('res', res)
        self.res_valid = self.addIn('res_valid', res_valid)
        self.tag_in = self.addIn('tag_in', tag_in)
        self.relu_en = self.addIn('relu_en', relu_en)
        self.req_mult = self.addIn('req_mult', req_mult)
        self.req_shift = self.addIn('req_shift', req_shift)
        self.bias_host_addr = self.addIn('bias_host_addr', bias_host_addr)
        self.bias_host_wdata = self.addIn('bias_host_wdata', bias_host_wdata)
        self.bias_host_we = self.addIn('bias_host_we', bias_host_we)
        self.bias_host_re = self.addIn('bias_host_re', bias_host_re)
        self.bias_host_rdata = self.addOut('bias_host_rdata', bias_host_rdata)
        self.y_addr = self.addOut('y_addr', y_addr)
        self.y_wdata = self.addOut('y_wdata', y_wdata)
        self.y_we = self.addOut('y_we', y_we)
        self.y_wmask = self.addOut('y_wmask', y_wmask)

        self.N = y_wmask.getWidth()
        self.AW = bias_host_wdata.getWidth()
        self.DW = y_wdata.getWidth() // self.N
        self.IW = bias_host_addr.getWidth()
        assert res.getWidth() == self.N * self.AW
        assert tag_in.getWidth() == y_addr.getWidth() + self.N + self.IW

        # bias memory: its contents are not reset by rst_n
        self.bias = [0] * (1 << self.IW)
        self._reset_state()

    def _reset_state(self):
        N = self.N
        # each stage: (valid, values of the N lanes, tag)
        self.s1 = (0, [0] * N, 0)
        self.s2 = (0, [0] * N, 0)
        self.s3 = (0, [0] * N, 0)
        self.rdata_r = 0

    # --- stages -------------------------------------------------------------------------

    def _tag_fields(self, tag: int):
        """tag = {y_addr, lane_mask, row}"""
        row = tag & ((1 << self.IW) - 1)
        mask = (tag >> self.IW) & ((1 << self.N) - 1)
        return tag >> (self.IW + self.N), mask, row

    def _stage1(self):
        valid, tag = self.res_valid.get(), self.tag_in.get()
        _, _, row = self._tag_fields(tag)
        bus = self.res.get()
        lanes = [_signed(bus >> (self.AW * l), self.AW) for l in range(self.N)]
        bias = [self.bias[(row + l if self.BIAS_PER_LANE else row) & ((1 << self.IW) - 1)] for l in range(self.N)]
        return valid, [_signed(a + b, self.AW) for a, b in zip(lanes, bias)], tag

    def _stage2(self, s1):
        valid, a, tag = s1
        relu = self.relu_en.get()
        m = self.req_mult.get()
        return valid, [(max(0, x) if relu else x) * m for x in a], tag

    def _stage3(self, s2):
        valid, p, tag = s2
        s = self.req_shift.get()
        rnd = (1 << (s - 1)) if s > 0 else 0
        lo, hi = -(1 << (self.DW - 1)), (1 << (self.DW - 1)) - 1
        return valid, [min(hi, max(lo, (x + rnd) >> s)) for x in p], tag

    # --- behaviour --------------------------------------------------------------------------

    def clock(self):
        if not self.rst_n.get():
            self._reset_state()
        else:
            # pipeline: each stage takes the output of the previous stage (values from before the edge)
            self.s3, self.s2, self.s1 = self._stage3(self.s2), self._stage2(self.s1), self._stage1()
            # host port of the bias memory (active when busy = 0)
            if not self.busy.get():
                addr = self.bias_host_addr.get()
                if self.bias_host_re.get():
                    self.rdata_r = self.bias[addr] & ((1 << self.AW) - 1)
                if self.bias_host_we.get():
                    self.bias[addr] = _signed(self.bias_host_wdata.get(), self.AW)

        valid, y, tag = self.s3
        y_addr, mask, _ = self._tag_fields(tag)
        wdata = 0
        for l, v in enumerate(y):
            wdata |= (v & ((1 << self.DW) - 1)) << (self.DW * l)
        self.y_we.prepare(valid)
        self.y_addr.prepare(y_addr)
        self.y_wmask.prepare(mask if valid else 0)
        self.y_wdata.prepare(wdata)
        self.bias_host_rdata.prepare(self.rdata_r)


class PostProcessing_OS(_PostProcessing):
    """The N lanes are one and the same row of Y: they all receive bias[row]."""
    BIAS_PER_LANE = False


class PostProcessing_WS(_PostProcessing):
    """The N lanes are N consecutive rows of Y: lane l receives bias[row + l]."""
    BIAS_PER_LANE = True
