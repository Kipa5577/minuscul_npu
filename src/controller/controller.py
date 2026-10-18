"""
controller — GEMM sequencer (specification: src/controller/interface.md)

Hardwired FSM that walks through the tiles and emits, at the emission cycle, the SRAM addresses, the masks,
the array commands and the tags.

  - Controller_OS: loops i_t → j_t → k   ; states CLEAR, STREAM (K), FLUSH (2N − 1), DRAIN (N), TAIL (4)
  - Controller_WS: loops i_t → k_t → j   ; states PRELOAD (N), STREAM (N_dim), FLUSH (2N − 3), TAIL (2N + 4)

Timeline: a `go` pulse present at cycle g moves the FSM into its first state at the next edge;
the first emission takes place at cycle g + 1. `busy` is 1 in all states except IDLE and DONE; during DONE
(one cycle, `done` = 1), the last write of Y is finished. `cycles` counts the cycles outside IDLE since the
last `go` (DONE included). A `go` with a dimension equal to zero or greater than DIM_MAX goes directly to DONE.
A `go` received during a GEMM is ignored.

Behavioural description (Moore machine): clock() updates the state registers, the loop counters and the phase
counter, then publishes the outputs decoded from the new state. Since the outputs depend only on the state, they
are stable during the whole cycle; the block has no propagate(), which spares the simulator a false
combinational loop with the host_interface (go → controller, busy → host_interface).
"""

from py4hw import Logic, Wire

IDLE, CLEAR, PRELOAD, STREAM, FLUSH, DRAIN, TAIL, DONE = 'IDLE', 'CLEAR', 'PRELOAD', 'STREAM', 'FLUSH', 'DRAIN', 'TAIL', 'DONE'


class _Controller(Logic):

    def __init__(self, parent, name: str, rst_n: Wire, go: Wire, dim_m: Wire, dim_k: Wire, dim_n: Wire,
                 busy: Wire, done: Wire, cycles: Wire, w_addr: Wire, w_re: Wire, x_addr: Wire, x_re: Wire,
                 w_mask: Wire, x_mask: Wire, ctl: Wire, tag: Wire):
        super().__init__(parent, name)

        self.rst_n = self.addIn('rst_n', rst_n)
        self.go = self.addIn('go', go)
        self.dim_m = self.addIn('dim_m', dim_m)
        self.dim_k = self.addIn('dim_k', dim_k)
        self.dim_n = self.addIn('dim_n', dim_n)
        self.busy = self.addOut('busy', busy)
        self.done = self.addOut('done', done)
        self.cycles = self.addOut('cycles', cycles)
        self.w_addr = self.addOut('w_addr', w_addr)
        self.w_re = self.addOut('w_re', w_re)
        self.x_addr = self.addOut('x_addr', x_addr)
        self.x_re = self.addOut('x_re', x_re)
        self.w_mask = self.addOut('w_mask', w_mask)
        self.x_mask = self.addOut('x_mask', x_mask)
        self.ctl = self.addOut('ctl', ctl)
        self.tag = self.addOut('tag', tag)

        self.N = w_mask.getWidth()
        self.IW = dim_m.getWidth() - 1                    # log2(DIM_MAX)
        self.DIM_MAX = 1 << self.IW
        assert w_addr.getWidth() == x_addr.getWidth()
        assert tag.getWidth() == w_addr.getWidth() + self.N + self.IW

        self._reset_state()

    def _reset_state(self):
        self.state = IDLE
        self.cnt = 0                                       # position within the current state
        self.t_out = 0                                     # i_t
        self.t_mid = 0                                     # j_t (OS) or k_t (WS)
        self.M = self.K = self.Nd = 0                      # dimensions latched at go
        self.cycles_r = 0

    # --- tools --------------------------------------------------------------------------

    def tiles(self, D):
        return (D + self.N - 1) // self.N

    def lane_mask(self, t, D):
        return sum(1 << l for l in range(self.N) if self.N * t + l < D)

    def addr(self, t, u):
        return (t << self.IW) | u

    def tag_pack(self, y_addr, mask, row):
        return (y_addr << (self.N + self.IW)) | (mask << self.IW) | row

    def duration(self, state):
        raise NotImplementedError

    def _goto(self, state):
        """Enters `state`; zero-duration states are skipped."""
        self.state, self.cnt = state, 0
        if state not in (IDLE, DONE) and self.duration(state) == 0:
            self._goto(self._after(state))

    def _after(self, state):
        raise NotImplementedError

    # --- behaviour --------------------------------------------------------------------------

    def clock(self):
        if not self.rst_n.get():
            self._reset_state()
        else:
            self._step()
        self._emit()

    def _step(self):
        if self.state != IDLE:
            self.cycles_r += 1
        if self.state == IDLE:
            if self.go.get():
                self.M, self.K, self.Nd = self.dim_m.get(), self.dim_k.get(), self.dim_n.get()
                self.t_out = self.t_mid = 0
                self.cycles_r = 0
                valid = all(1 <= d <= self.DIM_MAX for d in (self.M, self.K, self.Nd))
                self._goto(self._first() if valid else DONE)
        elif self.state == DONE:
            self._goto(IDLE)
        elif self.cnt + 1 < self.duration(self.state):
            self.cnt += 1
        else:
            self._goto(self._after(self.state))

    def _emit(self):
        out = dict(w_addr=0, w_re=0, x_addr=0, x_re=0, w_mask=0, x_mask=0, ctl=0, tag=0)
        out.update(self._decode())
        self.busy.prepare(int(self.state not in (IDLE, DONE)))
        self.done.prepare(int(self.state == DONE))
        self.cycles.prepare(self.cycles_r)
        for name, v in out.items():
            getattr(self, name).prepare(v)


class Controller_OS(_Controller):
    """Loops i_t → j_t → k: CLEAR, STREAM, FLUSH, DRAIN per output tile, then TAIL and DONE."""

    def __init__(self, parent, name, rst_n, go, dim_m, dim_k, dim_n, busy, done, cycles,
                 w_addr, w_re, x_addr, x_re, w_mask, x_mask, ctl, tag, drain_en, drain_row):
        super().__init__(parent, name, rst_n, go, dim_m, dim_k, dim_n, busy, done, cycles,
                         w_addr, w_re, x_addr, x_re, w_mask, x_mask, ctl, tag)
        self.drain_en = self.addOut('drain_en', drain_en)
        self.drain_row = self.addOut('drain_row', drain_row)

    def _first(self):
        return CLEAR

    def duration(self, state):
        return {CLEAR: 1, STREAM: self.K, FLUSH: 2 * self.N - 1, DRAIN: self.N, TAIL: 4}[state]

    def _after(self, state):
        if state == CLEAR:
            return STREAM
        if state == STREAM:
            return FLUSH
        if state == FLUSH:
            return DRAIN
        if state == DRAIN:                                 # next tile: j_t, then i_t
            self.t_mid += 1
            if self.t_mid == self.tiles(self.Nd):
                self.t_mid = 0
                self.t_out += 1
            return CLEAR if self.t_out < self.tiles(self.M) else TAIL
        return DONE                                        # after TAIL

    def _decode(self):
        it, jt, c = self.t_out, self.t_mid, self.cnt
        out = dict(drain_en=0, drain_row=0)
        if self.state == CLEAR:
            out.update(ctl=1)
        elif self.state == STREAM:
            out.update(w_addr=self.addr(it, c), x_addr=self.addr(jt, c), w_re=1, x_re=1,
                       w_mask=self.lane_mask(it, self.M), x_mask=self.lane_mask(jt, self.Nd))
        elif self.state == DRAIN:
            i = self.N * it + c
            out.update(drain_en=1, drain_row=c,
                       tag=self.tag_pack(self.addr(jt, i), self.lane_mask(jt, self.Nd) if i < self.M else 0, i))
        self.drain_en.prepare(out.pop('drain_en'))
        self.drain_row.prepare(out.pop('drain_row'))
        return out


class Controller_WS(_Controller):
    """Loops i_t → k_t → j: PRELOAD, STREAM, FLUSH per (i_t, k_t) pair, then TAIL and DONE."""

    def __init__(self, parent, name, rst_n, go, dim_m, dim_k, dim_n, busy, done, cycles,
                 w_addr, w_re, x_addr, x_re, w_mask, x_mask, ctl, tag, s_valid, s_j, s_first, s_last):
        super().__init__(parent, name, rst_n, go, dim_m, dim_k, dim_n, busy, done, cycles,
                         w_addr, w_re, x_addr, x_re, w_mask, x_mask, ctl, tag)
        self.s_valid = self.addOut('s_valid', s_valid)
        self.s_j = self.addOut('s_j', s_j)
        self.s_first = self.addOut('s_first', s_first)
        self.s_last = self.addOut('s_last', s_last)

    def _first(self):
        return PRELOAD

    def duration(self, state):
        return {PRELOAD: self.N, STREAM: self.Nd, FLUSH: max(0, 2 * self.N - 3), TAIL: 2 * self.N + 4}[state]

    def _after(self, state):
        if state == PRELOAD:
            return STREAM
        if state == STREAM:
            return FLUSH
        if state == FLUSH:                                 # next pair: k_t, then i_t
            self.t_mid += 1
            if self.t_mid == self.tiles(self.K):
                self.t_mid = 0
                self.t_out += 1
            return PRELOAD if self.t_out < self.tiles(self.M) else TAIL
        return DONE                                        # after TAIL

    def _decode(self):
        it, kt, c = self.t_out, self.t_mid, self.cnt
        sb = dict(s_valid=0, s_j=0, s_first=0, s_last=0)
        out = {}
        if self.state == PRELOAD:
            k = self.N * kt + self.N - 1 - c               # the bottom row of the array first
            out.update(w_addr=self.addr(it, k), w_re=1, ctl=1,
                       w_mask=self.lane_mask(it, self.M) if k < self.K else 0)
        elif self.state == STREAM:
            out.update(x_addr=self.addr(kt, c), x_re=1, x_mask=self.lane_mask(kt, self.K),
                       tag=self.tag_pack(self.addr(it, c), self.lane_mask(it, self.M), self.N * it))
            sb.update(s_valid=1, s_j=c, s_first=int(kt == 0), s_last=int(kt == self.tiles(self.K) - 1))
        for name, v in sb.items():
            getattr(self, name).prepare(v)
        return out
