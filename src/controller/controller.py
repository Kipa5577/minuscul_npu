"""
controller — Séquenceur du GEMM (spécification : src/controller/interface.md)

FSM câblée qui parcourt les tuiles et émet, au cycle d'émission, les adresses SRAM, les masques,
les commandes de la grille et les tags.

  - Controller_OS : boucles i_t → j_t → k   ; états CLEAR, STREAM (K), FLUSH (2N − 1), DRAIN (N), TAIL (4)
  - Controller_WS : boucles i_t → k_t → j   ; états PRELOAD (N), STREAM (N_dim), FLUSH (2N − 3), TAIL (2N + 4)

Chronologie : une impulsion `go` présente au cycle g fait passer la FSM dans son premier état au front suivant ;
la première émission a lieu au cycle g + 1. `busy` vaut 1 dans tous les états sauf IDLE et DONE ; pendant DONE
(un cycle, `done` = 1), la dernière écriture de Y est terminée. `cycles` compte les cycles hors IDLE depuis le
dernier `go` (DONE compris). Un `go` avec une dimension nulle ou supérieure à DIM_MAX va directement à DONE.
Un `go` reçu pendant un GEMM est ignoré.

Description comportementale (machine de Moore) : clock() met à jour les registres d'état, les compteurs de boucles
et le compteur de phase, puis publie les sorties décodées à partir du nouvel état. Les sorties ne dépendant que de
l'état, elles sont stables pendant tout le cycle ; le bloc n'a pas de propagate(), ce qui évite au simulateur une
fausse boucle combinatoire avec le host_interface (go → controller, busy → host_interface).
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
        self.cnt = 0                                       # position dans l'état courant
        self.t_out = 0                                     # i_t
        self.t_mid = 0                                     # j_t (OS) ou k_t (WS)
        self.M = self.K = self.Nd = 0                      # dimensions mémorisées au go
        self.cycles_r = 0

    # --- outils -------------------------------------------------------------------------

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
        """Entre dans `state` ; les états de durée nulle sont sautés."""
        self.state, self.cnt = state, 0
        if state not in (IDLE, DONE) and self.duration(state) == 0:
            self._goto(self._after(state))

    def _after(self, state):
        raise NotImplementedError

    # --- comportement -----------------------------------------------------------------------

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
    """Boucles i_t → j_t → k : CLEAR, STREAM, FLUSH, DRAIN par tuile de sortie, puis TAIL et DONE."""

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
        if state == DRAIN:                                 # tuile suivante : j_t, puis i_t
            self.t_mid += 1
            if self.t_mid == self.tiles(self.Nd):
                self.t_mid = 0
                self.t_out += 1
            return CLEAR if self.t_out < self.tiles(self.M) else TAIL
        return DONE                                        # après TAIL

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
    """Boucles i_t → k_t → j : PRELOAD, STREAM, FLUSH par couple (i_t, k_t), puis TAIL et DONE."""

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
        if state == FLUSH:                                 # couple suivant : k_t, puis i_t
            self.t_mid += 1
            if self.t_mid == self.tiles(self.K):
                self.t_mid = 0
                self.t_out += 1
            return PRELOAD if self.t_out < self.tiles(self.M) else TAIL
        return DONE                                        # après TAIL

    def _decode(self):
        it, kt, c = self.t_out, self.t_mid, self.cnt
        sb = dict(s_valid=0, s_j=0, s_first=0, s_last=0)
        out = {}
        if self.state == PRELOAD:
            k = self.N * kt + self.N - 1 - c               # la ligne de grille du bas d'abord
            out.update(w_addr=self.addr(it, k), w_re=1, ctl=1,
                       w_mask=self.lane_mask(it, self.M) if k < self.K else 0)
        elif self.state == STREAM:
            out.update(x_addr=self.addr(kt, c), x_re=1, x_mask=self.lane_mask(kt, self.K),
                       tag=self.tag_pack(self.addr(it, c), self.lane_mask(it, self.M), self.N * it))
            sb.update(s_valid=1, s_j=c, s_first=int(kt == 0), s_last=int(kt == self.tiles(self.K) - 1))
        for name, v in sb.items():
            getattr(self, name).prepare(v)
        return out
