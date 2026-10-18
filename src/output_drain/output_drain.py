"""
output_drain — Collecte des résultats de la grille (spécification : src/output_drain/interface.md)

Sortie commune aux deux variantes : N voies int32 (`res`) + `res_valid` + `tag_out`.

  - OutputDrain_OS : lit une ligne d'accumulateurs de la grille par cycle (vidage d'une tuile en N cycles)
  - OutputDrain_WS : redresse la sortie en escalier de la grille, retarde le sideband du controller
                     de 2N cycles, et additionne les sommes partielles des tuiles successives de K
                     dans une mémoire d'accumulateurs (une entrée par colonne j)

Description comportementale : toutes les sorties sont des registres, mis à jour dans clock().
"""

from py4hw import Logic, Wire


class OutputDrain_OS(Logic):
    """res ← ligne drain_row des accumulateurs ; res_valid ← drain_en ; tag_out ← tag_in (latence 1)."""

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
        self.row_w = res.getWidth()                       # N·AW bits : une ligne de la grille
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
                # la ligne r occupe les bits [r·N·AW, (r+1)·N·AW) du bus acc (élément (i, j) à l'indice i·N + j)
                r = self.drain_row.get()
                self.res_r = (self.acc.get() >> (r * self.row_w)) & ((1 << self.row_w) - 1)
                self.tag_r = self.tag_in.get()

        self.res.prepare(self.res_r)
        self.res_valid.prepare(self.valid_r)
        self.tag_out.prepare(self.tag_r)


class OutputDrain_WS(Logic):
    """
    Redressement (la voie i traverse N−1−i registres) + ligne à retard de 2N cycles pour le sideband
    + mémoire d'accumulateurs indexée par s_j :
        somme = s_first ? psum : accmem[s_j] + psum
        s_last ? (res ← somme, res_valid ← 1, tag_out ← tag) : accmem[s_j] ← somme
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

        # mémoire d'accumulateurs : son contenu n'est pas réinitialisé par rst_n
        self.accmem = [[0] * N for _ in range(self.depth)]
        self._reset_state()

    def _reset_state(self):
        N = self.N
        self.deskew = [[0] * (N - 1 - i) for i in range(N)]      # voie i : N−1−i registres
        self.sideband = [None] * (2 * N)                          # ligne à retard de 2N cycles
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
            # valeurs visibles pendant ce cycle (avant le front)
            aligned = [self.deskew[i][-1] if N - 1 - i else south[i] for i in range(N)]
            sb = self.sideband[-1]

            self.valid_r = 0
            if sb is not None and sb['valid']:
                j = sb['j']
                somme = aligned if sb['first'] else [(a + p) & self.mask
                                                     for a, p in zip(self.accmem[j], aligned)]
                if sb['last']:
                    self.res_r, self.valid_r, self.tag_r = somme, 1, sb['tag']
                else:
                    self.accmem[j] = somme

            # avance des registres de redressement et de la ligne à retard
            for i in range(N - 1):
                self.deskew[i] = [south[i]] + self.deskew[i][:-1]
            self.sideband = [dict(valid=self.s_valid.get(), j=self.s_j.get(), first=self.s_first.get(),
                                  last=self.s_last.get(), tag=self.tag_in.get())] + self.sideband[:-1]

        self.res.prepare(self._pack(self.res_r))
        self.res_valid.prepare(self.valid_r)
        self.tag_out.prepare(self.tag_r)
