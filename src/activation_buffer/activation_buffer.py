"""
activation_buffer — SRAM des activations (spécification : src/activation_buffer/interface.md)

Deux bancs simple port indépendants de 2^BAW mots de N·DW bits (1024 × 32 bits chacun en v1) :
  - IN  : X, lu par le data_feeder pendant un GEMM (core_rd_*)
  - OUT : Y, écrit par le post_processing pendant un GEMM (core_wr_*, un bit d'écriture par octet)
Comme les bancs sont séparés, le cœur peut lire X et écrire Y dans le même cycle.

Quand busy = 0, l'hôte accède aux deux bancs (host_*) : le bit BAW de host_addr choisit le banc (0 = IN, 1 = OUT).
host_rdata vient du banc choisi lors de la lecture (le choix est mémorisé avec la donnée lue).
Lecture synchrone (latence 1) ; le contenu n'est pas réinitialisé (pas de rst_n).

Description comportementale : clock() uniquement.
"""

from py4hw import Logic, Wire


class ActivationBuffer(Logic):

    def __init__(self, parent, name: str, busy: Wire,
                 host_addr: Wire, host_wdata: Wire, host_we: Wire, host_re: Wire, host_rdata: Wire,
                 core_rd_addr: Wire, core_rd_re: Wire, core_rd_rdata: Wire,
                 core_wr_addr: Wire, core_wr_wdata: Wire, core_wr_we: Wire, core_wr_wmask: Wire):
        super().__init__(parent, name)

        self.busy = self.addIn('busy', busy)
        self.host_addr = self.addIn('host_addr', host_addr)
        self.host_wdata = self.addIn('host_wdata', host_wdata)
        self.host_we = self.addIn('host_we', host_we)
        self.host_re = self.addIn('host_re', host_re)
        self.host_rdata = self.addOut('host_rdata', host_rdata)
        self.core_rd_addr = self.addIn('core_rd_addr', core_rd_addr)
        self.core_rd_re = self.addIn('core_rd_re', core_rd_re)
        self.core_rd_rdata = self.addOut('core_rd_rdata', core_rd_rdata)
        self.core_wr_addr = self.addIn('core_wr_addr', core_wr_addr)
        self.core_wr_wdata = self.addIn('core_wr_wdata', core_wr_wdata)
        self.core_wr_we = self.addIn('core_wr_we', core_wr_we)
        self.core_wr_wmask = self.addIn('core_wr_wmask', core_wr_wmask)

        self.BAW = core_rd_addr.getWidth()
        assert host_addr.getWidth() == self.BAW + 1 and core_wr_addr.getWidth() == self.BAW
        self.N = core_wr_wmask.getWidth()
        self.DW = core_wr_wdata.getWidth() // self.N
        self.depth = 1 << self.BAW
        self.bank_in = [0] * self.depth
        self.bank_out = [0] * self.depth
        self.in_rdata_r = 0             # registre de lecture du banc IN (partagé hôte / cœur)
        self.out_rdata_r = 0            # registre de lecture du banc OUT (hôte)
        self.sel_r = 0                  # banc lu par l'hôte au dernier accès

    def _masked_write(self, old: int, new: int, mask: int) -> int:
        byte = (1 << self.DW) - 1
        for l in range(self.N):
            if (mask >> l) & 1:
                sh = self.DW * l
                old = (old & ~(byte << sh)) | (new & (byte << sh))
        return old

    def clock(self):
        if self.busy.get():
            if self.core_rd_re.get():
                self.in_rdata_r = self.bank_in[self.core_rd_addr.get()]
            if self.core_wr_we.get():
                a = self.core_wr_addr.get()
                self.bank_out[a] = self._masked_write(self.bank_out[a], self.core_wr_wdata.get(),
                                                      self.core_wr_wmask.get())
        else:
            haddr = self.host_addr.get()
            bank_sel = haddr >> self.BAW
            a = haddr & (self.depth - 1)
            bank = self.bank_out if bank_sel else self.bank_in
            if self.host_re.get():
                if bank_sel:
                    self.out_rdata_r = bank[a]
                else:
                    self.in_rdata_r = bank[a]
                self.sel_r = bank_sel
            if self.host_we.get():
                bank[a] = self.host_wdata.get()

        self.core_rd_rdata.prepare(self.in_rdata_r)
        self.host_rdata.prepare(self.out_rdata_r if self.sel_r else self.in_rdata_r)
