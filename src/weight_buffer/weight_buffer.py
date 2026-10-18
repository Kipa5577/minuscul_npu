"""
weight_buffer — SRAM des poids (spécification : src/weight_buffer/interface.md)

Mémoire simple port de 2^BAW mots de N·DW bits (1024 × 32 bits en v1), vue par deux ports logiques :
  - port hôte (host_*) : actif quand busy = 0 (chargement de W, relecture)
  - port cœur (core_*) : actif quand busy = 1 (lecture par le data_feeder pendant un GEMM)
Les accès du port inactif sont ignorés.

Lecture synchrone : rdata est un registre, mis à jour au front si re = 1 sur le port actif, sinon il garde sa valeur.
host_rdata et core_rdata sont la même valeur (un seul registre de lecture, comme une vraie SRAM simple port).
Le contenu n'est pas réinitialisé (pas de rst_n).

Description comportementale : clock() uniquement.
"""

from py4hw import Logic, Wire


class WeightBuffer(Logic):

    def __init__(self, parent, name: str, busy: Wire,
                 host_addr: Wire, host_wdata: Wire, host_we: Wire, host_re: Wire, host_rdata: Wire,
                 core_addr: Wire, core_re: Wire, core_rdata: Wire):
        super().__init__(parent, name)

        self.busy = self.addIn('busy', busy)
        self.host_addr = self.addIn('host_addr', host_addr)
        self.host_wdata = self.addIn('host_wdata', host_wdata)
        self.host_we = self.addIn('host_we', host_we)
        self.host_re = self.addIn('host_re', host_re)
        self.host_rdata = self.addOut('host_rdata', host_rdata)
        self.core_addr = self.addIn('core_addr', core_addr)
        self.core_re = self.addIn('core_re', core_re)
        self.core_rdata = self.addOut('core_rdata', core_rdata)

        assert core_addr.getWidth() == host_addr.getWidth()
        assert host_wdata.getWidth() == host_rdata.getWidth() == core_rdata.getWidth()
        self.depth = 1 << host_addr.getWidth()
        self.mem = [0] * self.depth
        self.rdata_r = 0

    def clock(self):
        if self.busy.get():
            if self.core_re.get():
                self.rdata_r = self.mem[self.core_addr.get()]
        else:
            addr = self.host_addr.get()
            if self.host_re.get():
                self.rdata_r = self.mem[addr]                 # lecture de l'ancienne valeur si écriture simultanée
            if self.host_we.get():
                self.mem[addr] = self.host_wdata.get()

        self.host_rdata.prepare(self.rdata_r)
        self.core_rdata.prepare(self.rdata_r)
