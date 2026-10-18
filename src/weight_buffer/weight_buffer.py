"""
weight_buffer — Weight SRAM (specification: src/weight_buffer/interface.md)

Single-port memory of 2^BAW words of N·DW bits (1024 × 32 bits in v1), seen through two logical ports:
  - host port (host_*): active when busy = 0 (loading W, reading back)
  - core port (core_*): active when busy = 1 (read by the data_feeder during a GEMM)
Accesses on the inactive port are ignored.

Synchronous read: rdata is a register, updated at the edge if re = 1 on the active port, otherwise it keeps its value.
host_rdata and core_rdata are the same value (a single read register, like a real single-port SRAM).
The contents are not reset (no rst_n).

Behavioural description: clock() only.
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
                self.rdata_r = self.mem[addr]                 # reads the old value on a simultaneous write
            if self.host_we.get():
                self.mem[addr] = self.host_wdata.get()

        self.host_rdata.prepare(self.rdata_r)
        self.core_rdata.prepare(self.rdata_r)
