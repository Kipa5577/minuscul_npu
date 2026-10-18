"""
host_interface — Host bus and registers (specification: src/host_interface/interface.md)

Decodes the 32-bit parallel bus (13-bit word address):
  bus_addr[12:10] = 000 → registers (bus_addr[3:0])
                    001 → WEIGHT : weight_buffer      (wb_addr  = bus_addr[9:0])
                    01x → ACT_IN / ACT_OUT : activation_buffer (ab_addr = bus_addr[10:0], bit 10 selects the bank)
                    100 → BIAS   : bias memory of the post_processing (bias_addr = bus_addr[5:0])
                    others → reads return 0, writes are ignored

Registers: ID (RO), CTRL (GO, RELU_EN), STATUS (RO: BUSY, DONE), DIM_M/K/N, REQ_MULT, REQ_SHIFT, CYCLES (RO).
  - writing CTRL with GO = 1 sends a one-cycle `go` pulse to the controller (in the cycle after the write);
  - DONE is set to 1 by the `done` pulse of the controller and cleared to 0 by GO; `irq` = DONE;
  - "busy" = busy of the controller OR go pulse in progress: the controller only sets busy to 1 one cycle after
    the pulse, and without this a host access right after GO would go through before the computation starts;
  - when the NPU is busy: configuration writes, GO and memory accesses are ignored, STATUS.BUSY is 1.

Read: latency 1 for all addresses. The target being read is latched at the bus_re cycle; in the next cycle,
bus_rdata presents the register value (captured at the cycle of the request) or the data returned by the memory.

Behavioural description:
  - clock()    : configuration registers, go pulse, DONE, latched read target and data
  - propagate(): address decoding to the memories (combinational) and bus_rdata multiplexer
"""

from py4hw import Logic, Wire

REG, WEIGHT, ACT, BIAS, NONE = 'REG', 'WEIGHT', 'ACT', 'BIAS', 'NONE'
ID, CTRL, STATUS, DIM_M, DIM_K, DIM_N, REQ_MULT, REQ_SHIFT, CYCLES = range(9)
VERSION = 1


class HostInterface(Logic):

    def __init__(self, parent, name: str, rst_n: Wire,
                 bus_addr: Wire, bus_wdata: Wire, bus_we: Wire, bus_re: Wire, bus_rdata: Wire,
                 dim_m: Wire, dim_k: Wire, dim_n: Wire, req_mult: Wire, req_shift: Wire, relu_en: Wire,
                 go: Wire, busy: Wire, done: Wire, cycles: Wire,
                 host_wdata: Wire,
                 wb_addr: Wire, wb_we: Wire, wb_re: Wire, wb_rdata: Wire,
                 ab_addr: Wire, ab_we: Wire, ab_re: Wire, ab_rdata: Wire,
                 bias_addr: Wire, bias_we: Wire, bias_re: Wire, bias_rdata: Wire,
                 irq: Wire, dataflow: str = 'OS', N: int = 4):
        super().__init__(parent, name)

        self.rst_n = self.addIn('rst_n', rst_n)
        self.bus_addr = self.addIn('bus_addr', bus_addr)
        self.bus_wdata = self.addIn('bus_wdata', bus_wdata)
        self.bus_we = self.addIn('bus_we', bus_we)
        self.bus_re = self.addIn('bus_re', bus_re)
        self.bus_rdata = self.addOut('bus_rdata', bus_rdata)
        self.dim_m = self.addOut('dim_m', dim_m)
        self.dim_k = self.addOut('dim_k', dim_k)
        self.dim_n = self.addOut('dim_n', dim_n)
        self.req_mult = self.addOut('req_mult', req_mult)
        self.req_shift = self.addOut('req_shift', req_shift)
        self.relu_en = self.addOut('relu_en', relu_en)
        self.go = self.addOut('go', go)
        self.busy = self.addIn('busy', busy)
        self.done = self.addIn('done', done)
        self.cycles = self.addIn('cycles', cycles)
        self.host_wdata = self.addOut('host_wdata', host_wdata)
        self.wb_addr = self.addOut('wb_addr', wb_addr)
        self.wb_we = self.addOut('wb_we', wb_we)
        self.wb_re = self.addOut('wb_re', wb_re)
        self.wb_rdata = self.addIn('wb_rdata', wb_rdata)
        self.ab_addr = self.addOut('ab_addr', ab_addr)
        self.ab_we = self.addOut('ab_we', ab_we)
        self.ab_re = self.addOut('ab_re', ab_re)
        self.ab_rdata = self.addIn('ab_rdata', ab_rdata)
        self.bias_addr = self.addOut('bias_addr', bias_addr)
        self.bias_we = self.addOut('bias_we', bias_we)
        self.bias_re = self.addOut('bias_re', bias_re)
        self.bias_rdata = self.addIn('bias_rdata', bias_rdata)
        self.irq = self.addOut('irq', irq)

        assert bus_addr.getWidth() == 13 and bus_wdata.getWidth() == 32
        self.BAW = wb_addr.getWidth()
        self.IW = bias_addr.getWidth()
        self.DIM_MAX = 1 << (dim_m.getWidth() - 1)
        self.id_value = N | ((1 if dataflow == 'WS' else 0) << 4) | (self.DIM_MAX << 8) | (VERSION << 24)
        self._reset_state()

    def _reset_state(self):
        self.regs = {DIM_M: 0, DIM_K: 0, DIM_N: 0, REQ_MULT: 1, REQ_SHIFT: 0}
        self.relu_r = 0
        self.go_r = 0
        self.done_r = 0
        self.sel_r = NONE                  # target of the last read
        self.reg_rdata_r = 0               # register value captured at the cycle of the request

    # --- decoding -------------------------------------------------------------------------

    def _target(self, addr: int) -> str:
        top = addr >> 10
        return {0: REG, 1: WEIGHT, 2: ACT, 3: ACT, 4: BIAS}.get(top, NONE)

    def _is_busy(self) -> int:
        return self.busy.get() | self.go_r

    def _read_reg(self, r: int) -> int:
        if r == ID:
            return self.id_value
        if r == CTRL:
            return self.relu_r << 1                                    # GO reads back as 0
        if r == STATUS:
            return self._is_busy() | (self.done_r << 1)
        if r == CYCLES:
            return self.cycles.get()
        return self.regs.get(r, 0)

    # --- behaviour -------------------------------------------------------------------------

    def clock(self):
        if not self.rst_n.get():
            self._reset_state()
            return
        addr, we, re = self.bus_addr.get(), self.bus_we.get(), self.bus_re.get()
        target = self._target(addr)
        busy = self._is_busy()

        # read: target and register value latched for the next cycle
        if re:
            self.sel_r = target
            if target == REG:
                self.reg_rdata_r = self._read_reg(addr & 0xF)

        # go pulse and DONE (GO has priority: it starts a new GEMM)
        self.go_r = 0
        if self.done.get():
            self.done_r = 1
        if we and target == REG and not busy:
            r, v = addr & 0xF, self.bus_wdata.get()
            if r == CTRL:
                self.relu_r = (v >> 1) & 1
                if v & 1:
                    self.go_r, self.done_r = 1, 0
            elif r in (DIM_M, DIM_K, DIM_N):
                self.regs[r] = v & 0x7F
            elif r == REQ_MULT:
                self.regs[r] = v & 0xFFFF
            elif r == REQ_SHIFT:
                self.regs[r] = v & 0x1F

    def propagate(self):
        addr, we, re = self.bus_addr.get(), self.bus_we.get(), self.bus_re.get()
        target = self._target(addr)
        if self._is_busy():                                  # no memory access during a GEMM
            we = re = 0

        # to the memories (combinational)
        self.host_wdata.put(self.bus_wdata.get())
        self.wb_addr.put(addr & ((1 << self.BAW) - 1))
        self.wb_we.put(int(we and target == WEIGHT))
        self.wb_re.put(int(re and target == WEIGHT))
        self.ab_addr.put(addr & ((1 << (self.BAW + 1)) - 1))
        self.ab_we.put(int(we and target == ACT))
        self.ab_re.put(int(re and target == ACT))
        self.bias_addr.put(addr & ((1 << self.IW) - 1))
        self.bias_we.put(int(we and target == BIAS))
        self.bias_re.put(int(re and target == BIAS))

        # configuration
        self.dim_m.put(self.regs[DIM_M])
        self.dim_k.put(self.regs[DIM_K])
        self.dim_n.put(self.regs[DIM_N])
        self.req_mult.put(self.regs[REQ_MULT])
        self.req_shift.put(self.regs[REQ_SHIFT])
        self.relu_en.put(self.relu_r)
        self.go.put(self.go_r)
        self.irq.put(self.done_r)

        # data read: captured register, or data returned by the memory targeted in the previous cycle
        self.bus_rdata.put({REG: self.reg_rdata_r, WEIGHT: self.wb_rdata.get(), ACT: self.ab_rdata.get(),
                            BIAS: self.bias_rdata.get(), NONE: 0}[self.sel_r])
