# host_interface — Host bus and registers

This block decodes the 32-bit parallel bus, holds the configuration and status registers, and redirects the accesses
to the memory windows (see the memory map in D7).

## Parameters
`N` = 4 (the memory map assumes a 32-bit buffer word), `DIM_MAX` = 64, `BAW` = 10, `IW` = 6, `DATAFLOW`

## Bus-side ports
| Signal | Dir | Width | Description |
|---|---|---|---|
| `clk`, `rst_n` | in | 1 | |
| `bus_addr` | in | 13 | word address |
| `bus_wdata` | in | 32 | |
| `bus_we` | in | 1 | |
| `bus_re` | in | 1 | |
| `bus_rdata` | out | 32 | **latency 1** for all addresses |

## Internal ports
| Signal | Dir | Width | To / from |
|---|---|---|---|
| `dim_m`, `dim_k`, `dim_n` | out | 7 each | `controller` |
| `req_mult` | out | 16 | `post_processing` |
| `req_shift` | out | 5 | `post_processing` |
| `relu_en` | out | 1 | `post_processing` |
| `go` | out | 1 | pulse to the `controller` |
| `busy` | in | 1 | `controller` |
| `done` | in | 1 | pulse from the `controller` |
| `cycles` | in | 32 | `controller` |
| `host_wdata` | out | 32 | shared by all memories (= `bus_wdata`) |
| `wb_addr`, `wb_we`, `wb_re` / `wb_rdata` | out / in | BAW, 1, 1 / 32 | `weight_buffer` (host port) |
| `ab_addr`, `ab_we`, `ab_re` / `ab_rdata` | out / in | BAW+1, 1, 1 / 32 | `activation_buffer` (host port) |
| `bias_addr`, `bias_we`, `bias_re` / `bias_rdata` | out / in | IW, 1, 1 / 32 | `post_processing` (bias port) |

## Address decoding
| `bus_addr[12:10]` | Target | Forwarded address |
|---|---|---|
| `000` | registers | `bus_addr[3:0]` |
| `001` | `WEIGHT` | `wb_addr = bus_addr[9:0]` |
| `010`, `011` | `ACT_IN`, `ACT_OUT` | `ab_addr = bus_addr[10:0]` (bit 10 selects the bank) |
| `100` | `BIAS` | `bias_addr = bus_addr[5:0]` |
| others | — | reads return 0, writes are ignored |

## Registers
| Addr. | Name | Fields | Reset |
|---|---|---|---|
| 0 | `ID` (RO) | [3:0] N · [4] DATAFLOW (0 = OS, 1 = WS) · [15:8] DIM_MAX · [31:24] version = 1 | constant |
| 1 | `CTRL` | [0] `GO` (writing 1 sends the pulse, reads back as 0) · [1] `RELU_EN` | 0 |
| 2 | `STATUS` (RO) | [0] `BUSY` · [1] `DONE` (set to 1 by `done`, cleared to 0 by `GO`) | 0 |
| 3–5 | `DIM_M/K/N` | [6:0] | 0 |
| 6 | `REQ_MULT` | [15:0] | 1 |
| 7 | `REQ_SHIFT` | [4:0] | 0 |
| 8 | `CYCLES` (RO) | [31:0] | 0 |

- With the reset values, the requantization computes `y = sat8(acc + b)`.
- The NPU is **busy** when the controller has `busy = 1` **or** a `go` pulse is in progress. The controller only sets
  `busy` to 1 one cycle after the pulse: without this rule, a host access right after GO would go through before the
  computation starts (defect found by the `npu_top` tests).
- When the NPU is busy: configuration writes, `GO` and **all memory accesses** (WEIGHT, ACT_IN, ACT_OUT, BIAS) are
  ignored, and `STATUS.BUSY` is 1. The registers remain readable.
- The read of `bus_rdata` goes through a multiplexer whose selection is latched at the `bus_re` cycle, which aligns it
  with the latency of 1 of the memories.
- `irq = STATUS.DONE` is exported by `npu_top`.

---

## Implementation

- File: [`host_interface.py`](host_interface.py) — class `HostInterface(…, dataflow='OS', N=4)`.
- `clock()`: configuration registers, `go` pulse (one-cycle register), latched DONE (GO has priority if both arrive in
  the same cycle), target and register value latched at the `bus_re` cycle.
- `propagate()`: address decoding to the host ports of the memories (combinational, forced to 0 when the NPU is busy),
  configuration outputs, `irq`, and `bus_rdata` multiplexer according to the latched target.
- A write of GO at cycle t produces `go` during cycle t + 1; the computation starts at cycle t + 2.

| Constructor | Ports, in order |
|---|---|
| `HostInterface(parent, name, …, dataflow, N)` | `rst_n, bus_addr, bus_wdata, bus_we, bus_re, bus_rdata, dim_m, dim_k, dim_n, req_mult, req_shift, relu_en, go, busy, done, cycles, host_wdata, wb_addr, wb_we, wb_re, wb_rdata, ab_addr, ab_we, ab_re, ab_rdata, bias_addr, bias_we, bias_re, bias_rdata, irq` |

**Verification** ([`test/host_interface/`](../../test/host_interface/)): ID and reset values, registers and width of
their fields, decoding of each window, `go` pulse, DONE and `irq`, blocking during `busy` and during the `go` pulse,
read latency of 1 for all targets.

**Measured on the complete chip**: going through the bus brings the throughput of a 64³ layer down from 13.5 to 11.6
MACs/cycle in OS (from 14.0 to 12.0 in WS); for a 4³ layer, the host takes up more than half of the time.
