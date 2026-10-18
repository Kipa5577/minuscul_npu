# weight_buffer — Weight SRAM

This is a **single-port** memory of 1024 × 32 bits (4 KiB). It is seen through two logical ports, the host port and
the core port, and the `busy` signal selects which of the two is active.
The data layout is described in D6: the word at address `{i_t, k}` holds `W[4·i_t + l][k]`.

## Parameters
`N` = 4, `DW` = 8 (word width N·DW = 32), `BAW` = 10 (1024 words)

## Ports
| Signal | Dir | Width | Description |
|---|---|---|---|
| `clk` | in | 1 | (the contents are not reset by `rst_n`) |
| `busy` | in | 1 | 0: host port active · 1: core port active |
| `host_addr` | in | BAW | word address |
| `host_wdata` | in | N·DW | |
| `host_we` | in | 1 | write of a full word |
| `host_re` | in | 1 | |
| `host_rdata` | out | N·DW | latency 1 |
| `core_addr` | in | BAW | from the `controller` |
| `core_re` | in | 1 | |
| `core_rdata` | out | N·DW | latency 1, to the `data_feeder` |

## Behaviour
```
active port = busy ? core : host        (accesses on the inactive port are ignored)
write       : if we (active port) : mem[addr] ← wdata
read        : if re (active port) : rdata ← mem[addr]   (at the next edge, otherwise rdata keeps its value)
```
- `host_rdata` and `core_rdata` may be the same physical wire.

## Latencies
Read: L = 1. Write: takes effect at the next edge.

---

## Implementation

- File: [`weight_buffer.py`](weight_buffer.py) — class `WeightBuffer`, `clock()` only (no `rst_n`).
- A single read register, shared by the two logical ports: `host_rdata` and `core_rdata` always have the same value.
  A read and a write at the same address in the same cycle return the old value.
- The depth is deduced from the width of `host_addr`, the word width from that of `host_wdata`.

| Constructor | Ports, in order |
|---|---|
| `WeightBuffer(parent, name, …)` | `busy, host_addr, host_wdata, host_we, host_re, host_rdata, core_addr, core_re, core_rdata` |

**Verification** ([`test/weight_buffer/`](../../test/weight_buffer/)): latency 1, read data held, simultaneous read
and write, active port selected by `busy`, reference model over 3000 random accesses; W stored by the host according to
D6 (`test/layout.py`), then read back at the addresses emitted by the `controller` scheduler: each read returns exactly
the word expected by the compute core.

**Measured**: in OS, each word of W is read again once per tile of X columns (8 times the minimum for a 32³ GEMM, port
busy 73 % of the compute time); in WS, each word is read only once (port busy 10 % of the time).
