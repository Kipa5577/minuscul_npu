# activation_buffer — Activation SRAM

This block contains two independent **single-port** banks of 1024 × 32 bits each:
- **IN**: X, read by the core;
- **OUT**: Y, written by `post_processing`.

Since the banks are separate, the core can read X and write Y in the same cycle.
The data layout is described in D6, and depends on the variant.

## Parameters
`N` = 4, `DW` = 8, `BAW` = 10

## Ports
| Signal | Dir | Width | Description |
|---|---|---|---|
| `clk` | in | 1 | (the contents are not reset by `rst_n`) |
| `busy` | in | 1 | 0: host port active on both banks · 1: core ports active |
| `host_addr` | in | BAW+1 | bit BAW: 0 = IN, 1 = OUT; bits [BAW−1:0]: word address |
| `host_wdata` | in | N·DW | |
| `host_we` | in | 1 | write of a full word |
| `host_re` | in | 1 | |
| `host_rdata` | out | N·DW | latency 1 |
| `core_rd_addr` | in | BAW | IN bank, from the `controller` |
| `core_rd_re` | in | 1 | |
| `core_rd_rdata` | out | N·DW | latency 1, to the `data_feeder` |
| `core_wr_addr` | in | BAW | OUT bank, from `post_processing` (`y_addr`) |
| `core_wr_wdata` | in | N·DW | `y_wdata` |
| `core_wr_we` | in | 1 | `y_we` |
| `core_wr_wmask` | in | N | `y_wmask`: one write-enable bit per byte (lane) |

## Behaviour
```
IN bank    : busy ? core read : host access (if host_addr[BAW] = 0)
OUT bank   : busy ? core write (bytes enabled by wmask) : host access (if host_addr[BAW] = 1)
host_rdata : word of the bank selected by host_addr[BAW] (the selection is remembered for one cycle)
```

## Latencies
Read: L = 1. Write: takes effect at the next edge.

---

## Implementation

- File: [`activation_buffer.py`](activation_buffer.py) — class `ActivationBuffer`, `clock()` only (no `rst_n`).
- Each bank has its own read register; `host_rdata` presents the one of the bank last read by the host (the selection
  is remembered along with the data). `core_rd_rdata` is the read register of the IN bank.
- A core write into the OUT bank only modifies the bytes enabled by `core_wr_wmask`.

| Constructor | Ports, in order |
|---|---|
| `ActivationBuffer(parent, name, …)` | `busy, host_addr, host_wdata, host_we, host_re, host_rdata, core_rd_addr, core_rd_re, core_rd_rdata, core_wr_addr, core_wr_wdata, core_wr_we, core_wr_wmask` |

**Verification** ([`test/activation_buffer/`](../../test/activation_buffer/)): host access to both banks, IN read and
OUT write in the same cycle, per-byte mask, active side selected by `busy`, reference model over 3000 accesses, X
stored according to D6 then read back at the scheduler addresses. **End to end with the memories**: the host loads W,
X and the biases word by word, the core computes, the host reads Y back from the OUT bank; and **two chained layers**
by copying Y as is from the OUT bank to the IN bank, which checks that Y does come out in the X format of the variant
(D6).

**Measured**: without DMA, the host transfers one word per cycle and the core waits: loading and reading back account
for half of the total time for a 4³ GEMM, and still 14 % for a 64³ GEMM. Between two layers, the OUT → IN copy by the
host costs 3 cycles per word of Y (578 cycles for the 64 → 48 layer of an MLP on a batch of 16).
