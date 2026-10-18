# data_feeder — Alignment, masking and skewing of the streams

This block sits between the buffers and the array. It has three roles:
1. **Align**: it delays the commands emitted by the `controller` by one cycle, so that they arrive at the same time as
   the data from the SRAM (read latency = 1).
2. **Mask**: it forces out-of-bounds lanes to 0 (D9, dimensions that are not multiples of 4).
3. **Skew**: it delays lane l by l cycles (systolic staircase).

The ports are the same for both variants. Only the internal routing changes.

## Parameters
`N` = 4, `DW` = 8, `DATAFLOW` ∈ {OS, WS}

## Ports
| Signal | Dir | Width | Description |
|---|---|---|---|
| `clk`, `rst_n` | in | 1 | |
| `w_rdata` | in | N·DW | word read from `weight_buffer` (`core_rdata`) |
| `x_rdata` | in | N·DW | word read from `activation_buffer`, IN area (`core_rd_rdata`) |
| `w_mask` | in | N | valid W lanes, at the emission cycle |
| `x_mask` | in | N | valid X lanes, at the emission cycle |
| `ctl_in` | in | 1 | OS: `clear` · WS: `load`, at the emission cycle |
| `w_arr` | out | N·DW | OS → `w_west` · WS → `w_north` |
| `x_arr` | out | N·DW | OS → `x_north` · WS → `x_west` |
| `ctl_arr` | out | 1 | OS → `clear` · WS → `load` |

## Behaviour
```
alignment stage (registers) : w_mask_r ← w_mask ; x_mask_r ← x_mask ; ctl_arr ← ctl_in
masking (combinational)     : w_m[l] = w_mask_r[l] ? w_rdata[l] : 0      (same for x)
skewing                     : lane l → chain of l registers of DW bits (reset to 0 by rst_n)
```

| Stream | OS | WS |
|---|---|---|
| W | masked then **skewed** | masked, **not skewed** (the preload is broadcast) |
| X | masked then **skewed** | masked then **skewed** |

- When nothing is read, the `controller` sets the masks to 0, so the block injects zeros.

## Latencies (from the emission cycle)
| Output | L |
|---|---|
| lane l of a skewed stream | 1 + l |
| `w_arr` in WS | 1 |
| `ctl_arr` | 1 |

## Cost (N = 4)
- Skewing: N(N−1)/2 = 6 8-bit registers per skewed stream, i.e. 96 flip-flops in OS and 48 in WS.
- Alignment: 2N + 1 = 9 flip-flops.

---

## Implementation

- File: [`data_feeder.py`](data_feeder.py) — classes `DataFeeder_OS` (W and X skewed) and `DataFeeder_WS` (W not
  skewed), on a common base.
- `clock()`: alignment registers (`w_mask_r`, `x_mask_r`, `ctl_r`) and skew chains; `propagate()`: masking and outputs.
  Lane 0 has no skew register: it is **combinational** from `rdata`, hence the latency 1 + l measured from emission
  (the SRAM provides the first cycle).
- `N` is deduced from the width of `w_mask`, `DW` from that of `w_rdata`.

| Constructor | Ports, in order |
|---|---|
| `DataFeeder_OS` / `DataFeeder_WS(parent, name, …)` | `rst_n, w_rdata, x_rdata, w_mask, x_mask, ctl_in, w_arr, x_arr, ctl_arr` |

**Verification** ([`test/data_feeder/`](../../test/data_feeder/)): latency of each lane for N = 1, 2, 4, 8, masking,
zeros injected when idle, reference model over 1000 random cycles, and a `data_feeder` → `pe_array` chain on GEMMs
whose dimensions are not multiples of N, with non-zero out-of-bounds values in memory.

**Measured**: data reuse is 2 times better in WS (3.8 MACs per byte read versus 2.0 in OS for N = 4) and the required
SRAM bandwidth is 2 times lower; edge masking wastes on average 8 % of the lanes for N = 4, 15 % for N = 8.
