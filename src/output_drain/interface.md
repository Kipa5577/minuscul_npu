# output_drain — Collecting the results from the array

This block collects the int32 results from the array and presents them to `post_processing` in a form common to both
variants: **N int32 lanes + `res_valid` + tag**.
The tag (Y address, write mask, row) is defined in [architecture.md §5](../architecture.md#5-interface-conventions).

## Parameters
`N` = 4, `AW` = 32, `IW` = 6, `TAG_W` = 20, `DATAFLOW`

N is an explicit parameter of both variants: it cannot be deduced from the width of `drain_row` when N is not a power
of 2.

## Common output
| Signal | Dir | Width | Description |
|---|---|---|---|
| `res` | out | N·AW | N int32 results |
| `res_valid` | out | 1 | `res` is valid |
| `tag_out` | out | TAG_W | metadata accompanying `res` |

---

## OS variant: draining the accumulators

### Input ports
| Signal | Dir | Width | Description |
|---|---|---|---|
| `clk`, `rst_n` | in | 1 | |
| `acc` | in | N·N·AW | from `pe_array` |
| `drain_en` | in | 1 | from the `controller` |
| `drain_row` | in | log2 N | row r of the tile to read |
| `tag_in` | in | TAG_W | from the `controller` |

### Behaviour
```
res       ← acc[drain_row][0..N−1]      (lane j = PE(drain_row, j))
res_valid ← drain_en
tag_out   ← tag_in
```
- Latency: L = 1. Draining a tile takes N cycles (one row per cycle).
- Cost: an N-to-1 multiplexer on N·AW bits, plus N·AW + 1 + TAG_W = 149 flip-flops.

---

## WS variant: deskewing and accumulation over the K tiles

### Input ports
| Signal | Dir | Width | Description |
|---|---|---|---|
| `clk`, `rst_n` | in | 1 | |
| `psum_south` | in | N·AW | from `pe_array`, lane i is i cycles late |
| `s_valid` | in | 1 | sideband from the `controller`, at the emission cycle of X |
| `s_j` | in | IW | column j, used as the address in the accumulator memory |
| `s_first` | in | 1 | first K tile (k_t = 0): overwrite instead of adding |
| `s_last` | in | 1 | last K tile: emit the result |
| `tag_in` | in | TAG_W | |

### Internal structure
1. **Deskewing**: lane i goes through N−1−i registers. The N lanes of column j are then aligned at cycle `e + 2N`,
   where e is the emission cycle of X for this column.
2. **Sideband delay line**: `s_valid`, `s_j`, `s_first`, `s_last` and `tag_in` are delayed by **2N cycles** so that
   they arrive at the same time as the deskewed data.
3. **Accumulator memory**: 2^IW = 64 entries × N·AW = 128 bits (1 KiB). Combinational read, synchronous write.

### Behaviour (at the aligned cycle, if the delayed s_valid is set)
```
sum = s_first ? psum_aligned : accmem[s_j] + psum_aligned      (lane-by-lane addition)
if !s_last : accmem[s_j] ← sum
if  s_last : res ← sum ; res_valid ← 1 ; tag_out ← delayed tag
```
- Latency: `res_valid` arrives at cycle `e + 2N + 1`.
- Cost (N = 4): 6 × 32 bits of deskewing, 8 × 29 bits of delay line, 1 KiB of memory and N 32-bit adders.
- Possible optimisation: rebuild the tag locally (it depends only on i_t and j) instead of delaying it.

---

## Implementation

- File: [`output_drain.py`](output_drain.py) — classes `OutputDrain_OS` and `OutputDrain_WS`.
- All outputs are registers (`clock()` only). In WS, the accumulator memory (`accmem`) is not reset by `rst_n`; the
  deskew registers, the delay line and `res` are.

| Constructor | Ports, in order |
|---|---|
| `OutputDrain_OS(parent, name, …, N=4)` | `rst_n, acc, drain_en, drain_row, tag_in, res, res_valid, tag_out` |
| `OutputDrain_WS(parent, name, …, N=4)` | `rst_n, psum_south, s_valid, s_j, s_first, s_last, tag_in, res, res_valid, tag_out` |

**Verification** ([`test/output_drain/`](../../test/output_drain/)): row-by-row draining, deskewing and exact latency
of 2N + 1 for N = 1, 2, 4, 8, accumulation over three K tiles, single-tile case, cycle-by-cycle reference models.
**End to end**: the complete compute core (`data_feeder` → `pe_array` → `output_drain`), driven by a Python scheduler
that reproduces the FSM of the `controller`, computes GEMMs up to 10×7×9; Y is rebuilt only from the (tag, `res`) pairs
that come out, and each element must be written exactly once.

**Measured**: on a 16×16×16 GEMM, the results come out in bursts of N rows in OS (448 cycles) and as a continuous
stream during the last K tile in WS (403 cycles).
