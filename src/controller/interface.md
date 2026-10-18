# controller — GEMM sequencer

A hardwired FSM that walks through the tiles and emits, **at the emission cycle**, the SRAM addresses, the masks, the
array commands and the tags. It knows all the downstream latencies (centralised control, D12).

## Parameters
`N` = 4, `DIM_MAX` = 64, `IW` = 6, `TW` = 4, `BAW` = 10, `TAG_W` = 20, `DATAFLOW`

## Ports
| Signal | Dir | Width | Description |
|---|---|---|---|
| `clk`, `rst_n` | in | 1 | |
| `go` | in | 1 | pulse sent by `host_interface` |
| `dim_m`, `dim_k`, `dim_n` | in | 7 each | dimensions, from 1 to 64 |
| `busy` | out | 1 | a GEMM is in progress; also used to select the active port of the memories |
| `done` | out | 1 | one-cycle pulse at the end of the GEMM |
| `cycles` | out | 32 | number of cycles of the current GEMM or of the last GEMM (reset to 0 by `go`) |
| `w_addr`, `w_re` | out | BAW, 1 | to `weight_buffer` (core port) |
| `x_addr`, `x_re` | out | BAW, 1 | to `activation_buffer` (read of the IN bank) |
| `w_mask`, `x_mask` | out | N, N | to `data_feeder` |
| `ctl` | out | 1 | to `data_feeder.ctl_in` (OS: clear · WS: load) |
| `tag` | out | TAG_W | OS: to `output_drain` at drain time · WS: sideband at streaming time |
| `drain_en`, `drain_row` | out | 1, log2 N | **OS only**, to `output_drain` |
| `s_valid`, `s_j`, `s_first`, `s_last` | out | 1, IW, 1, 1 | **WS only**, to `output_drain` |

## General rules
- A `go` with a dimension equal to 0 or greater than 64 raises `done` at the next cycle, without computing anything.
- Number of tiles: `T(D) = ⌈D/N⌉ = (D + N − 1) >> 2`.
- Mask of lane l for tile t of dimension D: `4·t + l < D`.

## OS FSM — loops i_t → j_t → k

| State | Duration | What is emitted |
|---|---|---|
| `IDLE` | — | nothing; waiting for `go` |
| `CLEAR` | 1 | `ctl = 1` |
| `STREAM` | K | for each k: `w_addr = {i_t, k}`, `x_addr = {j_t, k}`, `w_re = x_re = 1`, masks of i_t (W) and of j_t (X) |
| `FLUSH` | 2N − 1 | masks at 0: the array finishes its computation |
| `DRAIN` | N | for r = 0..N−1: `drain_en = 1`, `drain_row = r`, `tag = {y_addr = {j_t, i}, lane_mask = mask(j_t) if i < M else 0, row = i}` with `i = 4·i_t + r` |
| → next tile (`CLEAR`), or `TAIL` | | |
| `TAIL` | 4 | waiting for the last write of `post_processing` |
| `DONE` | 1 | `done = 1`, then back to `IDLE` |

**Timing diagram of a tile** (`CLEAR` emitted at cycle c0):
- the array is reset to zero during c0 + 1, and term k is emitted at c0 + 1 + k;
- PE(i,j) computes term k during cycle `c0 + 2 + i + j + k`;
- the last term is computed during c0 + K + 2N − 1, and the accumulators can be read from c0 + K + 2N onwards;
- draining occupies cycles c0 + K + 2N to c0 + K + 3N − 1.

**Period of a tile: `K + 3N` cycles.**

## WS FSM — loops i_t → k_t → j

| State | Duration | What is emitted |
|---|---|---|
| `IDLE` | — | waiting for `go` |
| `PRELOAD` | N | for n = 0..N−1: `w_addr = {i_t, 4·k_t + N−1−n}`, `w_re = 1`, `ctl = 1`, `w_mask = mask(i_t)` if `4·k_t + N−1−n < K`, else 0 |
| `STREAM` | N_dim | for each j: `x_addr = {k_t, j}`, `x_re = 1`, `x_mask = mask_K(k_t)`, `s_valid = 1`, `s_j = j`, `s_first = (k_t = 0)`, `s_last = (k_t = T(K)−1)`, `tag = {y_addr = {i_t, j}, lane_mask = mask(i_t), row = 4·i_t}` |
| `FLUSH` | 2N − 3 | the X values already injected finish crossing the array before the weights are overwritten |
| → next (i_t, k_t) (`PRELOAD`), or `TAIL` | | |
| `TAIL` | 2N + 4 | waiting for the last write |
| `DONE` | 1 | `done = 1` |

**Why FLUSH lasts 2N − 3 cycles**: let s be the cycle at which the last X is emitted. PE(N−1,N−1) uses it during
s + 2N − 1. The first `load` of the next preload reaches the array at cycle p' + 1, and only modifies the weights at
the end of that cycle. It is therefore enough that p' + 1 ≥ s + 2N − 1, i.e. p' = s + 2N − 2, which leaves 2N − 3
cycles of FLUSH.
*This limit is exact, with no margin. It is **confirmed in simulation** by `test/pe_array/test.py`
(`test_ws_minimum_gap_before_reload`): with one cycle less, the last column of Y is wrong.*

**Period of an (i_t, k_t) pair: `N_dim + 3N − 3` cycles.**

## Expected performance (N = 4, 64×64×64 GEMM, ideal = 64³/16 = 16,384 cycles)
| | Formula (without TAIL) | Cycles | Utilisation |
|---|---|---|---|
| OS | `T(M)·T(N_dim)·(K + 3N)` | 16·16·76 = 19,456 | 84 % |
| WS | `T(M)·T(K)·(N_dim + 3N − 3)` | 16·16·73 = 18,688 | 88 % |

When K is small (for example 4), the utilisation in OS drops to 25 %. When N_dim is small, it is WS that collapses.
`benchmark.py` must bring out this asymmetry.

**Verified in simulation** (`test/output_drain/benchmark.py`, complete compute core): 16×16×16 GEMM in OS = 448 cycles,
exactly `T(M)·T(N)·(K + 3N)` = 16 · 28; in WS, last result at cycle 403 (400 + output latency).

## Cost
Counters i_t, j_t or k_t (TW bits), k or j (IW bits), phase counter, mask comparators (2N), `cycles` counter (32 bits).

---

## Implementation

- File: [`controller.py`](controller.py) — classes `Controller_OS` and `Controller_WS`, Moore machine: `clock()`
  updates the state, the loop counters (`t_out` = i_t, `t_mid` = j_t or k_t) and the position counter within the
  state (`cnt`), then publishes all the outputs decoded from the new state. The block has no `propagate()`: this
  spares the simulator a false combinational loop with the `host_interface`.
- The dimensions are latched at `go`. `N` is deduced from the width of `w_mask`, `DIM_MAX` from that of `dim_m`.
- Zero-duration states are skipped (FLUSH in WS when N = 1).

**Timeline** (verified by the tests):
- a `go` pulse present at cycle g moves the FSM into its first state at the next edge: the first emission takes place
  at cycle g + 1;
- `busy` is 1 in all states except IDLE and DONE; during DONE (one cycle, `done` = 1), the last write of Y has already
  been done and the host can read the memory back;
- `cycles` counts the cycles outside IDLE since the last `go`, DONE included (plan + 1), and keeps its value until the
  next `go`;
- a dimension equal to zero or greater than `DIM_MAX` leads directly to DONE; a `go` received during a GEMM is ignored.

| Constructor | Ports, in order |
|---|---|
| `Controller_OS(parent, name, …)` | `rst_n, go, dim_m, dim_k, dim_n, busy, done, cycles, w_addr, w_re, x_addr, x_re, w_mask, x_mask, ctl, tag, drain_en, drain_row` |
| `Controller_WS(parent, name, …)` | `rst_n, go, dim_m, dim_k, dim_n, busy, done, cycles, w_addr, w_re, x_addr, x_re, w_mask, x_mask, ctl, tag, s_valid, s_j, s_first, s_last` |

**Verification** ([`test/controller/`](../../test/controller/)): for 7 GEMM shapes (from 1×1×1 to 64×2×5) and N = 2
or 4, the controller emits, **exactly cycle by cycle**, what the reference scheduler emits (`plan_os` / `plan_ws` from
[`test/output_drain/test.py`](../../test/output_drain/test.py)), itself validated end to end; cycle counter, invalid
dimensions, spurious `go`, reset midway, successive GEMMs. **End to end**: the controller drives the whole datapath
with its memories (the whole NPU except `host_interface`), started by `go` and waited for on `done`, including two
chained layers.

**Measured**: the measured cycles follow the formulas above exactly (zero difference from 1×1×1 to 64×64×64).
Depending on the shape of the GEMM, OS or WS can be up to 3.4 times faster (decision map in
`test/controller/benchmark.py`).
