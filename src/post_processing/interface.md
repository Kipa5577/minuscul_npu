# post_processing — Bias, ReLU, requantization

A 3-stage pipeline that processes **N lanes in parallel** and accepts a new input every cycle.
The block contains the bias memory and writes directly into the OUT bank of `activation_buffer`.

## Parameters
`N` = 4, `AW` = 32, `DW` = 8, `IW` = 6, `BAW` = 10, `TAG_W` = 20, `DATAFLOW`

## Ports
| Signal | Dir | Width | Description |
|---|---|---|---|
| `clk`, `rst_n` | in | 1 | |
| `busy` | in | 1 | 0: the host port of the biases is active |
| `res` | in | N·AW | from `output_drain` |
| `res_valid` | in | 1 | |
| `tag_in` | in | TAG_W | `{y_addr, lane_mask, row}` |
| `relu_en` | in | 1 | register `CTRL.RELU_EN` |
| `req_mult` | in | 16 | register `REQ_MULT`, unsigned |
| `req_shift` | in | 5 | register `REQ_SHIFT` |
| `bias_host_addr` | in | IW | index of the bias (0..63) |
| `bias_host_wdata` | in | AW | |
| `bias_host_we` | in | 1 | |
| `bias_host_re` | in | 1 | |
| `bias_host_rdata` | out | AW | latency 1 |
| `y_addr` | out | BAW | to `activation_buffer.core_wr_addr` |
| `y_wdata` | out | N·DW | to `core_wr_wdata` |
| `y_we` | out | 1 | to `core_wr_we` |
| `y_wmask` | out | N | to `core_wr_wmask` |

## Bias memory
- 64 × int32, stored in **N banks of 16 entries**: bias b is in bank `b mod N`, at address `b div N`.
- The read is combinational. The index comes from `row[IW−1:2]`, and the 4 banks are read at once.
- Choice of the bias for lane l:
  - **OS**: the 4 lanes belong to the same row of Y, so they all receive `bank[row mod N]`.
  - **WS**: the 4 lanes correspond to 4 consecutive rows (`row` is a multiple of 4), so lane l receives `bank[l]`.

## Pipeline (per lane l)
| Stage | Computation | Width of the result |
|---|---|---|
| S1 | `a = res_l + bias_l` | 32 bits signed (wraps modulo 2^32) |
| S2 | `r = relu_en ? max(0, a) : a` ; `p = r × req_mult` | 48 bits signed |
| S3 | `q = (p + (s > 0 ? 2^(s−1) : 0)) >>> s` ; `y_l = sat(q, −128, 127)` | 8 bits signed |

After S3, the outputs are registered: `y_we = valid`, `y_addr = tag.y_addr`, `y_wmask = tag.lane_mask`.
The `valid` bit and the tag move through the pipeline together with the data.

## Latencies
`res_valid` present at cycle t gives `y_we` at cycle **t + 3**, and the word is written at the next edge.

## Cost (N = 4)
4 32-bit adders, 4 32×16 multipliers, 4 48-bit rounding adders, 4 saturators and 2 Kibit of biases.
Trade-off to be measured: a 1-lane version (4 times fewer multipliers) would slow down the draining by a factor of 4.

---

## Implementation

- File: [`post_processing.py`](post_processing.py) — classes `PostProcessing_OS` (one bias for the N lanes) and
  `PostProcessing_WS` (one bias per lane), on a common base.
- The 3 stages are registers updated in `clock()`; the output register of S3 directly drives `y_we`, `y_addr`,
  `y_wmask` (forced to 0 when `y_we` = 0) and `y_wdata`.
- The bias memory is modelled as a list of 64 entries: reading `bias[row]` (OS) or `bias[row + l]` (WS) is exactly
  equivalent to reading the N banks described above. It is not reset by `rst_n`.
- `relu_en` and `req_mult` are read at stage S2, `req_shift` at stage S3: they must remain constant during a GEMM.
- The width of M is deduced from that of the `req_mult` wire (16 bits in v1), which lets the benchmark explore it.

| Constructor | Ports, in order |
|---|---|
| `PostProcessing_OS` / `PostProcessing_WS(parent, name, …)` | `rst_n, busy, res, res_valid, tag_in, relu_en, req_mult, req_shift, bias_host_addr, bias_host_wdata, bias_host_we, bias_host_re, bias_host_rdata, y_addr, y_wdata, y_we, y_wmask` |

**Verification** ([`test/post_processing/`](../../test/post_processing/)): exact latency of 3 cycles and throughput of
one input per cycle, choice of the bias according to the variant, ReLU, rounding (including negative halves),
saturation and extreme values (M = 65,535, s = 31), modulo 2^32 overflow of the bias addition, host port (latency 1,
inactive during `busy`), reference model over 1200 cycles. **End to end**: the chain `data_feeder` → `pe_array` →
`output_drain` → `post_processing` computes `requant(ReLU(W·X + b))`; the int8 Y is rebuilt from the writes alone, and
OS and WS write exactly the same Y.

**Measured**: with M on 16 bits, the requantization is identical to the ideal `round(x · scale)` (from 14 bits
onwards); with 8 bits, the error reaches 1 int8 step. A 64 → 32 → 16 MLP run on the simulated NPU reproduces the
floating-point network with a correlation of 0.9998.
