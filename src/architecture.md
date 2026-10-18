# Architecture of minuscul_npu

Reference document for the architecture decisions.
Prototyping in py4hw (work shared with AI agents), then manual transcription into Verilog (done entirely by a human,
outside the scope of the agents). Onboarding guide for the agents: [`CLAUDE.md`](../CLAUDE.md).

Statuses: `OPEN` (to be decided) · `PROPOSED` (recommendation, to be confirmed) · `DECIDED`

The implementation progress and the measured results are in [section 3b](#3b-implementation-and-verification).

---

## 0. Goals and roadmap

**Main goal**: learn and understand a *working* NPU architecture, and explore its trade-offs and its points of
improvement.

**Guiding principle**: a **parameterisable** architecture (array size, bit widths, buffer sizes) so that the impact of
each choice can be measured.

| Version | Target | Priority | Architecture |
|---|---|---|---|
| v1 | Simulation (py4hw) | Clarity, understanding | Reference |
| v2 | FPGA | Fit and work on real hardware | Derived from v1, may diverge |
| v3 | Tiny Tapeout | Minimal area and pins | May diverge strongly |

---

## 1. Overview

```
 Host ◄──► host_interface ──► controller ──────────────── (control to all blocks)
                                   │
 Ext. mem. ◄──► dma (v2) ──► weight_buffer ──────┐
                         └──► activation_buffer ─┼──► data_feeder ──► pe_array (N×N pe)
                                      ▲          │                        │
                                      │          │                  output_drain
                                      │          │                        │
                                      └──────────┴──────────────── post_processing
```

## 2. Functional blocks

The detailed interface of each block (ports, widths, latencies) is in `src/<block>/interface.md`, following the
conventions of [section 5](#5-interface-conventions).

| Block | Function |
|---|---|
| `npu_top` | Assembly of all the blocks, pins of the chip |
| `host_interface` | Host bus protocol + configuration/status register bank |
| `controller` | Sequencer: tile loops, generation of the addresses and control signals |
| `dma` | Transfers external memory ↔ internal buffers |
| `weight_buffer` | Local SRAM for the weights W |
| `activation_buffer` | Local SRAM for the activations X (layer inputs and outputs) |
| `data_feeder` | Reading of the buffers and skewed (staircase) injection into the array |
| `pe` | Processing Element: w/x pass-through registers + MAC + accumulator |
| `pe_array` | N×N array of `pe` connected to their neighbours |
| `output_drain` | Collection and serialisation of the accumulators |
| `post_processing` | Bias → activation → requantization → saturation |

---

## 3. Architecture decisions

### D1 — Implementation target · `DECIDED`: in stages (simulation → FPGA → Tiny Tapeout)
- v1 in simulation, optimised for learning: no physical constraint.
- v2 on FPGA (model to be specified later).
- v3 on Tiny Tapeout (~8 inputs / 8 outputs / 8 bidirectional, tiny area).
- Each version may have its own architecture. The differences with respect to v1 will be documented.

### D2 — Reference workload · `DECIDED`: generic GEMM
- The NPU computes `Y = W · X (+ b)` for parameterisable dimensions M, K, N.
- No native convolution: a convolution goes through im2col, done on the host side.
- Consequence: pooling is outside the scope of `post_processing` (D10).

### D3 — Dataflow · `DECIDED`: output-stationary AND weight-stationary, comparable
- Both variants are implemented behind **the same external interface**, so that they can be compared.
- Row-stationary: rejected (too complex, and designed for convolutions).

| | Output-stationary (OS) | Weight-stationary (WS) |
|---|---|---|
| Stationary in the PE | int32 accumulator | int8 weight |
| Flows between PEs | W (8 b) →, X (8 b) ↓ | X (8 b) →, psum (32 b) ↓ |
| Preload phase | no | yes (weights) |
| Output of the results | draining of the accumulators | stream at the bottom of the array |
| Array for one tile | M × N | K × M |

**Variant boundary — the "compute core"**: it is not only the PE that changes.
`data_feeder`, `pe`, `pe_array` and `output_drain` differ between OS and WS.
The compute core is therefore the set of these four blocks. Its common interface:
- as input: reading from `weight_buffer` and `activation_buffer` + commands from the `controller`;
- as output: stream of int32 results to `post_processing`.

The buffers, the `controller` (apart from a few phases), `post_processing` and `host_interface` are shared.

**Comparison metrics**: number of cycles per GEMM, number of SRAM reads, number of registers and of wire bits between
PEs, PE utilisation rate.

### D4 — Size of the PE array · `DECIDED`: 4×4 by default, parameterisable
- 16 PEs: enough to observe the pipeline, the skewing and the tiling.
- The parameter N remains free (2×2 for debugging, 8×8 for exploration).
- Area ∝ N², peak throughput = N² MACs/cycle.

### D5 — Number format · `DECIDED`: signed int8 × signed int8 → signed int32 accumulator
- Two's complement everywhere.
- Signed 8×8 product: at most (−128)² = 16,384 = 2^14. An int32 accumulator therefore cannot overflow as long as
  K ≤ 131,071 (value measured on the PE in `test/pe/benchmark.py`), far beyond `DIM_MAX` = 64.
  For K ≤ 64, 22 bits would be enough: a reduction lead for v3.
- The widths remain parameters (int4 exploration, reduced accumulator, etc.).

### D6 — Internal memory · `DECIDED`
- **Separate** `weight_buffer` and `activation_buffer`, read in parallel (no access conflict).
- **Single buffer**: no overlap between loading and computation in v1.
  Double buffering is a trade-off to be explored later (PE utilisation versus ×2 memory).
- Read width: one row of N int8 elements per cycle (N×8 bits), to feed the array at full throughput.
- The int8 results are written into `activation_buffer` (output area), where the host reads them back.
- **Sizes**: matrices up to **64×64** (parameter `DIM_MAX`). Everything fits in memory, and the `controller` does all
  the tiling.

| Memory | Contents | Size (DIM_MAX = 64) |
|---|---|---|
| `weight_buffer` | W, int8, M×K | 4 KiB |
| `activation_buffer` | X as input (K×N) + Y as output (M×N), int8 | 8 KiB (2 areas) |
| biases (in `post_processing`) | b, int32, M values | 256 B |
| accumulators (in `output_drain`, **WS only**) | int32 partial sums: one entry per column j (64), N = 4 lanes | 1 KiB |

- Dimension registers M, K, N: 7 bits each, values from 1 to 64.
- With a 4×4 array, a 64×64×64 GEMM has (64/4)² = 256 output tiles, each with 64 steps in K.

#### Data layout · `DECIDED`: tiled, specific to each variant
- A memory word is 32 bits wide and holds **4 × int8**. These are exactly the 4 elements that the array consumes in
  one cycle. The element of lane `l` (l = 0..3) occupies byte `l`, byte 0 being the least significant one.
- The tile size is fixed (`DIM_MAX` = 64), so the address is formed by **bit concatenation**, without a multiplier:
  `address = {tile_index[3:0], index[5:0]}`.

| Matrix | Variant | Word at address `{t, u}` | holds |
|---|---|---|---|
| W | OS and WS | `{i_t, k}` | `W[4·i_t + l][k]` |
| X | OS | `{j_t, k}` | `X[k][4·j_t + l]` |
| X | WS | `{k_t, j}` | `X[4·k_t + l][j]` |
| Y | OS | `{j_t, i}` | `Y[i][4·j_t + l]` |
| Y | WS | `{i_t, j}` | `Y[4·i_t + l][j]` |

- The format of **W is common** to both variants. That of **X is transposed** between OS and WS.
- Each variant writes Y **in the X format it expects as input**. The output of a layer can therefore be given directly
  as the input of the next layer, without reformatting.
- The host formats W and X according to the variant. Trade-off set aside for now: a hardware transposer, or
  diagonally interleaved banks (*skewed storage*), to be explored in a later version.

### D7 — Host interface · `DECIDED`: simple 32-bit parallel bus
- Signals: `addr[12:0]` (word address), `wdata[31:0]`, `we`, `re`, `rdata[31:0]`.
  One access per cycle, fixed read latency.
- A bus word corresponds to a buffer word (4 × int8) or to an int32 bias.
- Planned adapters: Wishbone (v2 FPGA), SPI (v3 Tiny Tapeout).

#### Memory map · `DECIDED`
| Address (words) | Name | Access | Contents |
|---|---|---|---|
| `0x0000` | `ID` | RO | dataflow type (0 = OS, 1 = WS), array size N, `DIM_MAX` |
| `0x0001` | `CTRL` | RW | b0 `GO` (automatically cleared to 0), b1 `RELU_EN` |
| `0x0002` | `STATUS` | RO | b0 `BUSY`, b1 `DONE` (cleared to 0 by `GO`) |
| `0x0003` | `DIM_M` | RW | 1..64 |
| `0x0004` | `DIM_K` | RW | 1..64 |
| `0x0005` | `DIM_N` | RW | 1..64 |
| `0x0006` | `REQ_MULT` | RW | M, 16 bits unsigned |
| `0x0007` | `REQ_SHIFT` | RW | s, 5 bits |
| `0x0008` | `CYCLES` | RO | number of cycles of the last GEMM, for the benchmarks |
| `0x0009`–`0x03FF` | — | — | reserved |
| `0x0400`–`0x07FF` | `WEIGHT` | RW | `weight_buffer`, 1024 words |
| `0x0800`–`0x0BFF` | `ACT_IN` | RW | `activation_buffer`, input area X |
| `0x0C00`–`0x0FFF` | `ACT_OUT` | RW | `activation_buffer`, output area Y |
| `0x1000`–`0x103F` | `BIAS` | RW | int32 biases, one per output row |

- During a GEMM (`BUSY = 1`, or a `go` pulse in progress), the host's accesses to the memories and the configuration
  writes are **ignored**; the registers remain readable.
- Possible improvement: a bit that swaps the roles of `ACT_IN` and `ACT_OUT`, to chain layers without a copy.

### D8 — External memory access · `DECIDED`: no DMA in v1
- The buffers are mapped into the address space of `host_interface`, and the host fills them and reads them back
  directly.
- The `dma/` folder is reserved for v2 (FPGA).

### D9 — Control · `DECIDED`: hardwired FSM
- The host writes the configuration (M, K, N, post-processing parameters; the memory areas are fixed) into registers,
  then issues `go`.
- The `controller` runs the complete GEMM (all the tiles) and raises `done`.
- The phases depend on the dataflow: WS adds a weight preload phase.
- Possible evolution: a mini-ISA in a later version (trade-off to be explored).

#### Order of the tile loops · `DECIDED`
```
OS :  for i_t in tiles(M):           WS :  for i_t in tiles(M):
        for j_t in tiles(N):                 for k_t in tiles(K):
          clear accumulators                   preload W[4·i_t.., 4·k_t..]
          for k in 0..K-1:                     for j in 0..N-1:
            inject W, X                          inject X[4·k_t.., j]
          drain → post_processing                acc[.., j] += psum at the bottom of the array
                                             acc → post_processing
```
- **OS**: k is necessarily the inner loop, since the accumulation takes place in the PE. There is no cache between the
  buffers and the array, so the order of (i_t, j_t) does not change the number of SRAM reads.
- **WS**: with i_t on the outside and k_t in the middle, the accumulator memory is limited to 4 × N int32 (1 KiB).
  The order k_t → i_t → j would require M × N int32 (16 KiB). This is a counter-example to be measured (not done yet).
- In both cases, i_t is on the outside, so post-processing receives the output rows in order.

#### Dimensions that are not multiples of 4 · `DECIDED`: masking on read and on write
- The `data_feeder` injects **0** into every lane whose index goes beyond the limit (i ≥ M, j ≥ N or k ≥ K).
  A 0 is neutral for the MAC, so the computation stays correct without the host having to pad its matrices.
- Out-of-bounds writes of Y (i ≥ M or j ≥ N) are **blocked**: each byte has its own write mask.
- In OS, K does not need this masking (we simply send K steps). In WS, it is the lanes k ≥ K that are masked.
- Hardware cost: one comparator per lane and per dimension.
- Variant to be compared (not done yet): an enable signal per PE, which freezes the unused PEs to save energy.

### D10 — Post-processing operations · `DECIDED`: bias → ReLU → int8 requantization
Pipeline, in this order:
1. **Bias**: `acc + b[i]`, with b in int32, one bias per output row.
2. **ReLU**: `max(0, x)`, enabled or not by a configuration bit.
3. **Requantization**: `y = sat8( (x × M + 2^(s-1)) >> s )`
   - M: 16 bits unsigned (register `REQ_MULT`). s: 5 bits, from 0 to 31 (register `REQ_SHIFT`). If s = 0, there is no
     rounding.
   - 32 × 16-bit multiplier, 48-bit product. The shift `>>` is arithmetic (it preserves the sign).
   - Round to nearest, half towards +∞ (addition of 2^(s-1)).
   - Saturation to [-128, 127].
   - **Per-layer granularity**: a single (M, s) pair for the whole GEMM.
     One pair per output row is a possible evolution (64 more words, next to the biases).
- **Symmetric quantization**: the zero point is 0 for W, X and Y, so there is no zero-point correction.
- **Throughput**: `post_processing` processes **4 lanes in parallel**, at the pace of the array. It therefore contains
  4 32×16 multipliers. A single-lane version, smaller but slower at draining, is a trade-off to be measured.
- Rejected: pooling (outside the GEMM scope), sigmoid/tanh LUT, int32 bypass.
- Note: without a bypass, the raw accumulators will be checked directly in the py4hw simulation.

### D11 — Clock and reset · `DECIDED`: a single clock, synchronous active-low reset
- A single clock domain `clk`, hence no clock domain crossing (CDC).
- Synchronous active-low reset `rst_n`, as in the Tiny Tapeout convention.
- Frequency: not applicable in v1. It will be set in v2 according to the FPGA.

### D12 — Protocol between blocks · `DECIDED`: centralised control
- The `controller` knows the latency of each block and drives everything deterministically, to the exact cycle.
- No internal valid/ready in v1. It will become necessary in v2 with a DMA (variable memory latency).

---

## 3b. Implementation and verification

### Implementation style
Each block is written in **behavioural py4hw**, in `src/<block>/<block>.py`: a `Logic` class whose `clock()` method
describes the registers (updated at the edge, outputs published with `prepare()`) and, if the block has combinational
outputs, a `propagate()` method (outputs published with `put()`). Purely structural blocks (`pe_array`) merely
instantiate and wire other blocks.
The dataflow variants are distinct classes (`…_OS`, `…_WS`) with the same interface.

### Verification
Each block and each assembly has a testbench in `test/<block>/`, made of `test.py` (functional), `benchmark.py`
(limits and trade-offs, matplotlib figures) and `animation.py` (cycle-by-cycle visualisation).
The complete convention is described in [`test/README.md`](../test/README.md).

### Progress
| Block | Implementation | Tests | Benchmark | Animation |
|---|---|---|---|---|
| `pe` | ✅ `PE_OS`, `PE_WS` | ✅ 18 | ✅ | ✅ |
| `pe_array` | ✅ `PE_Array_OS`, `PE_Array_WS` | ✅ 10 | ✅ | ✅ |
| `data_feeder` | ✅ `DataFeeder_OS`, `DataFeeder_WS` | ✅ 7 | ✅ | ✅ |
| `output_drain` | ✅ `OutputDrain_OS`, `OutputDrain_WS` | ✅ 10, including end-to-end GEMM | ✅ | ✅ |
| `post_processing` | ✅ `PostProcessing_OS`, `PostProcessing_WS` | ✅ 12, including end-to-end GEMM | ✅ | ✅ |
| `weight_buffer` | ✅ `WeightBuffer` | ✅ 6 | ✅ | ✅ |
| `activation_buffer` | ✅ `ActivationBuffer` | ✅ 8, including two layers chained through the memory | ✅ | ✅ |
| `controller` | ✅ `Controller_OS`, `Controller_WS` | ✅ 8, including cycle-by-cycle identity with the scheduler and complete NPU without `host_interface` | ✅ | ✅ |
| `host_interface` | ✅ `HostInterface` | ✅ 9 | ✅ | ✅ |
| `npu_top` | ✅ `NpuTop` | ✅ 8, complete chip driven through its bus | ✅ | ✅ |
| `dma` | v2 | | | |

The **complete datapath with its memories** (`weight_buffer`, `activation_buffer` → `data_feeder` → `pe_array` →
`output_drain` → `post_processing` → `activation_buffer`) is verified end to end, driven as the host will drive it:
loading of W, X and the biases according to the D6 layout, multi-tile GEMM (dimensions that are not multiples of N,
"dirty" memory, biases, ReLU, requantization), reading Y back, then two layers chained by a simple OUT → IN copy. Both
variants write exactly the same Y.

**The complete chip (`npu_top`) works**: driven only through its 32-bit bus and its memory map, by reference host
software (`test/npu_top/test.py`), it computes correct layers in both variants, chains layers and runs a small
quantized neural network.

### Measured results (N = 4)
| Measurement | OS | WS | Source |
|---|---|---|---|
| PE latencies | 1 cycle | 1 cycle | `test/pe/benchmark.py` |
| Cycles for an isolated tile (stream of length L) | L + 2N − 2 | L + 3N − 2 (preload included) | `test/pe_array/benchmark.py` |
| Minimum gap before reloading the weights | — | 2N − 2 cycles (FLUSH = 2N − 3) | `test/pe_array/test.py` |
| Useful MACs per byte read from SRAM (tile, stream of 64) | 2.0 | 3.8 | `test/data_feeder/benchmark.py` |
| Required SRAM bandwidth (tile, stream of 64) | 7.2 bytes/cycle | 3.6 bytes/cycle | `test/data_feeder/benchmark.py` |
| Lanes lost to masking (D = 1..64, average) | 8 % | 8 % | `test/data_feeder/benchmark.py` |
| End-to-end 16×16×16 GEMM (last result) | 448 cycles | 403 cycles | `test/output_drain/benchmark.py` |
| Array utilisation, 16³ / 32³ GEMM | 57 % / 73 % | 64 % / 78 % | `test/output_drain/benchmark.py` |
| Last word of Y written, 16³ GEMM (with post_processing) | cycle 451 | cycle 406 | `test/post_processing/benchmark.py` |
| Requantization error, M on 16 bits / 8 bits | 0 / 1 int8 step | identical | `test/post_processing/benchmark.py` |
| 64 → 32 → 16 MLP on the simulated NPU versus floating point | correlation 0.9998 | identical | `test/post_processing/benchmark.py` |
| Reads of W for a 32³ GEMM (minimum 256) | 2048 | 256 | `test/weight_buffer/benchmark.py` |
| Share of computation in the total time (host included), 4³ / 64³ GEMM | 50 % / 86 % | 56 % / 86 % | `test/activation_buffer/benchmark.py` |
| 64 → 48 layer on a batch of 16 (WS worse when the batch is small) | 3652 cycles | 4812 cycles | `test/activation_buffer/benchmark.py` |
| 64³ GEMM driven by the controller (go → done) | 19,461 cycles | 18,701 cycles | `test/controller/benchmark.py` |
| Utilisation when K = 4 (M = N = 32) / when N = 4 (M = K = 32) | 25 % / 72 % | 75 % / 30 % | `test/controller/benchmark.py` |
| Chip throughput, 64³ layer, host included / computation alone | 11.6 / 13.5 MACs/cycle | 12.0 / 14.0 MACs/cycle | `test/host_interface/benchmark.py` |
| Quantized floating-point 64 → 32 → 16 MLP, on the chip through the bus | correlation 0.9999 | identical | `test/npu_top/benchmark.py` |
| 64 → 64 layer, cycles per sample, batch of 1 / of 64 | 2446 / 353 | 3702 / 341 | `test/npu_top/benchmark.py` |

The measured cycles follow exactly the formulas of `src/controller/interface.md`
(OS: `T(M)·T(N)·(K + 3N)` = 16 · 28 = 448).

### Identified leads, not yet explored
- OS: overlap computation and draining (shadow register per PE) — draining and reading take 39 % of the cycles of a 16³
  GEMM.
- WS: double weight buffer to overlap the preload (16 % of the cycles of a 16³ GEMM).
- Reduced accumulator (22 bits are enough for K ≤ 64).
- WS loop order k_t → i_t → j, per-PE enable, double buffering of the SRAMs.
- DMA (v2): loading by the host weighs up to half of the time on small GEMMs.
- Bit swapping the IN / OUT roles: would remove the copy of Y between two layers.

---

## 4. Decision log

| Date | Decision | Choice | Rationale |
|---|---|---|---|
| 2026-10-18 | Split into blocks | 11 blocks (section 2) | 1:1 correspondence with Verilog modules |
| 2026-10-18 | D1 Target | Simulation → FPGA → Tiny Tapeout | Learn first, then confront real constraints |
| 2026-10-18 | D2 Workload | Generic GEMM | Kernel of every network, controlled scope |
| 2026-10-18 | D3 Dataflow | Comparable OS + WS | Explore the weight reuse / simplicity trade-off |
| 2026-10-18 | D4 Array | 4×4 by default, parameterisable | Readable in simulation, non-trivial tiling |
| 2026-10-18 | D5 Format | signed int8 → int32 acc | Standard for quantized inference |
| 2026-10-18 | D9 Control | Hardwired FSM | Simplicity; the mini-ISA remains a possible evolution |
| 2026-10-18 | D6 Memory | Separate W/X buffers, single buffer | Parallel reads without conflict, simple control |
| 2026-10-18 | D8 DMA | None in v1, the host writes directly | Focus the effort on the compute core |
| 2026-10-18 | D10 Post-processing | Bias → ReLU → int8 requant | Output reusable as the input of the next layer |
| 2026-10-18 | D12 Protocol | Centralised control | Deterministic, typical of a systolic array |
| 2026-10-18 | D6 Sizes | Matrices up to 64×64 | Tiling entirely internal, reasonable size in simulation |
| 2026-10-18 | D7 Interface | Simple parallel bus | Simple in simulation, adaptable to Wishbone or SPI |
| 2026-10-18 | D11 Clock/reset | 1 clk, synchronous rst_n | Tiny Tapeout convention, no CDC |
| 2026-10-18 | D6 Layout | Tiled, specific to each variant | 1 word = 1 read of the array, addresses by concatenation, Y in X format |
| 2026-10-18 | D10 Requant | M 16 b, s 5 b, rounding, per layer, symmetric | Good precision for a 32×16 multiplier |
| 2026-10-18 | D7 Memory map | 32 b bus, 13 address bits (section D7) | Registers + direct windows onto the buffers |
| 2026-10-18 | D9 Loops | OS i→j→k, WS i→k→j | WS accumulators limited to 1 KiB |
| 2026-10-18 | Edges | Masking on read (0) and on write | Transparent for the host, a few comparators |
| 2026-10-18 | pe OS | `clear` broadcast to all PEs, with priority | Simple reset to zero, without a signal that follows the data |
| 2026-10-18 | pe WS | Weights preloaded by vertical shifting (`load`) | No write bus to each PE |
| 2026-10-18 | OS: draining | No computation / draining overlap, period K + 3N | Simplicity; a shadow register per PE would allow overlapping (to be explored) |
| 2026-10-18 | WS: flush | 2N − 3 cycles before the next preload, period N_dim + 3N − 3 | Exact limit, **confirmed in simulation**: with one cycle less, the last column is wrong (`test/pe_array/test.py`) |
| 2026-10-18 | data_feeder | It aligns the commands (+1 cycle), masks and skews | The `controller` emits everything at the same emission cycle |
| 2026-10-18 | Memories | Single-port SRAM, synchronous read L = 1, port selected by `busy` | Realistic for an ASIC or a BRAM; two IN / OUT banks for the activations |
| 2026-10-18 | Tag | 20 bits `{y_addr, lane_mask, row}` travelling with the results | Writing Y does not need the `controller` |
| 2026-10-18 | output_drain WS | Deskewing + 2N-cycle delay line + accumulator memory with combinational read | Read-modify-write in one cycle (simple in simulation, to be pipelined for an ASIC) |
| 2026-10-18 | post_processing | 3-stage pipeline, N lanes, biases in N banks | Throughput of one input per cycle; one bias per lane in WS, a single broadcast one in OS |
| 2026-10-18 | host_interface | Read latency 1 everywhere, reset REQ_MULT = 1, `irq = DONE` | Uniform behaviour; neutral requantization by default |
| 2026-10-18 | Invalid dimensions | 0 or > 64: immediate `done`, nothing is computed | Defined rather than undefined behaviour |
| 2026-10-18 | Implementation | Behavioural py4hw (`clock()` / `propagate()`), one class per variant | Readable code, close to the specification; the transcription into Verilog is done by hand |
| 2026-10-18 | output_drain OS | N passed as an explicit parameter | N cannot be deduced from the width of `drain_row` when N is not a power of 2 |
| 2026-10-18 | controller | Moore machine; `busy` = 0 during DONE; `cycles` = cycles outside IDLE since `go`, DONE included | During DONE the last write of Y is finished: the host can read back as soon as `done` is raised |
| 2026-10-18 | controller | Outputs published in `clock()`, no `propagate()` | Avoids a false combinational loop host_interface ⇄ controller in the simulator |
| 2026-10-18 | host_interface | "Busy" = `busy` OR `go` pulse in progress; memory and configuration accesses ignored when busy | Without this, an access right after GO went through before the computation started (found by the npu_top tests) |
| 2026-10-18 | Language | The whole repository translated into English: documentation, code and identifiers, comments, messages, figures | Readable by a wider audience on the public repository |

---

## 5. Interface conventions

These rules apply to all the `src/<block>/interface.md` files.

### 5.1 Common parameters
| Name | Default | Definition | Role |
|---|---|---|---|
| `DATAFLOW` | OS | OS or WS | variant of the compute core, fixed at construction time (it is not a register) |
| `N` | 4 | — | side of the array; number of lanes |
| `DW` | 8 | — | width of a data word (signed int8) |
| `AW` | 32 | — | width of an accumulator or of a partial sum (signed int32) |
| `DIM_MAX` | 64 | — | maximum dimension of M, K and N |
| `IW` | 6 | log2(DIM_MAX) | width of an element index (i, j, k) |
| `TW` | 4 | log2(DIM_MAX / N) | width of a tile index (i_t, j_t, k_t) |
| `BAW` | 10 | TW + IW | width of a word address in a buffer |
| `DIMW` | 7 | log2(DIM_MAX) + 1 | width of the dimension registers (from 1 to 64) |
| `TAG_W` | 20 | BAW + N + IW | width of the tag |

The blocks of the compute core remain parameterisable in N. The v1 of `host_interface` and `npu_top` assumes N = 4,
since a buffer word is then 32 bits wide, the width of the bus.

### 5.2 Naming and signal conventions
- Signals are in `snake_case`. They are active high, except `rst_n`.
- All sequential blocks receive `clk` and `rst_n` (synchronous active-low reset, D11).
  The contents of the memories are not reset.
- Data are in two's complement: int8 for the data, int32 for the accumulators.
- Suffixes: `_addr` (address), `_we` / `_re` (write / read enable), `_wdata` / `_rdata` (data), `_mask` (one bit per
  lane), `_valid` (the data accompanying this bit can be used).
- On the array side, the ports are named after the cardinal points: `_west`, `_north`, `_south`.
- **Lane ordering**: a vector of N elements of W bits is packed, lane `l` occupying bits `[W·l + W−1 : W·l]`. Lane 0
  is therefore in the least significant bits. This is also the order of the bytes in a memory word (D6).
- A matrix of N×N elements (`acc` in OS) is stored row by row: element (i, j) is at index `i·N + j`.

### 5.3 Latency and emission cycle
- **Latency L = n**: an input present during cycle t produces its effect on the output during cycle t + n.
  L = 0 denotes a combinational path.
- A write requested during cycle t takes effect at the edge ending that cycle: the data can be read from t + 1 onwards.
- **Memories**: the read is synchronous, with L = 1. `rdata` is valid in the cycle following `re`.
- **Emission cycle**: this is the cycle in which the `controller` presents an SRAM address and all the commands that go
  with it (masks, `ctl`, sideband, tag). This cycle is the reference for all the timing diagrams.
  Each block placed after the SRAM delays its commands by as much as the data: the `data_feeder` adds 1 cycle, the
  delay line of `output_drain` (WS) adds 2N cycles.
- **Centralised control** (D12): there is no handshake. Each latency is fixed and known to the `controller`.

### 5.4 Tag
The tag accompanies each group of N results, from `output_drain` up to the write into `activation_buffer`.

| Bits | Field | Width | Contents |
|---|---|---|---|
| [19:10] | `y_addr` | BAW | address of the Y word in the OUT bank (OS: `{j_t, i}` · WS: `{i_t, j}`) |
| [9:6] | `lane_mask` | N | write enable of each lane (0 when out of bounds) |
| [5:0] | `row` | IW | row index for the bias (OS: i · WS: 4·i_t) |

### 5.5 Summary of the latencies
| Path | Latency |
|---|---|
| Emission → lane l at the input of the array (skewed stream) | 1 + l |
| Emission → `w_north` in WS (preload) | 1 |
| `pe`: input → output | 1 |
| `output_drain` OS: `drain_en` → `res_valid` | 1 |
| `output_drain` WS: emission of X → `res_valid` | 2N + 1 |
| `post_processing`: `res_valid` → `y_we` | 3 |
| Host bus: `bus_re` → `bus_rdata` | 1 |
