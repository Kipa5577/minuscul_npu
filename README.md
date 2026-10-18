# minuscul_npu

A small NPU (*Neural Processing Unit*, a neural network accelerator) designed for **learning**: understanding how a
real inference chip works, from multiply-accumulate all the way to the host bus, and **measuring its architectural
trade-offs** instead of guessing them.

The NPU is prototyped and simulated in Python with [py4hw](https://pypi.org/project/py4hw/), cycle-accurate. Each
block has its specification, its tests, its measurements and an animation that shows it working cycle by cycle. Once
the architecture is settled, the design is transcribed by hand into Verilog, first for FPGA, then for
[Tiny Tapeout](https://tinytapeout.com/).

> **Status: v1 complete.** The chip, driven only through its 32-bit bus, computes correct neural network layers in
> both of its variants, chains layers and runs a small quantized network that reproduces the floating-point
> computation (correlation 0.9999). 96 tests, 10 benchmarks, 10 animations.

---

## Contents

1. [What the NPU computes](#what-the-npu-computes)
2. [How it works](#how-it-works)
3. [The two variants: output-stationary and weight-stationary](#the-two-variants-output-stationary-and-weight-stationary)
4. [Chip architecture](#chip-architecture)
5. [Programming the chip](#programming-the-chip)
6. [Measured results](#measured-results)
7. [Repository layout](#repository-layout)
8. [Installation](#installation)
9. [Tests, benchmarks and animations](#tests-benchmarks-and-animations)
10. [Working method](#working-method)
11. [Roadmap](#roadmap)
12. [Contributing](#contributing)
13. [License](#license)

---

## What the NPU computes

More than 90 % of the computation of a neural network is a **matrix product**. The NPU computes a complete layer,
quantized to integers:

```
Y = requant( ReLU( W · X + b ) )
```

| Element | Format | Size |
|---|---|---|
| W: weights | signed int8 | M × K |
| X: input activations | signed int8 | K × N |
| b: biases | signed int32 | M |
| accumulation | signed int32 | — |
| Y: output activations | signed int8 | M × N |

- M, K and N range from 1 to 64, with no multiple constraint: the edges are handled by masking.
- **Requantization** brings the int32 accumulator back to int8: `y = sat8((x · M_q + 2^(s−1)) >> s)`, with a 16-bit
  multiplier `M_q`, a 5-bit shift `s` and round-to-nearest.
- The quantization is symmetric (zero point = 0). A convolution can be reduced to this computation by im2col, on the
  host side.
- Y comes out in the same format as X: the output of one layer is directly used as the input of the next.

## How it works

A conventional processor spends most of its energy **moving** data, not computing. An NPU is a machine for
**reusing** each piece of data it reads as many times as possible before discarding it.

The heart of the NPU is a **systolic array** of 4×4 *processing elements* (PEs). Each PE performs one int8 × int8 →
int32 multiply-accumulate per cycle and only talks to its immediate neighbours. The data "pulse" from PE to PE: a
value read once from memory feeds 4 MACs as it crosses the array. For the right values to meet at the right PE, the
data enter **in a staircase**: lane l is delayed by l cycles.

```
            x0       x1       x2       x3          ← activations, skewed in time
            ↓        ↓        ↓        ↓
   w0 →  [ PE ] → [ PE ] → [ PE ] → [ PE ]
            ↓        ↓        ↓        ↓
   w1 →  [ PE ] → [ PE ] → [ PE ] → [ PE ]        16 MACs per cycle at most
            ↓        ↓        ↓        ↓
   ...
```

A matrix larger than the array is cut into 4×4 **tiles**, which the sequencer (`controller`) walks through one by one.

## The two variants: output-stationary and weight-stationary

The central question of an NPU is: *which piece of data stays still in the PE?* The project implements the two
classic answers behind the same interface, to compare them with figures to back it up.

| | Output-stationary (OS) | Weight-stationary (WS) |
|---|---|---|
| Stays in the PE | the int32 accumulator | the int8 weight (preloaded) |
| Flows between the PEs | W to the east, X to the south (8 bits) | X to the east, partial sums to the south (32 bits) |
| End of a tile | the accumulators are read row by row | the results come out in a staircase at the south |
| Reuse | W and X read again for each tile | each weight read only once per GEMM |
| Measured advantage | large K, small N, small batches | large N, large batches |

The variant is chosen when the chip is constructed (`dataflow='OS'` or `'WS'`). Both produce exactly the same Y, bit
for bit.

## Chip architecture

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

| Block | Role |
|---|---|
| `pe` | Processing element: int8 → int32 multiply-accumulate, pass-through registers to the neighbours |
| `pe_array` | N×N array of PEs (4×4 by default, parameterisable) |
| `data_feeder` | Aligns the commands with the memory, masks out-of-bounds lanes, creates the systolic staircase |
| `output_drain` | Collects the results from the array; in WS, adds up the partial sums of the K tiles |
| `post_processing` | Bias → ReLU → int8 requantization, on 4 lanes in parallel; bias memory |
| `weight_buffer` | Weight SRAM (4 KiB) |
| `activation_buffer` | Activation SRAM: IN bank (X) and OUT bank (Y), 4 KiB each |
| `controller` | FSM that walks through the tiles and emits addresses, masks and commands, to the exact cycle |
| `host_interface` | 32-bit bus, configuration and status registers, windows onto the memories |
| `npu_top` | Assembly: the complete chip, seen from the outside through its bus and its interrupt line |

Main choices (detailed and justified in [`src/architecture.md`](src/architecture.md)):
- a single clock domain, with a synchronous active-low reset;
- centralised, deterministic control: each latency is fixed and known to the `controller`, with no handshake;
- single-port memories with synchronous read: the host accesses them when the chip is idle, the core during the
  computation;
- a 20-bit tag (Y address, write mask, row) that travels with each group of results, so that writing Y does not need
  the `controller`.

## Programming the chip

### Memory map (addresses of 32-bit words)

| Address | Name | Access | Contents |
|---|---|---|---|
| `0x0000` | `ID` | read | array size, variant (0 = OS, 1 = WS), `DIM_MAX`, version |
| `0x0001` | `CTRL` | read / write | bit 0 `GO` (starts the computation), bit 1 `RELU_EN` |
| `0x0002` | `STATUS` | read | bit 0 `BUSY`, bit 1 `DONE` (also on the `irq` line) |
| `0x0003`–`0x0005` | `DIM_M`, `DIM_K`, `DIM_N` | read / write | dimensions, from 1 to 64 |
| `0x0006` | `REQ_MULT` | read / write | requantization multiplier (16 bits) |
| `0x0007` | `REQ_SHIFT` | read / write | requantization shift (5 bits) |
| `0x0008` | `CYCLES` | read | duration of the last computation, in cycles |
| `0x0400`–`0x07FF` | `WEIGHT` | read / write | weight memory |
| `0x0800`–`0x0BFF` | `ACT_IN` | read / write | input activations X |
| `0x0C00`–`0x0FFF` | `ACT_OUT` | read / write | output activations Y |
| `0x1000`–`0x103F` | `BIAS` | read / write | int32 biases |

A memory word holds 4 int8 values. The way W, X and Y are stored in words depends on the variant: it is described in
`src/architecture.md` (D6) and implemented on the host side in [`test/layout.py`](test/layout.py). During a
computation, memory accesses and configuration writes are ignored; the registers remain readable.

### Sequence for one layer

1. Write W into `WEIGHT`, X into `ACT_IN` and b into `BIAS`.
2. Write `DIM_M`, `DIM_K`, `DIM_N`, `REQ_MULT` and `REQ_SHIFT`.
3. Write `CTRL = GO | RELU_EN`.
4. Wait for `irq`, or poll `STATUS.DONE`.
5. Read Y from `ACT_OUT`, and `CYCLES` for the measurement.

For the next layer, it is enough to copy the words of Y from `ACT_OUT` to `ACT_IN`.

### In Python, on the simulated chip

The class `Npu` in [`test/npu_top/test.py`](test/npu_top/test.py) is the reference host software. It knows only the
bus and the memory map. From the root of the repository:

```python
import sys
sys.path.insert(0, 'test')
from common import load_test_module
import numpy as np

npu_test = load_test_module('npu_top')
npu = npu_test.Npu('OS')                      # or 'WS'

rng = np.random.default_rng(0)
W = rng.integers(-128, 128, (16, 32))         # int8 weights, M×K
X = rng.integers(-128, 128, (32, 8))          # int8 activations, K×N
b = list(rng.integers(-1000, 1000, 16))       # int32 biases, one per row of Y

Y, cycles, phases = npu.gemm(W, X, b, mult=300, shift=16, relu=True)
print(Y.shape, cycles, phases)
# (16, 8) 357 {'loading': 213, 'compute': 360, 'readback': 33}
```

## Measured results

All the values come from the simulation (4×4 array). The complete list, with the file that produces each measurement,
is in `src/architecture.md` (section 3b).

| Measurement | OS | WS |
|---|---|---|
| 64×64×64 GEMM, computation alone (go → done) | 19,461 cycles | 18,701 cycles |
| Throughput of a 64³ layer, computation alone / host transfers included | 13.5 / 11.6 MACs/cycle | 14.0 / 12.0 MACs/cycle |
| PE utilisation when K = 4 (M = N = 32) | 25 % | 75 % |
| PE utilisation when N = 4 (M = K = 32) | 72 % | 30 % |
| Useful MACs per byte read from memory (tile, stream of 64) | 2.0 | 3.8 |
| Reads of W for a 32³ GEMM (minimum: 256) | 2,048 | 256 |
| 64 → 64 layer: cycles per sample, batch of 1 / batch of 64 | 2,446 / 353 | 3,702 / 341 |
| Quantized 64 → 32 → 16 MLP, compared with the floating-point computation | correlation 0.9999 | identical |

What these figures teach:
- **No variant wins everywhere.** Depending on the shape of the GEMM, one is up to 3.4 times faster than the other
  (decision map in `test/controller/benchmark.py`).
- **WS reuses the data better**: each weight is read only once, and it needs half the memory bandwidth.
- **Without DMA, the host is expensive**: on a small layer, loading and reading back take more than half of the total
  time.
- **The int32 accumulator is oversized**: for K ≤ 64, 22 bits are enough; a 14-bit requantization multiplier already
  gives the exact result.

## Repository layout

```
README.md                    this file
CLAUDE.md                    onboarding guide for AI agents
LICENSE                      Apache 2.0 license
src/
  architecture.md            architecture decisions, interface conventions, status, results, log
  <block>/interface.md       specification of a block: ports, behaviour, latencies, implementation
  <block>/<block>.py         py4hw implementation (behavioural)
test/
  README.md                  organisation and conventions of the tests
  common.py                  py4hw testbench and utilities
  viz.py                     style of the figures, fast rendering of the animations
  layout.py                  data layout in memory, on the host side
  <block>/test.py            functional tests
  <block>/benchmark.py       measurements of the limits and trade-offs
  <block>/animation.py       cycle-by-cycle visualisation of the block
```

Documents to read, in order:
1. [`src/architecture.md`](src/architecture.md): the *why* (decisions D1 to D12 and their rationale);
2. `src/<block>/interface.md`: the *what* of each block;
3. [`test/README.md`](test/README.md): how everything is verified.

## Installation

Tested with Python 3.13.

```
python -m venv .venv
source .venv/bin/activate
pip install py4hw numpy matplotlib
```

The benchmarks and the animations open a matplotlib window: a graphical backend is needed (for example Tk).
With the `--save` option, they write a file and do not need one.

## Tests, benchmarks and animations

Each block has three files in `test/<block>/`, to be run from the root of the repository:

| File | Role | Command |
|---|---|---|
| `test.py` | checks that the block does exactly what its specification says | `python test/<block>/test.py` |
| `benchmark.py` | measures its limits and its trade-offs (4-panel figure) | `python test/<block>/benchmark.py [--save fig.png]` |
| `animation.py` | shows it working, cycle by cycle (≥ 30 frames/s) | `python test/<block>/animation.py [--save anim.gif]` |

All the tests:

```
for t in pe pe_array data_feeder output_drain post_processing weight_buffer activation_buffer controller host_interface npu_top; do
    python test/$t/test.py | tail -1
done
```

A few entry points:

```
python test/npu_top/animation.py              # the complete chip during one layer
python test/npu_top/animation.py --variant WS # the same thing in weight-stationary
python test/pe_array/animation.py             # the data crossing the systolic array
python test/controller/benchmark.py           # which variant to choose according to the shape of the GEMM
python test/npu_top/benchmark.py              # the chip on real workloads
```

## Working method

- **Specification first.** Each decision is made, justified and dated in `src/architecture.md` before being
  implemented; each block has its specification before its code.
- **Everything is checked against an independent reference**: NumPy for the results, cycle-by-cycle reference models
  for the behaviour, and the announced latencies to the exact cycle. To make sure that the tests really detect errors,
  defects are deliberately introduced into the blocks and we check that the tests fail.
- **We assemble step by step.** The compute core is first driven by a Python model of the sequencer, then we add the
  post-processing, the memories, the real `controller`, and finally the bus: at each step, the whole path is checked
  again end to end.
- **We measure the trade-offs instead of assuming them**, and we show them: each block has its benchmark and its
  animation.

## Roadmap

| Version | Target | Priority | Status |
|---|---|---|---|
| v1 | Simulation (py4hw) | Understand, measure the trade-offs | ✅ complete |
| v2 | FPGA | Fit and work on real hardware | upcoming |
| v3 | Tiny Tapeout | Minimal area and pins | upcoming |

Leads identified by the v1 measurements:
- a **DMA** and a double memory buffer, so that the computation no longer waits for the host transfers;
- in OS, **overlap the computation and the reading of the accumulators** (39 % of the cycles of a 16³ GEMM);
- in WS, a **double weight register** to overlap the preload (16 % of the cycles);
- a **reduced accumulator** (22 bits) and a narrower requantization multiplier, for v3;
- a bit that swaps the IN / OUT banks, to chain layers without a copy.

## Contributing

The py4hw architecture is developed by a human with the help of AI agents; the transcription into Verilog is done
entirely by hand by a human. AI agents (and people) who pick up the work will find in [`CLAUDE.md`](CLAUDE.md) the
scope, the code and test conventions, and the known pitfalls of py4hw.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
