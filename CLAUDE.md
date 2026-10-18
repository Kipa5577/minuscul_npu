# Onboarding guide for AI agents

This file is intended for the AI agents working on this repository. Read it in full before modifying anything, then
read [`src/architecture.md`](src/architecture.md) and [`test/README.md`](test/README.md).

## 1. The project and your scope

`minuscul_npu` is a small NPU (neural network accelerator) designed for **learning**: understanding a working
architecture and **exploring its trade-offs**. It computes `Y = requant(ReLU(W·X + b))` in int8 on a 4×4 systolic
array, in two compared variants: output-stationary (OS) and weight-stationary (WS).

| Part | Who |
|---|---|
| Architecture in py4hw (blocks, testbenches, benchmarks, animations, documentation) | **the AI agents** |
| Architecture decisions | **the user** (the AI analyses, proposes, then records the decision) |
| Transcription into Verilog, FPGA, Tiny Tapeout | **a human, entirely by hand** |

**Do not work on the Verilog**: do not write it, do not prepare it, do not propose it as a next step, do not add
"Verilog equivalent" comments in the code. The user may ask for help on other parts: do so only if they ask.

## 2. Current status (v1 complete)

The 10 blocks of v1 are implemented, tested (96 tests), measured and animated; the complete chip (`npu_top`), driven
only through its 32-bit bus, computes correct layers and runs a small quantized neural network.
The block-by-block status, the measured results and the **leads not yet explored** are in section 3b of
`src/architecture.md`: it is the natural starting point for what comes next (v2: DMA, double buffering, overlap of
computation / draining, reduced accumulator, etc.). The `dma/` folder is reserved for v2. The `algorithms/` folder, at
the root, is empty: ask the user what it is for before putting anything in it.

## 3. Repository map

```
README.md                    public presentation of the project
CLAUDE.md                    this guide
src/architecture.md          decisions (D1 to D12), interface conventions (§5), status and results (§3b), log (§4)
src/<block>/interface.md     specification of a block: ports, behaviour, latencies, then "Implementation"
src/<block>/<block>.py       py4hw implementation
test/README.md               test organisation, shared tools, drivers
test/common.py               Bench (py4hw testbench), two's complement, pack/unpack, load_test_module, run_tests
test/viz.py                  palette, style of the figures, fast rendering engine for the animations
test/layout.py               data layout in memory (D6), on the host software side
test/<block>/test.py         functional tests
test/<block>/benchmark.py    limits and trade-offs (matplotlib figure)
test/<block>/animation.py    cycle-by-cycle visualisation (≥ 30 frames/s)
```

Reference functions reused from one block to another (to be loaded with `load_test_module`):

| Where | What |
|---|---|
| `test/output_drain/test.py` | `plan_os` / `plan_ws` (what the controller FSM emits, cycle by cycle), `core(variant, post, mem, ctrl)` (incremental assembly of the NPU), `run_core`, `rebuild_y`, `tag_pack` |
| `test/post_processing/test.py` | `reference` (expected Y, requantization included), `gemm_full` |
| `test/activation_buffer/test.py` | host driver of the datapath: `datapath`, `load_*`, `read_ab`, `run_layer` (with a log of the accesses), `run_go` |
| `test/npu_top/test.py` | class `Npu`: the reference host software, which drives the chip only through its bus |
| `test/npu_top/animation.py` | `trace_chip`: activity of each block of the chip, cycle by cycle |
| `test/weight_buffer/animation.py` | `MemGrid`, `Timeline`, `cumulative_states`, `run_animation` (shared animation loop) |

## 4. Running the tests

Python 3.13 with `py4hw`, `numpy` and `matplotlib` (see the README). From the root:

```
for t in pe pe_array data_feeder output_drain post_processing weight_buffer activation_buffer controller host_interface npu_top; do
    python test/$t/test.py | tail -1
done
```

Everything must pass before and after each modification. The files are called `test.py`: pytest does not discover
them on its own, they are run with `python`. `test/compute_core/` contains only placeholder files: the compute core is
tested in `test/output_drain/`.

## 5. Code conventions (requested by the user)

- **Behavioural py4hw**: one `Logic` class per block and per variant (`…_OS`, `…_WS`), with `clock()` for the
  registers (outputs published by `wire.prepare()`) and `propagate()` for the combinational logic (`wire.put()`).
  Purely structural blocks (`pe_array`, `npu_top`) merely instantiate and wire.
- **Specification first**: a block follows `src/<block>/interface.md` and the conventions of `src/architecture.md` §5
  (parameters, naming, lane l in bits `[W·l + W−1 : W·l]`, latencies, emission cycle, tag). If the implementation
  forces a change to the specification, update it and record it in the log.
- **No hardware cost estimates** in the benchmarks (flip-flops, multipliers…): we measure the behaviour (cycles,
  throughput, utilisation, memory accesses, precision).
- **Language**: everything in the repository is in English (documentation, code and identifiers, comments,
  messages, figures). The user writes to you in French: answer them in French.
- **Public repository**: never write any information about the user's machine (local paths, operating system,
  username, workarounds presented as specific to their machine). Before finishing, check:
  `grep -rniE "/mnt/|wsl|windows|/home/|/tmp/" --include=*.py --include=*.md src test README.md` (this guide contains
  the pattern itself, so it is excluded). The "memory windows" of the host bus (D7) are expected matches.

## 6. Testbenches, benchmarks, animations

- `test.py`: each result is compared with an independent reference (NumPy, cycle-by-cycle reference model,
  scheduler). The announced latencies are checked **to the exact cycle** (neither before nor after).
  When tests pass on the first try, check that they bite: temporarily introduce a defect into the block (by replacing
  it at run time) and check that the relevant tests fail.
- `benchmark.py`: a 4-panel matplotlib figure, `--save file.png` option, numerical summary in the terminal, style
  `viz.BENCH_RC`. **Fixed colours: OS in blue (`viz.C_OS`), WS in orange (`viz.C_WS`)**; never use these two colours
  for anything else. Look at the produced figure before concluding (overlapping labels, log scales with zeros, etc.).
- `animation.py`: **at least 30 frames per second, at a constant speed**. The user cares about a polished visual
  style: tokens that move, cells that light up (see `test/pe_array/animation.py` and `test/npu_top/animation.py`).
  The recipe, always the same:
  1. simulate entirely before animating (per-cycle trace);
  2. draw tokens, cells and digits with `viz.Layer`, `viz.TextLayer`, `viz.TokenLayers` (one collection of paths per
     layer, digits converted into paths) — never thousands of `Text` or `Patch` objects;
  3. blitting: every animated artist must belong to an axes (animated title: `viz.title_axes`) and must not touch the
     edge of the axes (otherwise it leaves a trail); static text is covered by the animated layers;
  4. frame chosen according to the real clock (`run_animation` in `test/weight_buffer/animation.py`);
  5. check without a screen by going through the real mechanism of `FuncAnimation` (`anim._init_draw()`, then
     `anim._draw_next_frame(f, True)`), measure the time per frame (target < 15 ms on average) and look at some
     frames.
  `--save file.gif` option in every animation.

## 7. py4hw pitfalls encountered

- **False combinational loop**: the simulator sorts the blocks that have a `propagate()` *block by block*. Two blocks
  that read each other (even through registers) cause "Excessive loop count in topological count". Solution used: a
  block whose outputs depend only on its state (Moore machine) publishes its outputs in `clock()` and has no
  `propagate()` (this is the case of the `controller`).
- **Reading a combinational output**: after `Bench.step()`, you read the state *after* the edge. To read what a block
  outputs *during* a cycle, use `Bench.observe()` (inputs, propagation, read, then edge).
- **`import test`** picks up the `test` module of the standard library: load the testbenches with
  `load_test_module('<block>')`.
- **Fonts**: matplotlib rereads the font file for every text; depending on the file system, this is very slow.
  `common.setup_matplotlib_fonts()` (called by `viz.py`) uses the system copy when it exists.
- **Simulation speed** (order of magnitude): one PE ≈ 400,000 cycles/s, the 4×4 array ≈ 30,000, the complete chip
  ≈ 11,000 to 13,000 cycles/s. Size the benchmarks accordingly.
- **Zero values**: in the animations and the activity measurements, use non-zero data, otherwise a MAC with a zero
  operand is not seen as active. In WS, a MAC can take place during the first weight reload cycle: do not exclude the
  cycles where `load = 1` when counting the MACs.

## 8. Working method with the user

- The user writes in French, learns by building, and wants to **understand the trade-offs**.
- **Architecture decisions belong to the user**: explain the options and their consequences, make a recommendation,
  then ask the question; then record the decision in the log of `src/architecture.md` (§4). The detail decisions you
  make yourself during implementation must also be recorded there and reported to the user.
- After each block or change: update the "Implementation" section of `src/<block>/interface.md`, the status and the
  results (§3b) of `src/architecture.md`, `test/README.md`, the README, and this guide if needed; check that the links
  in the `.md` files point to existing files.
- The pattern that was validated for a block: implementation + `test.py` + `benchmark.py` + `animation.py` +
  documentation, with a report that points out the errors found and fixed along the way.
