# Test organisation

Each subfolder of `test/` is the testbench of **one block** or of **an assembly of blocks**.
The name of a block is the same as that of its folder in `src/`.

## Running the tests

From the root of the repository, with the project's Python environment activated:

```
python test/<block>/test.py                      # functional tests: summary "n/n tests passed"
python test/<block>/benchmark.py                 # matplotlib figure in a window
python test/<block>/benchmark.py --save b.png    # … or saved to a file
python test/<block>/animation.py                 # animation in a window (≥ 30 frames/s)
python test/<block>/animation.py --save a.gif    # … or saved as a GIF (about one minute)
```

The files are called `test.py` and not `test_*.py`: pytest does not discover them automatically. They are run directly
with `python`, and `test.py` returns a non-zero exit code on failure.

## The three files of a testbench

| File | Role | Question it answers | Result |
|---|---|---|---|
| `test.py` | **Functional** test | "Does the block do exactly what the specification requires?" | Pass / fail |
| `benchmark.py` | **Limits and trade-offs** | "How many cycles, what throughput, where does it break?" | 4-panel matplotlib figure + numerical summary in the terminal |
| `animation.py` | **Visualisation** cycle after cycle | "What happens at each clock edge?" | Animation of the values flowing through the block |

### Rules
- **Reference model**: every hardware result is compared with an independent computation — NumPy for the results, and
  a Python reference model written from the specification for cycle-by-cycle comparisons.
- **Dataflow variants**: for the blocks that exist in two versions (`pe`, `pe_array`, `data_feeder`, `output_drain`),
  the three files cover **both variants, OS and WS**, so that `benchmark.py` can compare them directly.
- **Parameters**: the tests sweep the parameters of the architecture (array size N, dimensions M/K/N, bit widths)
  rather than sticking to the default values.
- **Corner cases**: they are checked in `test.py`: extreme int8 values (-128, 127), overflow, dimensions that are not
  multiples of N, K = 1, reset during operation.
- **Exact timing**: when the specification announces a latency, the test checks that the result arrives **at that very
  cycle**, neither before nor after.
- `test.py` must pass before `benchmark.py` or `animation.py` are run.
- **Benchmarks**: we measure the behaviour (cycles, throughput, utilisation, numerical limits), not the hardware cost.
  Fixed colours: OS in blue, WS in orange.
- **Animations**: at least 30 frames per second, at a constant speed. All the displayed values come from the py4hw
  simulation, computed entirely before the animation.

## Shared tools

| File | Contents |
|---|---|
| [`common.py`](common.py) | `Bench`: minimal py4hw system around a block (`step()` advances by one edge; `observe()` reads the outputs **during** the cycle, useful for combinational outputs); `to_c2` / `from_c2` / `wrap` (two's complement); `pack` / `unpack` (lane buses); `load_test_module(block)` to reuse the testbenches and models of another block; `run_tests()`; `setup_matplotlib_fonts()`. |
| [`layout.py`](layout.py) | Data layout in memory (D6), on the host software side: `w_image`, `x_image`, `y_addresses`, `y_from_words`. |
| [`viz.py`](viz.py) | Palette and style of the figures; fast rendering engine for the animations: tokens, cells and digits are paths grouped into collections (`Layer`, `TextLayer`, `TokenLayers`), redrawn with blitting. |

The `test.py` file of another block is loaded with `load_test_module('<block>')`: a plain `import test` would pick up
the `test` module of the Python standard library.

## Testbenches

### Unit blocks
| Folder | Block under test | Status |
|---|---|---|
| `pe/` | A Processing Element (MAC, pass-through registers, accumulator clear, weight loading) | ✅ |
| `pe_array/` | N×N array of PEs, without the data_feeder (the streams are injected by hand according to the contract of the specification) | ✅ |
| `data_feeder/` | Alignment, masking and staircase skewing of the W and X streams; data_feeder → pe_array chain | ✅ |
| `output_drain/` | Draining (OS), deskewing and accumulation over the K tiles (WS); end-to-end compute core | ✅ |
| `post_processing/` | Bias → ReLU → requantization → saturation; complete chain up to the write of Y, 2-layer MLP | ✅ |
| `weight_buffer/` | Weight SRAM; layout of W read back at the scheduler addresses | ✅ |
| `activation_buffer/` | Activation SRAM (IN / OUT banks); complete datapath with its memories, driven as the host would; two chained layers | ✅ |
| `controller/` | FSM identical cycle by cycle to the reference scheduler; complete NPU without `host_interface`, driven by go / done; OS / WS decision map | ✅ |
| `host_interface/` | Bus decoding, registers, go pulse, DONE / irq, read latency; cost of the bus on the complete chip | ✅ |

### Assemblies
| Folder | Contents | Status |
|---|---|---|
| `compute_core/` | Compute core: `data_feeder` + `pe_array` + `output_drain` | placeholder files: the core is tested and measured in `output_drain/` |
| `npu_top/` | Complete chip driven only through its bus: the host writes W, X and b, starts GO, waits for DONE and reads Y back | ✅ |

`dma/` has no testbench: this block is planned for v2.

## Reference scheduler of the controller

[`output_drain/test.py`](output_drain/test.py) contains `plan_os` and `plan_ws`: for a given GEMM, they produce cycle by
cycle what the FSM of the `controller` must emit (words read from SRAM, masks, `clear` / `load`, drain commands or
sideband, tags, phase), according to [`src/controller/interface.md`](../src/controller/interface.md).
`core(variant, post=False, mem=False, ctrl=False)` and `run_core(bench, plan)` run the complete compute core with this
plan (with `post=True`, the `post_processing` is plugged in after it; with `mem=True`, the two buffers as well, and the
plan emits addresses instead of supplying the words; with `ctrl=True`, the real `controller` replaces the plan), and
`rebuild_y` rebuilds Y from the results that come out alone. These functions will serve as the reference for testing
the real `controller`.

## Host driver

[`activation_buffer/test.py`](activation_buffer/test.py) contains the functions that drive the complete datapath as
the host will: `datapath(variant)`, `load_wb`, `load_ab`, `read_ab`, `load_bias`, `copy_out_to_in` and `run_layer`
(loading, GEMM, reading Y back; with `log=[]`, a log of all memory accesses, cycle by cycle).
With the controller (`ctrl=True`), the GEMM is started by `run_go`: a `go` pulse with the dimensions, then waiting for
`done`.

## Reference host software

[`npu_top/test.py`](npu_top/test.py) contains the class `Npu`: it drives the complete chip only through its 32-bit bus
(`write`, `read`), as the real host processor will, with the memory map of `architecture.md` (D7) and the data layout
of `layout.py` (D6): `load_w`, `load_x`, `load_bias`, `configure`, `start`, `wait_done`, `read_y`, `copy_out_to_in`,
and `gemm`, which chains everything together for one layer. [`npu_top/animation.py`](npu_top/animation.py) provides
`trace_chip`, which records the activity of each block of the chip cycle by cycle.
