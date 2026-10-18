# npu_top — NPU assembly

This is the only block visible from the outside. It instantiates all the others and wires them together, with no logic
of its own.

## Parameters
| Name | Default | Role |
|---|---|---|
| `DATAFLOW` | OS | variant of the compute core (OS or WS), fixed at construction time |
| `N` | 4 | size of the array (v1 requires N = 4 because of the 32-bit bus) |
| `DIM_MAX` | 64 | maximum dimension of the matrices |

## External ports
| Signal | Dir | Width | Description |
|---|---|---|---|
| `clk` | in | 1 | single clock |
| `rst_n` | in | 1 | synchronous active-low reset |
| `bus_addr` | in | 13 | |
| `bus_wdata` | in | 32 | |
| `bus_we` | in | 1 | |
| `bus_re` | in | 1 | |
| `bus_rdata` | out | 32 | latency 1 |
| `irq` | out | 1 | = `STATUS.DONE` |

## Wiring
```
host_interface ──cfg──► controller, post_processing
host_interface ◄─busy, done, cycles── controller
host_interface ──host ports──► weight_buffer, activation_buffer, post_processing (biases)
controller.busy ──► weight_buffer.busy, activation_buffer.busy, post_processing.busy

controller.w_addr/w_re ──► weight_buffer.core_*          ──core_rdata──►    data_feeder.w_rdata
controller.x_addr/x_re ──► activation_buffer.core_rd_*   ──core_rd_rdata──► data_feeder.x_rdata
controller.w_mask/x_mask/ctl ──► data_feeder

┌──────────────────── compute core (DATAFLOW variant) ────────────────────────┐
│ data_feeder.w_arr/x_arr/ctl_arr ──► pe_array                                │
│ pe_array.acc (OS) | psum_south (WS) ──► output_drain                        │
│ controller.drain_* + tag (OS) | s_* + tag (WS) ──► output_drain             │
└─────────────────────────────────────────────────────────────────────────────┘

output_drain.res/res_valid/tag_out ──► post_processing
post_processing.y_* ──► activation_buffer.core_wr_*
```

## The compute core
The `data_feeder` + `pe_array` + `output_drain` assembly is the variant boundary (D3).
Its external interfaces are common to both variants:
- as inputs: `w_rdata`, `x_rdata`, `w_mask`, `x_mask`, `ctl`, `tag`, plus the variant-specific command (`drain_*` in
  OS, `s_*` in WS);
- as outputs: `res`, `res_valid`, `tag_out`.

It is tested end to end in `test/output_drain/` (with the following blocks plugged in one after the other).

## Typical host-side sequence
1. Write W into `WEIGHT`, X into `ACT_IN` (in the format of the variant, see `ID.DATAFLOW`) and b into `BIAS`.
2. Write `DIM_M`, `DIM_K`, `DIM_N`, `REQ_MULT`, `REQ_SHIFT`.
3. Write `CTRL = GO | RELU_EN`.
4. Wait for `irq`, or poll `STATUS.DONE`.
5. Read Y from `ACT_OUT`, and `CYCLES` for the measurement.

---

## Implementation

- File: [`npu_top.py`](npu_top.py) — class `NpuTop(parent, name, rst_n, bus_addr, bus_wdata, bus_we, bus_re,
  bus_rdata, irq, dataflow='OS', DIM_MAX=64)`. Structural block: it creates the internal wires and instantiates the 9
  blocks.
- The blocks are accessible as attributes (`host`, `controller`, `weight_buffer`, `activation_buffer`, `data_feeder`,
  `pe_array`, `output_drain`, `post_processing`), which lets the testbenches observe the chip.
- The outputs of the `controller` are published in `clock()` (Moore machine): otherwise the simulator would see a false
  combinational loop `host_interface` → `go` → `controller` → `busy` → `host_interface`.

**Verification** ([`test/npu_top/`](../../test/npu_top/)): the chip is driven only through its bus, by reference host
software (class `Npu`) that follows the sequence above: ID register, end-to-end layers in both variants ("dirty"
memory, dimensions that are not multiples of 4), two chained layers, `CYCLES` and `irq`, invalid dimensions, accesses
ignored during a GEMM, default requantization, OS and WS identical.

**Measured**: a floating-point 64 → 32 → 16 MLP, quantized to int8 and run on the chip, reproduces the floating-point
computation with a correlation of 0.9999; OS is faster for small batches, WS catches up around a batch of 64.
