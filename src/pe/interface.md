# pe — Processing Element

The basic building block of the array. There is one variant per dataflow, selected at construction time by the
`DATAFLOW` parameter.
Common conventions (parameters, naming, latencies): see [architecture.md §5](../architecture.md#5-interface-conventions).

## Parameters
| Name | Default | Role |
|---|---|---|
| `DW` | 8 | width of a data word (signed int8) |
| `AW` | 32 | width of the accumulator and of the partial sums (signed int32) |

---

## OS variant (output-stationary)

```
            x_in
             ↓
 w_in  →  [ acc += w·x ]  →  w_out
             ↓
            x_out          acc (parallel read)
```

### Ports
| Signal | Dir | Width | Description |
|---|---|---|---|
| `clk`, `rst_n` | in | 1 | clock, synchronous active-low reset |
| `w_in` | in | DW | weight coming from the west |
| `x_in` | in | DW | activation coming from the north |
| `clear` | in | 1 | synchronous reset of the accumulator to zero |
| `w_out` | out | DW | `w_in` delayed by one cycle, towards the east |
| `x_out` | out | DW | `x_in` delayed by one cycle, towards the south |
| `acc` | out | AW | value of the accumulator |

### Behaviour (on each rising edge)
```
if   !rst_n : w_r ← 0 ; x_r ← 0 ; acc ← 0
else        : w_r ← w_in ; x_r ← x_in
              if clear : acc ← 0                                  (clear has priority, the product of this cycle is lost)
              else     : acc ← acc + sext32(w_in × x_in)          (signed 8×8 product → 16 bits)
w_out = w_r ; x_out = x_r
```
- Overflow wraps around modulo 2^32. It cannot happen as long as K ≤ 65,536.
- For a cycle without useful data to add nothing, it is enough to have 0 on `w_in` or on `x_in`. The `data_feeder`
  guarantees these zeros.

### Latencies
| Path | L |
|---|---|
| `w_in` → `w_out`, `x_in` → `x_out` | 1 |
| `w_in`, `x_in`, `clear` → `acc` | 1 |

### Cost
48 flip-flops (8 + 8 + 32), 1 signed 8×8 multiplier, 1 32-bit adder.
The critical path goes through the multiplier, then the adder.
Wires to the neighbours: 8 bits to the east, 8 bits to the south.

---

## WS variant (weight-stationary)

```
          psum_in   w_load_in
             ↓         ↓
 x_in  →  [ w_r stored, psum + w·x ]  →  x_out
             ↓         ↓
          psum_out  w_load_out
```

### Ports
| Signal | Dir | Width | Description |
|---|---|---|---|
| `clk`, `rst_n` | in | 1 | clock, synchronous active-low reset |
| `x_in` | in | DW | activation coming from the west |
| `psum_in` | in | AW | partial sum coming from the north |
| `w_load_in` | in | DW | weight on the preload path, from the north |
| `load` | in | 1 | shifts the preload path: `w_r ← w_load_in` |
| `x_out` | out | DW | `x_in` delayed by one cycle, towards the east |
| `psum_out` | out | AW | `psum_in + w_r·x_in`, registered, towards the south |
| `w_load_out` | out | DW | stored weight `w_r`, towards the PE below (preload path) |

### Behaviour (on each rising edge)
```
if   !rst_n : w_r ← 0 ; x_r ← 0 ; psum_r ← 0
else        : if load : w_r ← w_load_in
              x_r    ← x_in
              psum_r ← psum_in + sext32(w_r × x_in)
x_out = x_r ; psum_out = psum_r ; w_load_out = w_r
```
- During a cycle where `load = 1`, the product uses the old weight. The `controller` guarantees that no useful data
  goes through at that moment.

### Latencies
| Path | L |
|---|---|
| `x_in` → `x_out` | 1 |
| `psum_in`, `x_in` → `psum_out` | 1 |
| `w_load_in` → `w_load_out` (if `load`) | 1 |

### Cost
48 flip-flops (8 + 8 + 32), 1 signed 8×8 multiplier, 1 32-bit adder.
Wires to the neighbours: 8 bits to the east, **32 + 8 bits** to the south.

---

## OS / WS comparison
| | OS | WS |
|---|---|---|
| Flip-flops per PE | 48 | 48 |
| Wires between PEs | 16 | 48 |
| Control | `clear` | `load` |
| Reading the result | `acc` in parallel | `psum_out` stream |

---

## Implementation

- File: [`pe.py`](pe.py) — classes `PE_OS` and `PE_WS`, in behavioural py4hw.
- All outputs are registers: the state (`w_r`, `x_r`, `acc_r` or `psum_r`) is updated in `clock()` and published with
  `prepare()`, which also reduces the value modulo 2^AW.
- `DW` and `AW` are deduced from the width of the connected wires (constraint: AW ≥ 2·DW).

| Constructor | Ports, in order |
|---|---|
| `PE_OS(parent, name, …)` | `rst_n, w_in, x_in, clear, w_out, x_out, acc` |
| `PE_WS(parent, name, …)` | `rst_n, x_in, psum_in, w_load_in, load, x_out, psum_out, w_load_out` |

**Verification** ([`test/pe/`](../../test/pe/)): reset, latencies, `clear` priority, `load` using the old weight,
extreme values, overflow modulo 2^AW, column of 4 stacked WS PEs, cycle-by-cycle comparison with a reference model over
2000 random cycles, other widths (int4, int16).

**Measured**: all latencies are 1; in int8 / int32, no overflow is possible as long as K ≤ 131,071.
