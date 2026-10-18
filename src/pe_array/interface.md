# pe_array — N×N array of PEs

This block contains only structure: it instantiates N×N `pe` and connects them to their neighbours.
It has no logic of its own. There is one variant per dataflow.

## Parameters
| Name | Default | Role |
|---|---|---|
| `N` | 4 | side of the array |
| `DW` | 8 | width of a data word |
| `AW` | 32 | width of the accumulators |

"Per-lane" vectors are packed: lane `l` occupies bits `[W·l + W−1 : W·l]`.

---

## OS variant

```
            x_north[0]  x_north[1] ...
                ↓           ↓
w_west[0] → PE(0,0) →  PE(0,1) → ...
                ↓           ↓
w_west[1] → PE(1,0) →  PE(1,1) → ...
```

### Ports
| Signal | Dir | Width | Description |
|---|---|---|---|
| `clk`, `rst_n` | in | 1 | |
| `w_west` | in | N·DW | lane i → `w_in` of PE(i,0) |
| `x_north` | in | N·DW | lane j → `x_in` of PE(0,j) |
| `clear` | in | 1 | broadcast to all PEs |
| `acc` | out | N·N·AW | accumulator of PE(i,j) at index `i·N + j` |

### Internal wiring
- `PE(i,j).w_in = PE(i,j−1).w_out`, and `PE(i,j).x_in = PE(i−1,j).x_out`.
- The east and south outputs of the edge PEs are not connected.

### Latencies and input contract
- A value presented on lane i of `w_west` at cycle t reaches PE(i,j) at cycle t + j.
- A value presented on lane j of `x_north` at cycle t reaches PE(i,j) at cycle t + i.
- **Contract**: if `W[i][k]` is presented on lane i at cycle `t0 + i + k`, and `X[k][j]` on lane j at cycle
  `t0 + j + k`, then PE(i,j) computes term k during cycle `t0 + i + j + k`.
- After a `clear`, `acc(i,j) = Σ_k W[i][k]·X[k][j]` can be read **from cycle `t0 + K + 2(N−1)` onwards**.

### Cost (N = 4)
16 PEs, i.e. 768 flip-flops, 16 multipliers and 512 output wires (`acc`).

---

## WS variant

```
              w_north[0]  w_north[1] ...   (preload)
psum = 0         ↓           ↓
x_west[0] →  PE(0,0) →  PE(0,1) → ...
                 ↓           ↓
x_west[1] →  PE(1,0) →  PE(1,1) → ...
                 ↓           ↓
            psum_south[0] psum_south[1]
```
PE(k,i) holds `W[i][k]`: row k of the array corresponds to input k, column i to output i.

### Ports
| Signal | Dir | Width | Description |
|---|---|---|---|
| `clk`, `rst_n` | in | 1 | |
| `x_west` | in | N·DW | lane k → `x_in` of PE(k,0) |
| `w_north` | in | N·DW | lane i → `w_load_in` of PE(0,i) |
| `load` | in | 1 | broadcast to all PEs |
| `psum_south` | out | N·AW | lane i = `psum_out` of PE(N−1,i) |

### Internal wiring
- `PE(k,i).x_in = PE(k,i−1).x_out`
- `PE(k,i).psum_in = PE(k−1,i).psum_out`, and 0 for row k = 0
- `PE(k,i).w_load_in = PE(k−1,i).w_load_out`, and `w_north[i]` for row k = 0

### Preload
- It lasts N cycles with `load = 1`. The value presented at the n-th cycle (n = 0..N−1) ends up in row `N−1−n`.
- The **row of weights that goes to the bottom must therefore be presented first**, i.e. the largest value of k.

### Latencies and input contract
- **Contract**: if `X[k][j]` is presented on lane k at cycle `t0 + j + k`, then lane i of `psum_south` carries
  `Σ_k W[i][k]·X[k][j]` (sum over the N rows of the tile) **at cycle `t0 + j + i + N`**.
- The output is therefore skewed in a staircase (lane i is i cycles late), and `output_drain` squares it up again.

### Cost (N = 4)
16 PEs, i.e. 768 flip-flops, 16 multipliers and 128 output wires (`psum_south`).

---

## Implementation

- File: [`pe_array.py`](pe_array.py) — classes `PE_Array_OS` and `PE_Array_WS`.
- Structural block: it instantiates N×N PEs and wires them; the lane buses are split with `Range` and reassembled with
  `ConcatenateLSBF`. In WS, row 0 receives a constant 0 as its partial sum.
- The attribute `pes[r][c]` gives access to each PE (used by the testbenches to observe the array).

| Constructor | Ports, in order |
|---|---|
| `PE_Array_OS(parent, name, …, N=4)` | `rst_n, w_west, x_north, clear, acc` |
| `PE_Array_WS(parent, name, …, N=4)` | `rst_n, x_west, w_north, load, psum_south` |

**Verification** ([`test/pe_array/`](../../test/pe_array/)): wiring compared cycle by cycle with an array of PE
reference models; the contracts of this document checked for N = 1, 2, 3, 4, 8 — result correct **and** available
exactly at the announced cycle (not one cycle earlier); WS preload; two successive tiles; minimum gap before reloading.

**Measured**: a tile takes L + 2N − 2 cycles in OS and L + 3N − 2 in WS (preload included) for a stream of length L.
In WS, **at least 2N − 2 cycles** are needed between the last column of X (lane 0) and the next `load`: with one cycle
less, the last column of Y is wrong.
