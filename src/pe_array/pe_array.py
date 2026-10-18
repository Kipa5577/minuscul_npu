"""
pe_array — N×N array of PEs (specification: src/pe_array/interface.md)

The block has no logic of its own: it instantiates N×N PEs and connects them to their neighbours.
"Per-lane" vectors are packed: lane l occupies bits [W·l + W−1 : W·l].

  - PE_Array_OS: PE(i,j) accumulates Y[i][j]; W enters from the west, X from the north
  - PE_Array_WS: PE(k,i) holds W[i][k]; X enters from the west, the partial sums leave at the south
"""

from py4hw import Logic, Wire, Range, ConcatenateLSBF, Constant

from src.pe.pe import PE_OS, PE_WS


def _lanes(parent: Logic, name: str, bus: Wire, n: int) -> list:
    """Splits a packed bus into n wires (lane 0 = least significant bits)."""
    w = bus.getWidth() // n
    out = []
    for l in range(n):
        lane = parent.wire(f'{name}_{l}', w)
        Range(parent, f'{name}_{l}', bus, w * l + w - 1, w * l, lane)
        out.append(lane)
    return out


class PE_Array_OS(Logic):
    """Output-stationary array: acc of PE(i,j) at index i·N + j of the `acc` bus."""

    def __init__(self, parent, name: str, rst_n: Wire, w_west: Wire, x_north: Wire, clear: Wire,
                 acc: Wire, N: int = 4):
        super().__init__(parent, name)

        rst_n = self.addIn('rst_n', rst_n)
        w_west = self.addIn('w_west', w_west)
        x_north = self.addIn('x_north', x_north)
        clear = self.addIn('clear', clear)
        acc = self.addOut('acc', acc)

        assert w_west.getWidth() % N == 0 and x_north.getWidth() == w_west.getWidth()
        assert acc.getWidth() % (N * N) == 0
        DW = w_west.getWidth() // N
        AW = acc.getWidth() // (N * N)
        self.N = N

        # h[i][j]: weight entering PE(i,j) from the west; v[i][j]: activation entering from the north
        h = [[lane] for lane in _lanes(self, 'w_west', w_west, N)]
        v = [_lanes(self, 'x_north', x_north, N)]
        for i in range(N):
            h[i] += [self.wire(f'h_{i}_{j + 1}', DW) for j in range(N)]
            v.append([self.wire(f'v_{i + 1}_{j}', DW) for j in range(N)])

        a = [[self.wire(f'acc_{i}_{j}', AW) for j in range(N)] for i in range(N)]

        self.pes = [[PE_OS(self, f'pe_{i}_{j}', rst_n, h[i][j], v[i][j], clear,
                           h[i][j + 1], v[i + 1][j], a[i][j])
                     for j in range(N)] for i in range(N)]

        ConcatenateLSBF(self, 'acc', [a[i][j] for i in range(N) for j in range(N)], acc)


class PE_Array_WS(Logic):
    """Weight-stationary array: PE(k,i) holds W[i][k]; lane i of `psum_south` = output of column i."""

    def __init__(self, parent, name: str, rst_n: Wire, x_west: Wire, w_north: Wire, load: Wire,
                 psum_south: Wire, N: int = 4):
        super().__init__(parent, name)

        rst_n = self.addIn('rst_n', rst_n)
        x_west = self.addIn('x_west', x_west)
        w_north = self.addIn('w_north', w_north)
        load = self.addIn('load', load)
        psum_south = self.addOut('psum_south', psum_south)

        assert x_west.getWidth() % N == 0 and w_north.getWidth() == x_west.getWidth()
        assert psum_south.getWidth() % N == 0
        DW = x_west.getWidth() // N
        AW = psum_south.getWidth() // N
        self.N = N

        zero = self.wire('zero', AW)
        Constant(self, 'zero', 0, zero)

        # h[k][i]: activation entering PE(k,i) from the west
        # p[k][i]: partial sum entering from the north; wl[k][i]: weight on the preload path
        h = [[lane] for lane in _lanes(self, 'x_west', x_west, N)]
        p = [[zero] * N]
        wl = [_lanes(self, 'w_north', w_north, N)]
        for k in range(N):
            h[k] += [self.wire(f'h_{k}_{i + 1}', DW) for i in range(N)]
            p.append([self.wire(f'p_{k + 1}_{i}', AW) for i in range(N)])
            wl.append([self.wire(f'wl_{k + 1}_{i}', DW) for i in range(N)])

        self.pes = [[PE_WS(self, f'pe_{k}_{i}', rst_n, h[k][i], p[k][i], wl[k][i], load,
                           h[k][i + 1], p[k + 1][i], wl[k + 1][i])
                     for i in range(N)] for k in range(N)]

        ConcatenateLSBF(self, 'psum_south', p[N], psum_south)
