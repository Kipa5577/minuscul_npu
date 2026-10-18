"""
Animation of npu_top: the complete chip during one layer, seen as a chip floorplan.

The host software drives the chip through its bus (test/npu_top/test.py) for one layer Y = requant(ReLU(W·X + b)),
8×6×8 GEMM: loading W, X and b, configuration, GO, waiting for DONE, reading Y back.

  - on the left, the host and the log of the bus transactions;
  - in the centre, the memories word by word: the cells fill up when the host writes, light up when the core
    reads, and the OUT bank fills up with the words of Y;
  - on the right, the 4×4 array: the values enter through its edges, each MAC lights up, and each cell shows its
    accumulator (OS) or its weight (WS); then output_drain and post_processing;
  - at the top, the state of the controller; at the bottom, the strip of the phases and the counters.
The data packets travel along the links. All the values come from the py4hw simulation.

The speed depends on the phase (loading and waiting are sped up). Rendering ≥ 30 frames/s (see test/viz.py).
This file also provides trace_chip(), reused by the animation of the host_interface.

Run: python test/npu_top/animation.py [--variant OS|WS] [--save file.gif] [--fps 30]
"""

import sys
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import from_c2, unpack, load_test_module
from viz import (SURFACE, TEXT, TEXT_2, CELL, C_OS, C_WS, C_W, C_X,
                 ease, box_path, Layer, TextLayer, TokenLayers, title_axes)
import layout

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

T = load_test_module('npu_top')
WA = load_test_module('weight_buffer', 'animation.py')
N, IW = 4, 6
C_Y = '#1baf7a'
C_P = '#eda100'
C_CFG = '#52514e'
C_B = '#9085e9'
BACKGROUND = '#f4f3ef'
BORDER = '#dcdbd5'
WIRE = '#e2e1db'
DIMS = (8, 6, 8)
SPEED = {'loading': 18, 'configuration': 3, 'compute': 6, 'readback': 9}
PHASES = [('loading', '#9085e9'), ('configuration', C_P), ('compute', '#3a2d9a'), ('readback', '#e87ba4')]


# --- Chip trace ---------------------------------------------------------------------------------------------------

def trace_chip(variant, dims=DIMS, seed=220):
    """
    Runs one layer through the bus and records, for each cycle: the phase of the driver, the bus access, the state of
    the registers and of the controller, the memory accesses, what enters the array and the activity of each PE.
    """
    M, K, Nd = dims
    rng = np.random.default_rng(seed)
    nz = [v for v in range(-9, 10) if v]               # non-zero values: every MAC is visible
    W, X = rng.choice(nz, (M, K)), rng.choice(nz, (K, Nd))
    bias = list(rng.integers(-20, 20, M))
    npu = T.Npu(variant)
    b, d = npu.b, npu.b.dut
    h, c = d.host, d.controller
    wb, ab, pp = d.weight_buffer, d.activation_buffer, d.post_processing
    trace, phase = [], ['loading']

    def pe_active(pe):
        if variant == 'OS':
            return bool(pe.w_in.get() and pe.x_in.get())
        return bool(pe.x_in.get() and pe.w_r)          # a MAC can take place during the 1st reload cycle

    def step(**v):
        b.set(**v)
        b.sim.propagateAll()
        host_busy = h._is_busy()
        free = not wb.busy.get()
        if variant == 'OS':
            values = [[from_c2(pe.acc.get(), 32) for pe in row] for row in d.pe_array.pes]
        else:
            values = [[from_c2(pe.w_r, 8) for pe in row] for row in d.pe_array.pes]
        trace.append(dict(
            phase=phase[0],
            bus_we=h.bus_we.get(), bus_re=h.bus_re.get(), addr=h.bus_addr.get(), wdata=h.bus_wdata.get(),
            rdata=h.bus_rdata.get(), go=h.go.get(), busy=c.busy.get(), irq=h.irq.get(), host_busy=host_busy,
            state=c.state, regs=dict(h.regs), relu=h.relu_r, cycles=c.cycles.get(),
            t_out=c.t_out, t_mid=c.t_mid, cnt=c.cnt,
            # memory accesses of the host (accepted) and of the core
            wb_hw=wb.host_addr.get() if free and wb.host_we.get() else None,
            ab_hw=ab.host_addr.get() if free and ab.host_we.get() else None,
            ab_hr=ab.host_addr.get() if free and ab.host_re.get() else None,
            bias_hw=pp.bias_host_addr.get() if free and pp.bias_host_we.get() else None,
            w_rd=wb.busy.get() and wb.core_re.get(), x_rd=ab.busy.get() and ab.core_rd_re.get(),
            w_addr=wb.core_addr.get(), x_addr=ab.core_rd_addr.get(),
            y_wr=ab.busy.get() and ab.core_wr_we.get(), y_addr=ab.core_wr_addr.get(),
            # datapath
            west=unpack(d.data_feeder.w_arr.get() if variant == 'OS' else d.data_feeder.x_arr.get(), N, 8),
            north=unpack(d.data_feeder.x_arr.get() if variant == 'OS' else d.data_feeder.w_arr.get(), N, 8),
            feeder=bool(d.data_feeder.w_arr.get() or d.data_feeder.x_arr.get()),
            pes=[[pe_active(pe) for pe in row] for row in d.pe_array.pes], values=values,
            drain=d.output_drain.res_valid.get(), res=unpack(d.output_drain.res.get(), N, 32),
            post=pp.y_we.get(), y=unpack(pp.y_wdata.get(), N, 8)))
        b.sim.clk(1)
        b.cycle += 1

    b.step = step
    npu.load_w(W)
    npu.load_x(X)
    npu.load_bias(bias)
    phase[0] = 'configuration'
    npu.configure(M, K, Nd, 5, 2)
    npu.start(True)
    phase[0] = 'compute'
    npu.wait_done()
    npu.read(T.CYCLES)
    phase[0] = 'readback'
    Y = npu.read_y(M, Nd)
    assert (Y == T.PP.reference(W, X, bias + [0] * 64, 5, 2, 1)).all(), variant
    return trace


def target(addr):
    return {0: 'REG', 1: 'WEIGHT', 2: 'ACT_IN', 3: 'ACT_OUT', 4: 'BIAS'}.get(addr >> 10, '—')


REG_NAMES = ['ID', 'CTRL', 'STATUS', 'DIM_M', 'DIM_K', 'DIM_N', 'REQ_MULT', 'REQ_SHIFT', 'CYCLES']


def address_name(addr):
    cb = target(addr)
    if cb == 'REG':
        r = addr & 0xF
        return REG_NAMES[r] if r < len(REG_NAMES) else f'REG[{r}]'
    base = {'WEIGHT': 0x400, 'ACT_IN': 0x800, 'ACT_OUT': 0xC00, 'BIAS': 0x1000}.get(cb)
    return f'{cb}[{addr - base}]' if base is not None else f'0x{addr:04X}'


# --- Geometry of the chip floorplan -------------------------------------------------------------------------------------

def path(*pts):
    """Polyline: returns a function u ∈ [0, 1] → point, at constant speed along the path."""
    pts = np.array(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate(([0], np.cumsum(seg)))

    def at(u):
        dist = min(max(u, 0.0), 1.0) * cum[-1]
        i = min(np.searchsorted(cum, dist, side='right') - 1, len(seg) - 1)
        f = (dist - cum[i]) / seg[i] if seg[i] else 0
        return pts[i] + (pts[i + 1] - pts[i]) * f

    at.pts = pts
    return at


class Plan:
    """Chip floorplan: positions of the blocks, of the memory maps and of the links."""

    def __init__(self, variant):
        M, K, Nd = DIMS
        T_ = layout.tiles
        self.variant = variant
        # memory maps: (rows, columns) of the useful words
        self.dim_w = (T_(M), K)
        self.dim_in = (T_(Nd), K) if variant == 'OS' else (T_(K), Nd)
        self.dim_out = (T_(Nd), M) if variant == 'OS' else (T_(M), Nd)
        self.cs = 1.0                                                  # pitch of the memory cells
        self.x_mem = 11.1
        self.y_w, self.y_in, self.y_out = 4.1, 8.4, 12.2
        self.pe0 = (22.4, 5.2)                                          # centre of the PE(0,0) cell
        self.pe_pitch = 1.25
        self.blocks = {  # name: (x0, y0, x1, y1)
            'host': (0.4, 2.6, 6.6, 14.2),
            'host_interface': (7.4, 2.6, 9.2, 14.2),
            'weight_buffer': (10.2, 2.6, 20.0, 5.9),
            'activation_buffer': (10.2, 6.6, 20.0, 14.2),
            'data_feeder': (20.6, 3.0, 21.5, 10.4),
            'pe_array': (21.7, 3.3, 27.6, 10.6),
            'output_drain': (28.3, 5.0, 31.5, 7.0),
            'post_processing': (28.3, 9.0, 31.5, 11.4),
            'controller': (10.2, 0.0, 31.5, 1.6),
        }
        y_mid_w = self.y_w + (self.dim_w[0] - 1) * self.cs / 2
        y_mid_in = self.y_in + (self.dim_in[0] - 1) * self.cs / 2
        y_mid_out = self.y_out + (self.dim_out[0] - 1) * self.cs / 2
        x_end = lambda dims: self.x_mem + (dims[1] - 0.5) * self.cs + 0.2
        pe_y = self.pe0[1] + 1.5 * self.pe_pitch
        self.links = {
            'bus': path((6.6, 8.4), (7.4, 8.4)),
            'hi_wb': path((9.2, 4.4), (10.2, 4.4), (self.x_mem - 0.5, y_mid_w)),
            'hi_in': path((9.2, 8.4), (10.2, 8.4), (self.x_mem - 0.5, y_mid_in)),
            'out_hi': path((self.x_mem - 0.5, y_mid_out), (10.2, 12.6), (9.2, 12.6)),
            'hi_ctrl': path((8.3, 2.6), (8.3, 0.8), (10.2, 0.8)),
            'hi_bias': path((8.3, 14.2), (8.3, 15.1), (29.9, 15.1), (29.9, 11.4)),
            'wb_df': path((x_end(self.dim_w), y_mid_w), (19.4, y_mid_w), (20.6, y_mid_w + 0.4)),
            'in_df': path((x_end(self.dim_in), y_mid_in), (19.4, y_mid_in), (20.6, 8.6)),
            'df_pe': path((21.5, pe_y), (21.7, pe_y)),
            'pe_od': path((27.6, 6.0), (28.3, 6.0)),
            'od_pp': path((29.9, 7.0), (29.9, 9.0)),
            'pp_out': path((28.3, 10.8), (27.9, 10.8), (27.9, 13.4), (20.0, 13.4), (x_end(self.dim_out), y_mid_out)),
        }


# --- Animation ----------------------------------------------------------------------------------------------------------

def blend(c1, c2, a):
    r1, r2 = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple(r1 + (r2 - r1) * min(1.0, max(0.0, a)))


class Chip:
    def __init__(self, ax, axt, variant, trace):
        self.ax, self.axt, self.variant, self.tr = ax, axt, variant, trace
        self.color = C_OS if variant == 'OS' else C_WS
        P = self.P = Plan(variant)
        ax.set_xlim(0, 32)
        ax.set_ylim(15.8, -0.4)
        ax.set_aspect('equal')
        ax.axis('off')
        # background: blocks and links (static)
        for name, (x0, y0, x1, y1) in P.blocks.items():
            v, _ = box_path(x1 - x0, 0.25, y1 - y0)
            ax.add_patch(plt.Polygon(v + ((x0 + x1) / 2, (y0 + y1) / 2), closed=True, fc=BACKGROUND, ec=BORDER, lw=1.2,
                                     zorder=0))
            if name not in ('controller', 'data_feeder'):
                ax.text(x0 + 0.25, y0 + 0.42, name, fontsize=8.5, fontweight='bold', color=TEXT_2, va='center')
        ax.text(21.05, 2.75, 'data_\nfeeder', fontsize=7, fontweight='bold', color=TEXT_2, ha='center', va='center')
        for link in P.links.values():
            ax.plot(link.pts[:, 0], link.pts[:, 1], color=WIRE, lw=3.2, solid_capstyle='round',
                    solid_joinstyle='round', zorder=0.5)
        ax.text(P.x_mem - 0.4, P.y_in - 0.95, 'IN bank: X ' + ('{j_t, k}' if variant == 'OS' else '{k_t, j}'),
                fontsize=7.5, color=TEXT_2, va='center')
        ax.text(P.x_mem - 0.4, P.y_out - 0.95, 'OUT bank: Y ' + ('{j_t, i}' if variant == 'OS' else '{i_t, j}'),
                fontsize=7.5, color=TEXT_2, va='center')
        # dynamic layers
        self.cells = Layer(ax, 2, lw=1.0, edge='white')
        self.glow = Layer(ax, 3, lw=2.4, edge=self.color)
        self.vals = TextLayer(ax, 4)
        self.tokens = TokenLayers(ax, 6)
        self.txt = TextLayer(ax, 8)
        # precomputations: cumulative state of the memories, counters, bus log
        self.states = WA.cumulative_states(trace, self._update)
        self.macs = np.cumsum([sum(map(sum, t['pes'])) for t in trace])
        self.nb_y = np.cumsum([bool(t['y_wr']) for t in trace])
        self.bus_log = self._bus_log()
        self._strip()

    # --- cumulative state ----------------------------------------------------------------------

    @staticmethod
    def _update(state, t):
        w, xin, out, reads_w, reads_x = state or (set(), set(), set(), {}, {})
        if t['wb_hw'] is not None:
            w.add(t['wb_hw'])
        if t['ab_hw'] is not None and not (t['ab_hw'] >> 10):
            xin.add(t['ab_hw'])
        if t['w_rd']:
            reads_w[t['w_addr']] = reads_w.get(t['w_addr'], 0) + 1
        if t['x_rd']:
            reads_x[t['x_addr']] = reads_x.get(t['x_addr'], 0) + 1
        if t['y_wr']:
            out.add(t['y_addr'])
        return w, xin, out, reads_w, reads_x

    def _bus_log(self):
        lines, res = [], []
        for k, t in enumerate(self.tr):
            if t['bus_we']:
                lines.append(('W', f'{address_name(t["addr"])} ← {t["wdata"] & 0xFFFFFFFF:08X}', k))
            elif t['bus_re']:
                lines.append(('R', f'{address_name(t["addr"])}', k))
            res.append(list(lines[-12:]))
        return res

    def _strip(self):
        ax, tr = self.axt, self.tr
        n = len(tr)
        ax.set_xlim(-0.5, n + 0.5)
        ax.set_ylim(1.7, -0.9)
        ax.axis('off')
        self.segments = []
        start = 0
        for k in range(1, n + 1):
            if k == n or tr[k]['phase'] != tr[start]['phase']:
                col = dict(PHASES)[tr[start]['phase']]
                ax.add_patch(plt.Rectangle((start - 0.5, 0), k - start, 0.55, color=col, alpha=0.22, lw=0))
                self.segments.append((start, k, col))
                ax.text((start + k) / 2 - 0.5, -0.45, tr[start]['phase'], ha='center', fontsize=8.5, color=TEXT_2)
                start = k
        self.prog = Layer(ax, 2)
        self.cursor = ax.axvline(0, ymin=0.15, ymax=0.62, color=TEXT, lw=1.4, animated=True)
        self.ftxt = TextLayer(ax, 3)

    # --- drawing -------------------------------------------------------------------------------

    def packet(self, link, s, letter, colour):
        u = ease(s / 0.9)
        x, y = self.P.links[link](u)
        self.tokens.token(x, y, letter, colour, size=0.52, alpha=1 - max(0.0, u - 0.85) / 0.15)

    def mem_map(self, x0, y0, dims, filled, reads, colour, active, s, dark):
        P = self.P
        cv, cc = box_path(P.cs * 0.86, 0.06)
        for r in range(dims[0]):
            for col in range(dims[1]):
                a = (r << IW) | col
                x, y = x0 + col * P.cs, y0 + r * P.cs
                n = reads.get(a, 0)
                if a in filled:
                    fc = blend(colour, dark, min(1.0, n / 4)) if n else blend('#ffffff', colour, 0.55)
                else:
                    fc = CELL
                self.cells.add(cv + (x, y), cc, fc)
                if a == active:
                    self.glow.add(cv * 1.25 + (x, y), cc, 'none', ec=TEXT, ec_alpha=1 - 0.6 * s)

    def draw(self, c, s):
        for l in (self.cells, self.glow, self.vals, self.tokens, self.txt, self.prog, self.ftxt):
            l.reset()
        tr, P = self.tr, self.P
        live = c < len(tr)
        k = min(c, len(tr) - 1)
        t = tr[k]
        w_ch, in_ch, out_ch, reads_w, reads_x = self.states[k] or (set(), set(), set(), {}, {})

        # memories: the write takes effect at the edge (middle of the cycle on screen)
        after_edge = live and s > 0.5
        wb_hw = t['wb_hw'] if live else None
        ab_hw = t['ab_hw'] if live and t['ab_hw'] is not None and not (t['ab_hw'] >> 10) else None
        ab_hr = t['ab_hr'] & 0x3FF if live and t['ab_hr'] is not None and t['ab_hr'] >> 10 else None
        self.mem_map(P.x_mem, P.y_w, P.dim_w, w_ch | ({wb_hw} if after_edge and wb_hw is not None else set()), reads_w, C_W,
                   t['w_addr'] if live and t['w_rd'] else wb_hw, s, '#241a6b')
        self.mem_map(P.x_mem, P.y_in, P.dim_in, in_ch | ({ab_hw} if after_edge and ab_hw is not None else set()), reads_x, C_X,
                   t['x_addr'] if live and t['x_rd'] else ab_hw, s, '#08402c')
        self.mem_map(P.x_mem, P.y_out, P.dim_out, out_ch | ({t['y_addr']} if after_edge and t['y_wr'] else set()), {}, C_Y,
                   t['y_addr'] if live and t['y_wr'] else ab_hr, s, '#08402c')

        # array: value of each PE, lit MACs, incoming values on the edges
        pv, pc = box_path(P.pe_pitch * 0.84, 0.1)
        for i in range(N):
            for j in range(N):
                x, y = P.pe0[0] + j * P.pe_pitch, P.pe0[1] + i * P.pe_pitch
                on = live and t['pes'][i][j]
                self.cells.add(pv + (x, y), pc, blend('#ffffff', self.color, 0.3) if on else '#ffffff')
                if on:
                    self.glow.add(pv * 1.08 + (x, y), pc, 'none', ec_alpha=1 - 0.7 * s)
                v = t['values'][i][j]
                if v:
                    self.vals.text(x, y + 0.28, ('Σ ' if self.variant == 'OS' else 'w ') + str(v), TEXT, pt=7.5)
        if live:
            u = ease(s / 0.5)
            cw, cn = (C_W, C_X) if self.variant == 'OS' else (C_X, C_W)
            for l in range(N):
                if t['west'][l]:
                    self.tokens.token(P.pe0[0] - 1.0 + 0.55 * u, P.pe0[1] + l * P.pe_pitch - 0.18, t['west'][l], cw,
                                      size=0.48)
                if t['north'][l]:
                    self.tokens.token(P.pe0[0] + l * P.pe_pitch, P.pe0[1] - 1.05 + 0.55 * u, t['north'][l], cn,
                                      size=0.48)

        # packets on the links, and blocks that are working
        if live:
            if t['bus_we'] or t['bus_re']:
                self.packet('bus', s, 'W' if t['bus_we'] else 'R', C_CFG)
            if wb_hw is not None:
                self.packet('hi_wb', s, 'W', C_W)
            if ab_hw is not None:
                self.packet('hi_in', s, 'X', C_X)
            if t['bias_hw'] is not None:
                self.packet('hi_bias', s, 'b', C_B)
            if t['bus_we'] and target(t['addr']) == 'REG':
                self.packet('hi_ctrl', s, 'GO' if (t['addr'] & 0xF) == 1 and t['wdata'] & 1 else 'cfg', C_P)
            if ab_hr is not None:
                self.packet('out_hi', s, 'Y', C_Y)
            if t['w_rd']:
                self.packet('wb_df', s, 'W', C_W)
            if t['x_rd']:
                self.packet('in_df', s, 'X', C_X)
            if t['drain']:
                self.packet('pe_od', s, 'Σ', C_P)
                self.packet('od_pp', s, 'Σ', C_P)
            if t['post']:
                self.packet('pp_out', s, 'Y', C_Y)
            for name, on in (('data_feeder', t['feeder']), ('output_drain', t['drain']),
                            ('post_processing', t['post']),
                            ('host_interface', t['bus_we'] or t['bus_re'] or t['go'])):
                if on:
                    x0, y0, x1, y1 = P.blocks[name]
                    v, cds = box_path(x1 - x0, 0.25, y1 - y0)
                    self.glow.add(v + ((x0 + x1) / 2, (y0 + y1) / 2), cds, 'none', ec_alpha=0.9 - 0.5 * s)
            if t['drain']:
                self.txt.text(29.9, 6.25, '  '.join(str(v) for v in t['res']), TEXT, bold=False, pt=7)
            if t['post']:
                self.txt.text(29.9, 10.55, '  '.join(str(v) for v in t['y']), TEXT, bold=False, pt=7)

        # controller: state badges
        states = ['IDLE', 'CLEAR', 'STREAM', 'FLUSH', 'DRAIN', 'TAIL', 'DONE'] if self.variant == 'OS' else \
            ['IDLE', 'PRELOAD', 'STREAM', 'FLUSH', 'TAIL', 'DONE']
        ev, ec = box_path(2.05, 0.2, 0.75)
        for n_, e in enumerate(states):
            x = 13.1 + n_ * 2.25
            on = e == t['state']
            self.cells.add(ev + (x, 0.8), ec, self.color if on else '#ffffff')
            self.vals.text(x, 0.8, e, 'white' if on else TEXT_2, pt=7)
        self.txt.text(11.15, 0.8, 'controller', TEXT_2, pt=8)
        if t['busy']:
            loops = ('i_t = {}   j_t = {}' if self.variant == 'OS' else 'i_t = {}   k_t = {}').format(t['t_out'],
                                                                                                    t['t_mid'])
            self.txt.text(29.9, 0.8, loops, self.color, pt=8)

        # host: log of the bus transactions (the most recent at the bottom) and lamps
        self.txt.text(3.5, 3.15, '32-bit bus', TEXT_2, bold=False, pt=7.5)
        for n_, (op, text, kk) in enumerate(self.bus_log[k]):
            age = k - kk
            colour = TEXT if age == 0 else TEXT_2
            self.txt.text(3.5, 3.85 + n_ * 0.62, f'{op}  {text}', colour, bold=age == 0, alpha=max(0.35, 1 - age / 30),
                          pt=7)
        lv, lc = box_path(0.42, 0.2)
        for n_, (lab, on, col) in enumerate((('GO', t['go'], self.color), ('BUSY', t['host_busy'], C_P),
                                             ('irq', t['irq'], C_Y))):
            x = 1.5 + n_ * 2.0
            self.cells.add(lv + (x, 12.9), lc, col if on and live else CELL)
            self.vals.text(x, 13.55, lab, TEXT_2, bold=False, pt=6.5)

        # strip: progress and counters
        M, K, Nd = DIMS
        n = len(tr)
        width = min(c + s, n)
        for a, b, col in self.segments:
            end = min(b, width)
            if end > a:
                self.prog.add(np.array([[a - 0.5, 0], [end - 0.5, 0], [end - 0.5, 0.55], [a - 0.5, 0.55], [a - 0.5, 0]]),
                              [1, 2, 2, 2, 79], col, 0.9)
        self.cursor.set_xdata([width - 0.5])
        self.ftxt.text(n / 2, 1.3, f'cycle {k}   ·   MACs {self.macs[k]} / {M * K * Nd}   ·   words of Y written '
                       f'{self.nb_y[k]} / {P.dim_out[0] * P.dim_out[1]}   ·   CYCLES register = {t["cycles"]}',
                       TEXT, bold=False, pt=9)
        return [self.cells.commit(), self.glow.commit(), self.vals.commit()] + self.tokens.commit() + \
            [self.txt.commit(), self.prog.commit(), self.cursor, self.ftxt.commit()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=('OS', 'WS'), default='OS', help='variant of the compute core')
    ap.add_argument('--save', help='saves the animation (gif) instead of showing it')
    ap.add_argument('--fps', type=int, default=30, help='frames per second')
    args = ap.parse_args()

    trace = trace_chip(args.variant)
    plt.rcParams['figure.facecolor'] = SURFACE
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes((0.01, 0.13, 0.98, 0.77))
    axt = fig.add_axes((0.06, 0.015, 0.88, 0.1))
    chip = Chip(ax, axt, args.variant, trace)
    color = C_OS if args.variant == 'OS' else C_WS
    name = 'output-stationary' if args.variant == 'OS' else 'weight-stationary'
    title = title_axes(fig, (0, 0.94, 1, 0.06)).text(0.5, 0.75, '', ha='center', va='top', fontsize=15,
                                                      fontweight='bold', color=TEXT, animated=True)
    M, K, Nd = DIMS
    fig.text(0.5, 0.922, f'{name}  ·  layer Y = requant(ReLU(W·X + b)), {M}×{K}×{Nd} GEMM, driven through the '
             '32-bit bus  ·  loading and waiting sped up', ha='center', fontsize=10, color=color)
    tl = WA.Timeline(trace, SPEED)

    def label(t):
        c, _ = tl.locate(t)
        return f'npu_top — the complete chip  ·  {trace[min(c, len(trace) - 1)]["phase"]}'

    return WA.run_animation(fig, [chip], [tl], title, label, args.fps, args.save)


if __name__ == '__main__':
    main()
