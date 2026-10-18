"""
Animation of the activation_buffer: the complete journey of a layer through the two banks.

The complete datapath (with its memories) runs an 8×12×8 GEMM, driven as the host will drive it:
  host → IN bank (X, word by word) → compute core (reads of X) → OUT bank (writes of the words of Y)
  → host (reading Y back).
Each cell is a 32-bit word (4 int8 values); the digit shows how many times the core read or wrote it.

  - OS (top)   : X is stored in words {j_t, k}, Y is written in words {j_t, i}
  - WS (bottom): X is stored in words {k_t, j}, Y is written in words {i_t, j}
In both cases, Y comes out in the X format of the variant: it can be used as is as the input of the next layer.

The speed depends on the phase (loading and reading back sped up). Rendering ≥ 30 frames/s (see test/viz.py).

Run: python test/activation_buffer/animation.py [--save file.gif] [--fps 30]
"""

import sys
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_test_module
from viz import SURFACE, TEXT, TEXT_2, C_OS, C_WS, C_X, ease, TokenLayers, title_axes
import layout

import matplotlib.pyplot as plt

WA = load_test_module('weight_buffer', 'animation.py')
C_Y = '#1baf7a'
VOL = WA.VOL


EMPTY = (set(), {}, set(), {}, 0)


def update_ab(state, e):
    loaded_in, reads_in, loaded_out, writes_out, read_back = state or (set(), {}, set(), {}, 0)
    if e['ab_w'] is not None and e['ab_w'][0] == 0:
        loaded_in.add(e['ab_w'][1])
    if e['x_rd'] is not None:
        reads_in[e['x_rd']] = reads_in.get(e['x_rd'], 0) + 1
    if e['y_wr'] is not None:
        loaded_out.add(e['y_wr'][0])
        writes_out[e['y_wr'][0]] = writes_out.get(e['y_wr'][0], 0) + 1
    return loaded_in, reads_in, loaded_out, writes_out, read_back + (e['ab_r'] is not None)


class Panel:
    def __init__(self, ax, variant, color, dims, log):
        self.ax, self.variant, self.log = ax, variant, log
        Mr, K, Nd = dims
        T = layout.tiles
        if variant == 'OS':
            xin, yout = (T(Nd), K, 'j_t', 'k', '{j_t, k}'), (T(Nd), Mr, 'j_t', 'i', '{j_t, i}')
        else:
            xin, yout = (T(K), Nd, 'k_t', 'j', '{k_t, j}'), (T(Mr), Nd, 'i_t', 'j', '{i_t, j}')
        rows = max(xin[0], yout[0])
        self.x_host_in = -3.6
        self.in_grid = WA.MemGrid(ax, 0, 0, xin[0], xin[1], C_X, f'IN bank: X in words {xin[4]}', xin[2], xin[3],
                                  dark='#08402c')
        self.x_core = xin[1] + 1.9
        x_out = self.x_core + 2.9
        self.out_grid = WA.MemGrid(ax, x_out, 0, yout[0], yout[1], C_Y, f'OUT bank: Y in words {yout[4]}',
                                   yout[2], yout[3], dark='#08402c')
        self.x_host_out = x_out + yout[1] + 1.6
        ax.set_xlim(self.x_host_in - 1.1, self.x_host_out + 1.4)
        ax.set_ylim(rows + 0.9, -2.4)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.text(self.x_host_in - 1.0, -2.05, 'output-stationary' if variant == 'OS' else 'weight-stationary',
                fontsize=12, fontweight='bold', color=color, va='center')
        for x, lab in ((self.x_host_in, 'host'), (self.x_core, 'compute\ncore'), (self.x_host_out, 'host')):
            ax.text(x, rows / 2 - 0.5, lab, ha='center', va='center', fontsize=9, color=TEXT_2)
        self.tokens = TokenLayers(ax, 6)
        self.info = ax.text(self.x_host_out + 1.3, -2.05, '', ha='right', va='center', fontsize=9, color=TEXT,
                            animated=True)
        self.states = WA.cumulative_states(log, update_ab)
        final = self.states[-1] or EMPTY
        self.vmax = max([1] + list(final[1].values()))

    def draw(self, c, s):
        self.tokens.reset()
        log = self.log
        cc = min(c, len(log))
        st = self.states[cc] or EMPTY
        loaded_in, reads_in, loaded_out, writes_out, read_back = set(st[0]), dict(st[1]), set(st[2]), dict(st[3]), st[4]
        act_in, act_out = set(), set()
        e = log[c] if c < len(log) else None
        if e is not None:
            if e['x_rd'] is not None:
                act_in.add(e['x_rd'])
            if e['y_wr'] is not None:
                act_out.add(e['y_wr'][0])
                if s > 0.5:
                    loaded_out.add(e['y_wr'][0])
                    writes_out[e['y_wr'][0]] = writes_out.get(e['y_wr'][0], 0) + 1
            if e['ab_w'] is not None and e['ab_w'][0] == 0 and s > 0.8:
                loaded_in.add(e['ab_w'][1])
            if e['ab_r'] is not None:
                act_out.add(e['ab_r'][1])
        # tokens in flight (a transfer lasts VOL cycles on screen)
        for back in range(VOL):
            k = c - back
            if not 0 <= k < len(log):
                continue
            f, u = log[k], ease((back + s) / VOL)
            fade = 1 - u ** 8
            if f['ab_w'] is not None and f['ab_w'][0] == 0:
                x, y = self.in_grid.pos(f['ab_w'][1])
                self.tokens.token(self.x_host_in + (x - self.x_host_in) * u, y, 'X', C_X, size=0.55, alpha=fade)
            if f['x_rd'] is not None:
                x, y = self.in_grid.pos(f['x_rd'])
                self.tokens.token(x + (self.x_core - x) * u, y, 'X', C_X, size=0.55, alpha=fade)
            if f['y_wr'] is not None:
                x, y = self.out_grid.pos(f['y_wr'][0])
                self.tokens.token(self.x_core + (x - self.x_core) * u, y, 'Y', C_Y, size=0.55, alpha=fade)
            if f['ab_r'] is not None and f['ab_r'][0] == 1:
                x, y = self.out_grid.pos(f['ab_r'][1])
                self.tokens.token(x + (self.x_host_out - x) * u, y, 'Y', C_Y, size=0.55, alpha=fade)
        arts = self.in_grid.draw(loaded_in, reads_in, act_in, s, self.vmax)
        arts += self.out_grid.draw(loaded_out, writes_out, act_out, s, 1, digits=False)   # always 1 write
        phase = e['phase'] if e else 'finished'
        self.info.set_text(f'{phase}  ·  reads of X: {sum(reads_in.values())}  ·  words of Y written: '
                           f'{len(loaded_out)}  ·  read back by the host: {read_back}')
        return arts + self.tokens.commit() + [self.info]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='saves the animation (gif) instead of showing it')
    ap.add_argument('--fps', type=int, default=30, help='frames per second')
    args = ap.parse_args()

    log_os, dims = WA.capture('OS')
    log_ws, _ = WA.capture('WS')
    # loading W and the biases does not concern the activation_buffer: only its accesses are kept
    keep = lambda e: e['phase'] != 'loading' or (e['ab_w'] is not None)
    log_os, log_ws = [e for e in log_os if keep(e)], [e for e in log_ws if keep(e)]

    plt.rcParams['figure.facecolor'] = SURFACE
    fig = plt.figure(figsize=(15, 7.8))
    gs = fig.add_gridspec(2, 1, hspace=0.05)
    P_os = Panel(fig.add_subplot(gs[0]), 'OS', C_OS, dims, log_os)
    P_ws = Panel(fig.add_subplot(gs[1]), 'WS', C_WS, dims, log_ws)
    title = title_axes(fig, (0, 0.95, 1, 0.05)).text(0.5, 0.75, '', ha='center', va='top', fontsize=14,
                                                      fontweight='bold', color=TEXT, animated=True)
    fig.text(0.5, 0.93, f'GEMM {dims[0]}×{dims[1]}×{dims[2]}   ·   digit in a cell of the IN bank: number '
             'of reads of the word by the core (each word of Y is written only once)   ·   loading and reading back sped up', ha='center', fontsize=9,
             color=TEXT_2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.01)
    tls = [WA.Timeline(log_os), WA.Timeline(log_ws)]
    return WA.run_animation(fig, [P_os, P_ws], tls, title, lambda t: f'activation_buffer — {t:4.1f} s',
                            args.fps, args.save)


if __name__ == '__main__':
    main()
