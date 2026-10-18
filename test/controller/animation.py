"""
Animation of the controller: the FSM that sequences a GEMM, cycle by cycle.

Both variants run the same 8×5×8 GEMM (only the controller is simulated). For each one:
  - the state diagram: the active state is lit up, with its progress (k, j, drained row…);
  - the tiles: OS walks through the output tiles (i_t, j_t), WS through the pairs (i_t, k_t);
  - the signals emitted during this cycle: addresses read, masks (one dot per lane), ctl, drain or sideband, tag;
at the bottom, the timing diagram of the states of both variants.

Rendering ≥ 30 frames/s (see test/viz.py).

Run: python test/controller/animation.py [--save file.gif] [--fps 30] [--cycle-ms 330]
"""

import sys
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_test_module
from viz import (SURFACE, TEXT, TEXT_2, CELL, GRID, C_OS, C_WS, ERR,
                 ease, box_path, RECT, Layer, TextLayer, title_axes)

import matplotlib.pyplot as plt

T = load_test_module('controller')
WA = load_test_module('weight_buffer', 'animation.py')
N, IW = 4, 6
DIMS = (8, 5, 8)
OK = '#1baf7a'
STATE_COLOR = {'IDLE': '#d9d8d2', 'CLEAR': '#9085e9', 'PRELOAD': '#eda100', 'STREAM': '#3a2d9a',
                'FLUSH': '#b9b8b2', 'DRAIN': '#e87ba4', 'TAIL': '#e4e3df', 'DONE': OK}
STATES = {'OS': ['IDLE', 'CLEAR', 'STREAM', 'FLUSH', 'DRAIN', 'TAIL', 'DONE'],
         'WS': ['IDLE', 'PRELOAD', 'STREAM', 'FLUSH', 'TAIL', 'DONE']}


# --- Trace ---------------------------------------------------------------------------------------------------

def capture(variant):
    b = T.bench(variant)
    f = b.dut
    recs = []

    def note(go=0):
        e = T.emission(b)
        recs.append(dict(state=f.state, cnt=f.cnt, t_out=f.t_out, t_mid=f.t_mid, go=go, e=e,
                         busy=b.get('busy'), done=b.get('done'), cycles=b.get('cycles'),
                         duration=f.duration(f.state) if f.state not in ('IDLE', 'DONE') else 1))

    for _ in range(2):
        note()
        b.step()
    note(go=1)
    T.start(b, *DIMS)
    while True:
        note()
        finished = b.get('done')
        b.step()
        if finished:
            break
    for _ in range(3):
        note()
        b.step()
    return recs


def decode_tag(tag):
    y_addr, mask, row = tag >> (N + IW), (tag >> IW) & 0xF, tag & ((1 << IW) - 1)
    return y_addr >> IW, y_addr & ((1 << IW) - 1), mask, row


# --- Panel -----------------------------------------------------------------------------------------------------

class Panel:
    BW, BH, STEP = 1.55, 0.72, 1.95
    X_TILES = 14.3
    X_SIG = 19.6

    def __init__(self, ax, variant, color, recs):
        self.ax, self.variant, self.color, self.recs = ax, variant, color, recs
        self.states = STATES[variant]
        M, K, Nd = DIMS
        self.rows = -(-M // N)
        self.cols = -(-Nd // N) if variant == 'OS' else -(-K // N)
        ax.set_xlim(-1.0, 27.4)
        ax.set_ylim(4.6, -2.0)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.text(-0.95, -1.7, 'output-stationary' if variant == 'OS' else 'weight-stationary', fontsize=12,
                fontweight='bold', color=color, va='center')
        # state diagram (static)
        self.pos = {s: (i * self.STEP, 0.6) for i, s in enumerate(self.states)}
        for a, b in zip(self.states[:-1], self.states[1:]):
            ax.annotate('', xy=(self.pos[b][0] - self.BW / 2, 0.6), xytext=(self.pos[a][0] + self.BW / 2, 0.6),
                        arrowprops=dict(arrowstyle='->', color=TEXT_2, lw=1))
        loop = ('DRAIN', 'CLEAR') if variant == 'OS' else ('FLUSH', 'PRELOAD')
        loop_label = 'next tile' if variant == 'OS' else 'next pair (i_t, k_t)'
        ax.annotate('', xy=(self.pos[loop[1]][0], 0.6 - self.BH / 2), xytext=(self.pos[loop[0]][0], 0.6 - self.BH / 2),
                    arrowprops=dict(arrowstyle='->', color=TEXT_2, lw=1, connectionstyle='arc3,rad=0.45'))
        ax.text((self.pos[loop[0]][0] + self.pos[loop[1]][0]) / 2, -0.85, loop_label, ha='center', fontsize=7,
                color=TEXT_2)
        ax.annotate('', xy=(self.pos['IDLE'][0], 0.6 + self.BH / 2), xytext=(self.pos['DONE'][0], 0.6 + self.BH / 2),
                    arrowprops=dict(arrowstyle='->', color=GRID, lw=1, connectionstyle='arc3,rad=0.12'))
        # tiles (static)
        ax.text(self.X_TILES - 0.4, -0.7, 'output tiles (i_t, j_t)' if variant == 'OS' else 'pairs (i_t, k_t)',
                fontsize=8, color=TEXT_2, va='center')
        for r in range(self.rows):
            ax.text(self.X_TILES - 0.65, r + 0.2, f'i_t={r}', ha='right', va='center', fontsize=6.5, color=TEXT_2)
        for c in range(self.cols):
            ax.text(self.X_TILES + c, self.rows - 0.15, f'{"j_t" if variant == "OS" else "k_t"}={c}', ha='center',
                    va='center', fontsize=6.5, color=TEXT_2)
        # signals (static)
        self.lines = ['w_addr', 'x_addr', 'w_mask', 'x_mask', 'ctl', 'drain' if variant == 'OS' else 'sideband', 'tag']
        for i, lab in enumerate(self.lines):
            ax.text(self.X_SIG, -1.15 + i * 0.78, lab, ha='left', va='center', fontsize=8, color=TEXT_2)
        self.boxes = Layer(ax, 2, lw=1.2, edge='white')
        self.glow = Layer(ax, 3, lw=2.5, edge=color)
        self.tiles = Layer(ax, 2, lw=1.0, edge='white')
        self.dots = Layer(ax, 4, lw=0.8, edge='white')
        self.txt = TextLayer(ax, 6)

    def draw(self, c, s):
        for l in (self.boxes, self.glow, self.tiles, self.dots, self.txt):
            l.reset()
        r = self.recs[min(c, len(self.recs) - 1)]
        bv, bc = box_path(self.BW, 0.12, self.BH)
        for st, (x, y) in self.pos.items():
            active = st == r['state']
            self.boxes.add(bv + (x, y), bc, self.color if active else CELL, 1.0)
            if active:
                self.glow.add(bv * 1.08 + (x, y), bc, 'none', ec_alpha=0.5 + 0.5 * (1 - s))
        # names of the states (that of the active state in white), and progress of the active state below
        for st, (xs, ys) in self.pos.items():
            self.txt.text(xs, ys, st, 'white' if st == r['state'] else TEXT_2, pt=7.5)
        x, y = self.pos[r['state']]
        prog = {'CLEAR': '', 'STREAM': f'{"k" if self.variant == "OS" else "j"} = {r["cnt"]}',
                'FLUSH': f'{r["cnt"] + 1}/{r["duration"]}', 'DRAIN': f'row {r["cnt"]}',
                'PRELOAD': f'k = {N * r["t_mid"] + N - 1 - r["cnt"]}', 'TAIL': f'{r["cnt"] + 1}/{r["duration"]}',
                'IDLE': 'go!' if r['go'] else 'waiting for go', 'DONE': 'done = 1'}[r['state']]
        if prog:
            self.txt.text(x, y + 0.75, prog, self.color if r['state'] not in ('IDLE', 'DONE') else TEXT, pt=7.5)
        # tiles
        cv, cc = box_path(0.84, 0.06)
        active = r['state'] not in ('IDLE', 'DONE', 'TAIL') and r['busy']
        done = r['state'] in ('TAIL', 'DONE') or (r['state'] == 'IDLE' and c > 3)
        for i in range(self.rows):
            for j in range(self.cols):
                before = (i, j) < (r['t_out'], r['t_mid'])
                current = active and (i, j) == (r['t_out'], r['t_mid'])
                fc = self.color if (before and r['busy']) or done else (CELL if not current else '#f1f0ec')
                self.tiles.add(cv + (self.X_TILES + j, i + 0.2), cc, fc, 0.85 if fc == self.color else 1)
                if current:
                    self.glow.add(cv * 1.12 + (self.X_TILES + j, i + 0.2), cc, 'none', ec_alpha=1.0)
        # signals emitted during this cycle
        e = r['e']
        xv = self.X_SIG + 2.1
        rows = [-1.15 + i * 0.78 for i in range(len(self.lines))]
        fmt = lambda a: f'{{{a >> IW}, {a & ((1 << IW) - 1)}}}' if a is not None else '—'
        self.txt.text(xv + 0.8, rows[0], fmt(e['w_addr']), TEXT if e['w_addr'] is not None else TEXT_2, pt=8)
        self.txt.text(xv + 0.8, rows[1], fmt(e['x_addr']), TEXT if e['x_addr'] is not None else TEXT_2, pt=8)
        for k, m in ((2, e['w_mask']), (3, e['x_mask'])):
            for l in range(N):
                dv, dc = box_path(0.3, 0.15)
                self.dots.add(dv + (xv + l * 0.42, rows[k]), dc, OK if (m >> l) & 1 else CELL)
        ctl_lab = ('clear' if self.variant == 'OS' else 'load') if e['ctl'] else '—'
        self.txt.text(xv + 0.8, rows[4], ctl_lab, self.color if e['ctl'] else TEXT_2, pt=8)
        if self.variant == 'OS':
            d = f'row {e["drain_row"]}' if e['drain_en'] else '—'
        else:
            d = (f'j = {e["s_j"]}' + (' F' if e['s_first'] else '') + (' L' if e['s_last'] else '')) if e['s_valid'] else '—'
        self.txt.text(xv + 0.8, rows[5], d, TEXT if d != '—' else TEXT_2, pt=8)
        if e['tag']:
            t, u, mask, row = decode_tag(e['tag'])
            self.txt.text(xv + 1.6, rows[6], f'{{{t}, {u}}}  row {row}', TEXT, pt=7.5)
        else:
            self.txt.text(xv + 0.8, rows[6], '—', TEXT_2, pt=8)
        return [self.boxes.commit(), self.glow.commit(), self.tiles.commit(), self.dots.commit(), self.txt.commit()]


class Gantt:
    def __init__(self, ax, recs_os, recs_ws):
        self.ax, self.recs = ax, {'OS': recs_os, 'WS': recs_ws}
        n = max(len(recs_os), len(recs_ws))
        ax.set_xlim(-0.5, n + 0.5)
        ax.set_ylim(-0.6, 1.6)
        ax.set_facecolor(SURFACE)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        ax.set_yticks([1, 0])
        ax.set_yticklabels(['OS', 'WS'])
        ax.tick_params(labelsize=8, colors=TEXT_2)
        ax.set_xlabel('cycle', fontsize=9, color=TEXT_2)
        # legend of the states
        x = 0
        for st, col in STATE_COLOR.items():
            ax.add_patch(plt.Rectangle((x, 1.42), 0.9, 0.14, color=col, clip_on=False))
            ax.text(x + 1.1, 1.49, st, fontsize=7, color=TEXT_2, va='center')
            x += 1.1 + 0.62 * len(st) + 0.6
        self.bars = Layer(ax, 2)
        self.cursor = ax.axvline(0, ymin=0.05, ymax=0.62, color=TEXT, lw=1, animated=True)

    def draw(self, c, s):
        self.bars.reset()
        hv, hc = RECT
        for y, v in ((1, 'OS'), (0, 'WS')):
            for k, r in enumerate(self.recs[v][:c + 1]):
                width = 1.0 if k < c else s
                self.bars.add(hv * (width, 0.6) + (k + width / 2 - 0.5, y), hc, STATE_COLOR[r['state']])
        self.cursor.set_xdata([c + s - 0.5])
        return [self.bars.commit(), self.cursor]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='saves the animation (gif) instead of showing it')
    ap.add_argument('--fps', type=int, default=30, help='frames per second')
    ap.add_argument('--cycle-ms', type=int, default=330, help='on-screen duration of one clock cycle (ms)')
    args = ap.parse_args()

    rec_os, rec_ws = capture('OS'), capture('WS')
    plt.rcParams['figure.facecolor'] = SURFACE
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.42], hspace=0.06)
    P_os = Panel(fig.add_subplot(gs[0]), 'OS', C_OS, rec_os)
    P_ws = Panel(fig.add_subplot(gs[1]), 'WS', C_WS, rec_ws)
    G = Gantt(fig.add_subplot(gs[2]), rec_os, rec_ws)
    title = title_axes(fig, (0, 0.95, 1, 0.05)).text(0.5, 0.75, '', ha='center', va='top', fontsize=14,
                                                      fontweight='bold', color=TEXT, animated=True)
    M, K, Nd = DIMS
    fig.text(0.5, 0.935, f'GEMM {M}×{K}×{Nd} (M×K×N)   ·   addresses {{tile, index}}   ·   green dot: valid lane '
             '(mask)   ·   F / L: first / last K tile', ha='center', fontsize=9, color=TEXT_2)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.05)

    n = max(len(rec_os), len(rec_ws))
    log = [dict(phase='compute')] * n
    tl = WA.Timeline(log, {'compute': 1000 / args.cycle_ms})

    class Everything:
        def draw(self, c, s):
            return P_os.draw(c, s) + P_ws.draw(c, s) + G.draw(min(c, n - 1), s if c < n else 1.0)

    def label(t):
        c, _ = tl.locate(t)
        return f'controller — cycle {min(c, n - 1)}'

    return WA.run_animation(fig, [Everything()], [tl], title, label, args.fps, args.save)


if __name__ == '__main__':
    main()
