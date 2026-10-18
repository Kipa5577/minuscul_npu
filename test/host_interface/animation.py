"""
Animation of the host_interface: the 32-bit bus seen from the inside, transaction by transaction.

The host software drives the complete chip for one layer (8×6×8 GEMM, OS variant):
loading W, X and b, configuration, GO, waiting for DONE (repeated read of STATUS), reading Y back.
For each cycle:
  - the current transaction (write or read, address, data) and its destination window,
    selected by bus_addr[12:10]: registers, WEIGHT, ACT_IN, ACT_OUT or BIAS;
  - a read returns its data in the next cycle (latency 1);
  - the live register table, and the go (pulse), busy (BUSY) and irq (DONE) lamps.

The speed depends on the phase. Rendering ≥ 30 frames/s (see test/viz.py).

Run: python test/host_interface/animation.py [--save file.gif] [--fps 30]
"""

import sys
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_test_module
from viz import (SURFACE, TEXT, TEXT_2, CELL, GRID, C_OS, C_W, C_X, ERR,
                 ease, box_path, Layer, TextLayer, TokenLayers, title_axes)

import matplotlib.pyplot as plt

TOP = load_test_module('npu_top', 'animation.py')
WA = load_test_module('weight_buffer', 'animation.py')
C_Y = '#1baf7a'
C_REG = '#eda100'
C_B = '#9085e9'
SPEED = {'loading': 12, 'configuration': 2.5, 'compute': 14, 'readback': 5}
WINDOWS = [('REG', 'registers', '0x0000', C_REG), ('WEIGHT', 'WEIGHT', '0x0400', C_W),
            ('ACT_IN', 'ACT_IN', '0x0800', C_X), ('ACT_OUT', 'ACT_OUT', '0x0C00', C_Y), ('BIAS', 'BIAS', '0x1000', C_B)]
REG_NAMES = ['ID', 'CTRL', 'STATUS', 'DIM_M', 'DIM_K', 'DIM_N', 'REQ_MULT', 'REQ_SHIFT', 'CYCLES']
ID_OS = 4 | (64 << 8) | (1 << 24)


def register_values(t):
    r = t['regs']
    return [ID_OS, t['relu'] << 1, t['host_busy'] | (t['irq'] << 1), r[3], r[4], r[5], r[6], r[7], t['cycles']]


class View:
    X_HOST, X_DEC, X_WIN, X_REG = 0.0, 4.2, 8.2, 12.6

    def __init__(self, ax, trace):
        self.ax, self.trace = ax, trace
        ax.set_xlim(-1.4, 19.6)
        ax.set_ylim(6.3, -1.6)
        ax.set_aspect('equal')
        ax.axis('off')
        self.y_win = {name: 0.3 + i * 1.15 for i, (name, *_) in enumerate(WINDOWS)}
        ax.text(self.X_HOST, 1.0, 'host', ha='center', fontsize=10, fontweight='bold', color=TEXT_2)
        ax.add_patch(plt.Rectangle((self.X_HOST - 1.1, 1.4), 2.2, 2.6, fc='none', ec=GRID, lw=1.2))
        ax.text(self.X_DEC, 0.9, 'decoding\nbus_addr[12:10]', ha='center', fontsize=8, color=TEXT_2)
        ax.add_patch(plt.Rectangle((self.X_DEC - 0.9, 1.4), 1.8, 2.6, fc='none', ec=GRID, lw=1.2))
        ax.annotate('', xy=(self.X_DEC - 0.9, 2.7), xytext=(self.X_HOST + 1.1, 2.7),
                    arrowprops=dict(arrowstyle='<->', color=GRID, lw=1.6))
        for name, lab, base, col in WINDOWS:
            y = self.y_win[name]
            ax.annotate('', xy=(self.X_WIN - 1.4, y), xytext=(self.X_DEC + 0.9, 2.7),
                        arrowprops=dict(arrowstyle='<->', color=GRID, lw=1.0))
        ax.text(self.X_REG + 2.6, -1.2, 'registers', ha='center', fontsize=9, fontweight='bold', color=TEXT_2)
        for i, n in enumerate(REG_NAMES):
            ax.text(self.X_REG, -0.6 + i * 0.62, n, fontsize=8, color=TEXT_2, va='center')
        for i, lab in enumerate(('go', 'busy (BUSY)', 'irq (DONE)')):
            ax.text(self.X_HOST - 0.6, 4.75 + i * 0.5, lab, fontsize=8, color=TEXT_2, va='center')
        self.boxes = Layer(ax, 1, lw=1.2, edge='white')
        self.glow = Layer(ax, 2, lw=2.2, edge=C_OS)
        self.lamps = Layer(ax, 3, lw=0.8, edge='white')
        self.tokens = TokenLayers(ax, 6)
        self.txt = TextLayer(ax, 8)

    def draw(self, c, s):
        for l in (self.boxes, self.glow, self.lamps, self.tokens, self.txt):
            l.reset()
        tr = self.trace
        live = c < len(tr)
        t = tr[min(c, len(tr) - 1)]
        prev = tr[c - 1] if 0 < c <= len(tr) else None
        bv, bc = box_path(2.8, 0.1, 0.85)
        target = TOP.target(t['addr']) if live and (t['bus_we'] or t['bus_re']) else None
        blocked = target not in (None, 'REG') and t['host_busy']
        for name, lab, base, col in WINDOWS:
            y = self.y_win[name]
            on = name == target
            self.boxes.add(bv + (self.X_WIN, y), bc, col if on else CELL, 0.9 if on else 1)
            self.txt.text(self.X_WIN - 0.3, y, lab, 'white' if on else TEXT_2, pt=8)
            self.txt.text(self.X_WIN + 0.95, y, base, 'white' if on else TEXT_2, bold=False, pt=6)
            if on:
                self.glow.add(bv * 1.06 + (self.X_WIN, y), bc, 'none', ec=ERR if blocked else C_OS,
                              ec_alpha=1 - 0.5 * s)
        u = ease(s / 0.85)
        # transaction of this cycle: from the host to the window
        if target:
            yf = self.y_win.get(target, 2.7) if target != '—' else 2.7
            x0, x1 = self.X_HOST + 1.1, self.X_WIN - 1.4
            col = dict((n, c_) for n, _, _, c_ in WINDOWS).get(target, TEXT_2)
            self.tokens.token(x0 + (x1 - x0) * u, 2.7 + (yf - 2.7) * u, 'W' if t['bus_we'] else 'R', col, size=0.45)
        # data read in the previous cycle: it comes back to the host (latency 1)
        if prev and prev['bus_re']:
            cp = TOP.target(prev['addr'])
            yf = self.y_win.get(cp, 2.7)
            x0, x1 = self.X_WIN - 1.4, self.X_HOST + 1.1
            self.tokens.token(x0 + (x1 - x0) * u, yf + (2.7 - yf) * u, 'D', C_OS, size=0.45)
        # card of the transaction
        if live:
            if t['bus_we']:
                l1, l2 = 'write', f'{self._name(t["addr"])}\n← 0x{t["wdata"]:08X}'
            elif t['bus_re']:
                l1, l2 = 'read', f'{self._name(t["addr"])}'
            else:
                l1, l2 = 'idle', ''
            self.txt.text(self.X_HOST, 1.9, l1, TEXT, pt=8.5)
            for i, line in enumerate(l2.split('\n')):
                self.txt.text(self.X_HOST, 2.45 + i * 0.45, line, TEXT_2, bold=False, pt=7)
            if prev and prev['bus_re']:
                self.txt.text(self.X_HOST, 3.6, f'→ 0x{t["rdata"]:08X}', C_OS, pt=7.5)
            if blocked:
                self.txt.text(self.X_WIN, -0.75, 'access ignored: NPU busy', ERR, pt=8)
        # registers: the written value appears at the clock edge (middle of the cycle on screen)
        after = tr[c + 1] if live and c + 1 < len(tr) and s >= 0.5 else t
        vals = register_values(after)
        active_reg = (t['addr'] & 0xF) if target == 'REG' else None
        for i, v in enumerate(vals):
            y = -0.6 + i * 0.62
            if i == active_reg:
                rv, rc = box_path(5.6, 0.08, 0.5)
                self.glow.add(rv + (self.X_REG + 2.75, y), rc, 'none', ec=C_REG, ec_alpha=1 - 0.5 * s)
            txt = f'{v}' if i not in (0, 1, 2) else f'0x{v:08X}'
            if i == 1:
                txt += f'  RELU_EN={after["relu"]}'
            if i == 2:
                txt = f'BUSY={after["host_busy"]}  DONE={after["irq"]}'
            self.txt.text(self.X_REG + 3.6, y, txt, TEXT, bold=False, pt=7.5)
        # lamps
        for i, (on, col) in enumerate(((t['go'], C_OS), (t['host_busy'], C_REG), (t['irq'], C_Y))):
            lv, lc = box_path(0.3, 0.15)
            self.lamps.add(lv + (self.X_HOST + 2.4, 4.75 + i * 0.5), lc, col if (on and live) else CELL)
        self.txt.text(self.X_WIN, 6.0, f'phase: {t["phase"]}', TEXT, pt=9)
        return [self.boxes.commit(), self.glow.commit(), self.lamps.commit()] + self.tokens.commit() + \
            [self.txt.commit()]

    @staticmethod
    def _name(addr):
        cb = TOP.target(addr)
        if cb == 'REG':
            r = addr & 0xF
            return REG_NAMES[r] if r < len(REG_NAMES) else f'REG[{r}]'
        if cb == '—':
            return f'0x{addr:04X} (empty)'
        base = {'WEIGHT': 0x400, 'ACT_IN': 0x800, 'ACT_OUT': 0xC00, 'BIAS': 0x1000}[cb]
        return f'{cb}[{addr - base}]'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='saves the animation (gif) instead of showing it')
    ap.add_argument('--fps', type=int, default=30, help='frames per second')
    args = ap.parse_args()

    trace = TOP.trace_chip('OS')
    plt.rcParams['figure.facecolor'] = SURFACE
    fig = plt.figure(figsize=(14, 6.6))
    view = View(fig.add_axes((0.01, 0.01, 0.98, 0.86)), trace)
    title = title_axes(fig, (0, 0.93, 1, 0.07)).text(0.5, 0.75, '', ha='center', va='top', fontsize=14,
                                                      fontweight='bold', color=TEXT, animated=True)
    fig.text(0.5, 0.905, 'W: write  ·  R: read request  ·  D: data returned in the next cycle  ·  '
             '8×6×8 GEMM in OS  ·  loading and waiting sped up', ha='center', fontsize=9, color=TEXT_2)
    tl = WA.Timeline(trace, SPEED)

    def label(t):
        c, _ = tl.locate(t)
        return f'host_interface — cycle {min(c, len(trace) - 1)}'

    return WA.run_animation(fig, [view], [tl], title, label, args.fps, args.save)


if __name__ == '__main__':
    main()
