"""
Animation du pe, cycle par cycle : les deux variantes côte à côte.

  OS : le PE accumule 6 produits w·x dans son accumulateur, après un clear.
  WS : le PE charge un poids (load), puis calcule psum_out = psum_in + w·x sur 6 colonnes.

Chaque image montre les entrées présentées pendant le cycle t et l'état des registres juste après le front.

Lancement : python test/pe/animation.py [--save fichier.gif] [--interval ms]
"""

import sys
import argparse
import pathlib
import importlib.util

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import common  # noqa: F401

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch

_spec = importlib.util.spec_from_file_location('pe_test', pathlib.Path(__file__).with_name('test.py'))
pe_test = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pe_test)


W_SEQ = [3, -2, 5, 1, -4, 2]
X_SEQ = [2, 7, -1, 3, 2, -5]
P_SEQ = [10, 20, 30, 40, 50, 60]
W_WS = 3


# --- Scénarios : on enregistre une trace (entrées du cycle, sorties après le front) ---

def trace_os():
    b = pe_test.bench_os()          # contient déjà un cycle de reset
    steps = [dict(rst_n=1, w_in=0, x_in=0, clear=1)]
    steps += [dict(rst_n=1, w_in=w, x_in=x, clear=0) for w, x in zip(W_SEQ, X_SEQ)]
    steps += [dict(rst_n=1, w_in=0, x_in=0, clear=0)] * 2
    frames, acc_before = [], 0
    for s in steps:
        b.step(**s)
        frames.append(dict(inp=s, out=b.snapshot(), acc_before=acc_before))
        acc_before = b.get('acc')
    return frames


def trace_ws():
    b = pe_test.bench_ws()
    steps = [dict(rst_n=1, x_in=0, psum_in=0, w_load_in=W_WS, load=1)]
    steps += [dict(rst_n=1, x_in=x, psum_in=p, w_load_in=0, load=0) for x, p in zip(X_SEQ, P_SEQ)]
    steps += [dict(rst_n=1, x_in=0, psum_in=0, w_load_in=0, load=0)] * 2
    frames, w_before = [], 0
    for s in steps:
        b.step(**s)
        frames.append(dict(inp=s, out=b.snapshot(), w_before=w_before))
        w_before = b.get('w_load_out')
    return frames


# --- Dessin -----------------------------------------------------------------------

ACTIVE = '#d62728'
IDLE = '#888888'
BOX = '#e8eef7'


def draw_pe(ax, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.add_patch(FancyBboxPatch((3, 3), 4, 4, boxstyle='round,pad=0.1', fc=BOX, ec='#335'))
    arrow = dict(arrowstyle='->', lw=1.8, color='#335')
    ax.annotate('', xy=(3, 5), xytext=(0.6, 5), arrowprops=arrow)      # ouest → PE
    ax.annotate('', xy=(5, 7), xytext=(5, 9.4), arrowprops=arrow)      # nord → PE
    ax.annotate('', xy=(9.4, 5), xytext=(7, 5), arrowprops=arrow)      # PE → est
    ax.annotate('', xy=(5, 0.6), xytext=(5, 3), arrowprops=arrow)      # PE → sud
    t = {}
    t['west'] = ax.text(0.5, 5.4, '', fontsize=10)
    t['north'] = ax.text(5.2, 9.0, '', fontsize=10)
    t['east'] = ax.text(7.3, 5.4, '', fontsize=10)
    t['south'] = ax.text(5.2, 1.0, '', fontsize=10)
    t['inside'] = ax.text(5, 5, '', fontsize=10, ha='center', va='center', family='monospace')
    t['ctl'] = ax.text(0.5, 8.6, '', fontsize=11, fontweight='bold')
    t['calc'] = ax.text(5, 2.2, '', fontsize=9, ha='center', color='#333',
                        bbox=dict(fc='white', ec='none', pad=1))
    return t


def update_os(t, f):
    i, o = f['inp'], f['out']
    t['west'].set_text(f"w_in = {i['w_in']}")
    t['north'].set_text(f"x_in = {i['x_in']}")
    t['east'].set_text(f"w_out = {o['w_out']}")
    t['south'].set_text(f"x_out = {o['x_out']}")
    t['inside'].set_text(f"w_r = {o['w_out']:>4}\nx_r = {o['x_out']:>4}\nacc = {o['acc']:>4}")
    t['ctl'].set_text('clear = 1' if i['clear'] else 'clear = 0')
    t['ctl'].set_color(ACTIVE if i['clear'] else IDLE)
    if i['clear']:
        t['calc'].set_text('acc ← 0')
    else:
        t['calc'].set_text(f"acc ← {f['acc_before']} + ({i['w_in']})·({i['x_in']}) = {o['acc']}")


def update_ws(t, f):
    i, o = f['inp'], f['out']
    t['west'].set_text(f"x_in = {i['x_in']}")
    t['north'].set_text(f"psum_in = {i['psum_in']}\nw_load_in = {i['w_load_in']}")
    t['east'].set_text(f"x_out = {o['x_out']}")
    t['south'].set_text(f"psum_out = {o['psum_out']}\nw_load_out = {o['w_load_out']}")
    t['inside'].set_text(f"w_r  = {o['w_load_out']:>4}\nx_r  = {o['x_out']:>4}\npsum = {o['psum_out']:>4}")
    t['ctl'].set_text('load = 1' if i['load'] else 'load = 0')
    t['ctl'].set_color(ACTIVE if i['load'] else IDLE)
    calc = f"psum ← {i['psum_in']} + ({f['w_before']})·({i['x_in']}) = {o['psum_out']}"
    if i['load']:
        calc += f"\nw_r ← {i['w_load_in']}"
    t['calc'].set_text(calc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='enregistre l\'animation (gif) au lieu de l\'afficher')
    ap.add_argument('--interval', type=int, default=1200, help='durée d\'une image en ms')
    args = ap.parse_args()

    fos, fws = trace_os(), trace_ws()
    n = len(fos)

    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.4])
    ax_os, ax_ws = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_tos, ax_tws = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    t_os = draw_pe(ax_os, 'PE output-stationary')
    t_ws = draw_pe(ax_ws, 'PE weight-stationary')
    title = fig.suptitle('', fontsize=13)

    acc = [f['out']['acc'] for f in fos]
    psum = [f['out']['psum_out'] for f in fws]
    for ax, data, lab in [(ax_tos, acc, 'acc'), (ax_tws, psum, 'psum_out')]:
        ax.set_xlim(-0.5, n - 0.5)
        lo, hi = min(data + [0]), max(data + [0])
        ax.set_ylim(lo - 0.1 * (hi - lo + 1), hi + 0.1 * (hi - lo + 1))
        ax.set_xlabel('cycle')
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    line_os, = ax_tos.plot([], [], 'o-', color='#1f77b4')
    line_ws, = ax_tws.plot([], [], 'o-', color='#2ca02c')

    def frame(k):
        update_os(t_os, fos[k])
        update_ws(t_ws, fws[k])
        line_os.set_data(range(k + 1), acc[:k + 1])
        line_ws.set_data(range(k + 1), psum[:k + 1])
        title.set_text(f'Cycle {k} : entrées présentées pendant le cycle → registres après le front')
        return []

    anim = FuncAnimation(fig, frame, frames=n, interval=args.interval, repeat=True)
    fig.tight_layout()
    if args.save:
        anim.save(args.save, writer=PillowWriter(fps=1000 / args.interval))
        print(f'animation enregistrée : {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
