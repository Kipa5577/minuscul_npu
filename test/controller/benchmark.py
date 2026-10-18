"""
Benchmark of the controller: how many cycles a GEMM costs, and which variant to choose according to its shape.

The controller alone simulates fast (no datapath): the `cycles` register is measured after `done`,
for GEMMs up to 64×64×64. The test separately checks that this controller drives the complete NPU correctly.

  1. Cycles per D×D×D GEMM, measured against the formulas of the specification
  2. Decision map: which variant is faster according to (K, N), for M = 32
  3. PE utilisation when K varies (M = N = 32): OS collapses when K is small
  4. PE utilisation when N varies (M = K = 32): WS collapses when N is small

Run: python test/controller/benchmark.py [--save file.png]
"""

import sys
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_test_module
from viz import BENCH_RC, SURFACE, TEXT, TEXT_2, REF, C_OS, C_WS

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

T = load_test_module('controller')
N = 4
plt.rcParams.update(BENCH_RC)

_benches = {}


def cycles(variant, M, K, Nd):
    """Starts a GEMM on the controller alone and reads the cycles register after done."""
    b = _benches.setdefault(variant, T.bench(variant))
    T.start(b, M, K, Nd)
    while not b.get('done'):
        b.step()
    b.step()
    return b.get('cycles')


def tiles(D):
    return -(-D // N)


def formula(variant, M, K, Nd):
    """Formulas of src/controller/interface.md, with the final states (TAIL and DONE)."""
    if variant == 'OS':
        return tiles(M) * tiles(Nd) * (K + 3 * N) + 4 + 1
    return tiles(M) * tiles(K) * (Nd + 3 * N - 3) + 2 * N + 4 + 1


def utilisation(variant, M, K, Nd):
    return 100 * M * K * Nd / (N * N * cycles(variant, M, K, Nd))


# --- 1. Measured cycles versus formula --------------------------------------------------------------------------

def measure_cycles(Ds=(1, 2, 4, 8, 12, 16, 24, 32, 48, 64)):
    res = {v: [cycles(v, D, D, D) for D in Ds] for v in ('OS', 'WS')}
    diffs = {v: max(abs(c - formula(v, D, D, D)) for c, D in zip(res[v], Ds)) for v in ('OS', 'WS')}
    return list(Ds), res, diffs


def plot_cycles(ax, data):
    Ds, res, diffs = data
    d = np.array(Ds)
    ax.plot(d, d ** 3 / (N * N), color=REF, lw=1, ls='--', label='ideal: D³ / 16 MACs per cycle')
    for v, color in (('OS', C_OS), ('WS', C_WS)):
        ax.plot(Ds, [formula(v, D, D, D) for D in Ds], color=color, lw=4, alpha=0.25)
        ax.plot(Ds, res[v], 'o', color=color, mec=SURFACE, mew=2, label=f'{v} measured (difference from the formula: {diffs[v]})')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(Ds)
    ax.set_xticklabels(Ds)
    ax.set_ylim(bottom=5)
    ax.set_xlabel('GEMM size D (D×D×D)')
    ax.set_ylabel('cycles (go → done)')
    ax.set_title('1. Cycles per GEMM: measurement (dots) and formula (wide line)', loc='left')
    ax.legend(loc='upper left')


# --- 2 and 3. OS / WS asymmetry --------------------------------------------------------------------------------

AXIS = (1, 2, 4, 8, 16, 32, 64)


def measure_asymmetry(base=32):
    k = {v: [utilisation(v, base, x, base) for x in AXIS] for v in ('OS', 'WS')}
    n = {v: [utilisation(v, base, base, x) for x in AXIS] for v in ('OS', 'WS')}
    return k, n


def plot_asym(ax, res, label, title):
    for v, color in (('OS', C_OS), ('WS', C_WS)):
        ax.plot(AXIS, res[v], 'o-', color=color, mec=SURFACE, mew=2, label=v)
    ax.set_xscale('log', base=2)
    ax.set_xticks(AXIS)
    ax.set_xticklabels(AXIS)
    ax.set_ylim(0, 100)
    ax.set_xlabel(label)
    ax.set_ylabel('PE utilisation (%)')
    ax.set_title(title, loc='left')
    ax.legend(loc='upper left')


# --- 4. Decision map ----------------------------------------------------------------------------------------------

def measure_map(M=32):
    r = np.array([[cycles('OS', M, k, n) / cycles('WS', M, k, n) for n in AXIS] for k in AXIS])
    return M, r


def plot_map(ax, data):
    M, r = data
    cmap = LinearSegmentedColormap.from_list('div', [C_WS, '#f1f0ec', C_OS])
    lr = np.log2(r)
    lim = np.abs(lr).max()
    ax.imshow(-lr, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), origin='lower')
    for (i, j), v in np.ndenumerate(r):
        winner = 'OS' if v < 1 else 'WS'
        f = 1 / v if v < 1 else v
        ax.text(j, i, f'{winner}\n×{f:.1f}' if abs(v - 1) > 0.02 else '=', ha='center', va='center', fontsize=6.5,
                color='white' if abs(np.log2(v)) > lim * 0.55 else TEXT)
    ax.set_xticks(range(len(AXIS)))
    ax.set_xticklabels(AXIS)
    ax.set_yticks(range(len(AXIS)))
    ax.set_yticklabels(AXIS)
    ax.set_xlabel('N (columns of X and Y)')
    ax.set_ylabel('K (reduction dimension)')
    ax.grid(False)
    ax.set_title(f'2. Faster variant (and by how much), M = {M}', loc='left')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='saves the figure (png) instead of showing it')
    args = ap.parse_args()

    print('cycles per GEMM...')
    cy = measure_cycles()
    print('OS / WS asymmetry...')
    k, n = measure_asymmetry()
    print('decision map...')
    ca = measure_map()

    Ds, res, diffs = cy
    print('\nSummary')
    for v in ('OS', 'WS'):
        print(f'  {v}: ' + ', '.join(f'D={d} {c}' for d, c in zip(Ds, res[v])) + f'  (max difference from the formula: {diffs[v]})')
    print('  utilisation according to K (M = N = 32): ' + ', '.join(f'K={x} OS {a:.0f} % / WS {b:.0f} %'
                                                         for x, a, b in zip(AXIS, k['OS'], k['WS'])))
    print('  utilisation according to N (M = K = 32): ' + ', '.join(f'N={x} OS {a:.0f} % / WS {b:.0f} %'
                                                         for x, a, b in zip(AXIS, n['OS'], n['WS'])))
    M, r = ca
    print(f'  map (M = {M}): OS faster on {np.sum(r < 0.98)} shapes, WS on {np.sum(r > 1.02)}, '
          f'tie on {np.sum(abs(r - 1) <= 0.02)}')

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    plot_cycles(axs[0, 0], cy)
    plot_map(axs[0, 1], ca)
    plot_asym(axs[1, 0], k, 'K (reduction dimension)', '3. Utilisation when K varies (M = N = 32)')
    plot_asym(axs[1, 1], n, 'N (columns of X and Y)', '4. Utilisation when N varies (M = K = 32)')
    fig.suptitle('controller benchmark — cost of a GEMM and choice of the variant', fontsize=13,
                 fontweight='bold', color=TEXT)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f'\nfigure saved: {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
