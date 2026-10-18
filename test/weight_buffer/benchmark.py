"""
Benchmark of the weight_buffer: how the compute core uses the weight memory.

All the measurements come from the complete datapath with its memories (see test/activation_buffer/test.py):
the host loads W and X, the core computes, and each read of the weight_buffer is counted.

  1. Number of reads of W for a D×D×D GEMM, compared with the minimum (each word read once)
  2. Occupation of the read port during the computation
  3. and 4. Timeline of the reads (address read at each cycle), 16×16×16 GEMM, OS then WS

Run: python test/weight_buffer/benchmark.py [--save file.png]
"""

import sys
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_test_module
from viz import BENCH_RC, SURFACE, TEXT, TEXT_2, REF, C_OS, C_WS
import layout

import numpy as np
import matplotlib.pyplot as plt

AB = load_test_module('activation_buffer')
plt.rcParams.update(BENCH_RC)


def gemm_log(variant, D, seed=150):
    rng = np.random.default_rng(seed)
    W, X = rng.integers(-128, 128, (D, D)), rng.integers(-128, 128, (D, D))
    b = AB.datapath(variant)
    log = []
    AB.run_layer(b, variant, W, X, [0] * D, 1, 8, 0, log=log)
    compute = [e for e in log if e['phase'] == 'compute']
    return compute


def measure_reads(Ds=(4, 8, 12, 16, 24, 32)):
    res = {v: dict(reads=[], occupation=[]) for v in ('OS', 'WS')}
    for D in Ds:
        for v in ('OS', 'WS'):
            compute = gemm_log(v, D)
            n = sum(e['w_rd'] is not None for e in compute)
            res[v]['reads'].append(n)
            res[v]['occupation'].append(100 * n / len(compute))
    minimum = [len(layout.w_image(np.zeros((D, D), int))) for D in Ds]
    return list(Ds), res, minimum


def plot_reads(ax, data):
    Ds, res, minimum = data
    ax.plot(Ds, minimum, color=REF, lw=4, alpha=0.35, label='minimum: each word read once (= WS)')
    for v, color in (('OS', C_OS), ('WS', C_WS)):
        ax.plot(Ds, res[v]['reads'], 'o-', color=color, mec=SURFACE, mew=2, label=v)
    ax.set_yscale('log')
    ax.set_xticks(Ds)
    ax.set_xlabel('GEMM size D (D×D×D)')
    ax.set_ylabel('weight_buffer reads')
    ax.set_title('1. Reads of W per GEMM', loc='left')
    ax.legend(loc='upper left')


def plot_occupation(ax, data):
    Ds, res, _ = data
    for v, color in (('OS', C_OS), ('WS', C_WS)):
        ax.plot(Ds, res[v]['occupation'], 'o-', color=color, mec=SURFACE, mew=2, label=v)
        ax.annotate(f'{res[v]["occupation"][-1]:.0f} %', (Ds[-1], res[v]['occupation'][-1]), xytext=(6, 0),
                    textcoords='offset points', va='center', fontsize=8, color=TEXT)
    ax.set_ylim(0, 100)
    ax.set_xticks(Ds)
    ax.set_xlabel('GEMM size D (D×D×D)')
    ax.set_ylabel('compute cycles with a read (%)')
    ax.set_title('2. Occupation of the read port during the computation', loc='left')
    ax.legend(loc='center right')


def timeline(variant, D=16):
    """(cycle, linear address t·D + k) of each read of W during the computation."""
    compute = gemm_log(variant, D)
    pts = [(c, (e['w_rd'] >> layout.IW) * D + (e['w_rd'] & ((1 << layout.IW) - 1)))
           for c, e in enumerate(compute) if e['w_rd'] is not None]
    return pts, len(compute), D


def plot_timeline(ax, data, variant, color, xmax):
    pts, n, D = data
    c, a = zip(*pts)
    ax.scatter(c, a, s=5, color=color, lw=0)
    for t in range(1, layout.tiles(D)):
        ax.axhline(t * D - 0.5, color=REF, lw=0.6, ls=':')
    ax.set_xlim(0, xmax)
    ax.set_ylim(-1, layout.tiles(D) * D)
    ax.set_yticks([t * D + D / 2 for t in range(layout.tiles(D))])
    ax.set_yticklabels([f'i_t = {t}' for t in range(layout.tiles(D))])
    ax.set_xlabel('compute cycle')
    ax.set_title(f'{3 if variant == "OS" else 4}. Reads of W during the computation, {D}³ GEMM — {variant} '
                 f'({len(pts)} reads)', loc='left')
    ax.grid(axis='y', visible=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='saves the figure (png) instead of showing it')
    args = ap.parse_args()

    print('reads of W as a function of the GEMM size...')
    rd = measure_reads()
    print('timeline of the reads...')
    ch_os, ch_ws = timeline('OS'), timeline('WS')

    Ds, res, minimum = rd
    print('\nSummary')
    for v in ('OS', 'WS'):
        print(f'  {v}: reads ' + ', '.join(f'D={d} {n} ({n / mn:.1f}× the minimum)'
                                             for d, n, mn in zip(Ds, res[v]['reads'], minimum)))
        print(f'      port occupation ' + ', '.join(f'D={d} {o:.0f} %' for d, o in zip(Ds, res[v]['occupation'])))

    fig, axs = plt.subplots(2, 2, figsize=(14, 9.5))
    plot_reads(axs[0, 0], rd)
    plot_occupation(axs[0, 1], rd)
    xmax = max(ch_os[1], ch_ws[1])
    plot_timeline(axs[1, 0], ch_os, 'OS', C_OS, xmax)
    plot_timeline(axs[1, 1], ch_ws, 'WS', C_WS, xmax)
    fig.suptitle('weight_buffer benchmark — accesses of the compute core to the weight memory', fontsize=13,
                 fontweight='bold', color=TEXT)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f'\nfigure saved: {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
