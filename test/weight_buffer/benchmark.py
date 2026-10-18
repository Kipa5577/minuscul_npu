"""
Benchmark du weight_buffer : comment le cœur de calcul utilise la mémoire des poids.

Toutes les mesures viennent du chemin de données complet avec ses mémoires (voir test/activation_buffer/test.py) :
l'hôte charge W et X, le cœur calcule, et chaque lecture du weight_buffer est comptée.

  1. Nombre de lectures de W pour un GEMM D×D×D, comparé au minimum (chaque mot lu une fois)
  2. Occupation du port de lecture pendant le calcul
  3. et 4. Chronologie des lectures (adresse lue à chaque cycle), GEMM 16×16×16, OS puis WS

Lancement : python test/weight_buffer/benchmark.py [--save fichier.png]
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
    calcul = [e for e in log if e['phase'] == 'calcul']
    return calcul


def mesure_lectures(Ds=(4, 8, 12, 16, 24, 32)):
    res = {v: dict(lectures=[], occupation=[]) for v in ('OS', 'WS')}
    for D in Ds:
        for v in ('OS', 'WS'):
            calcul = gemm_log(v, D)
            n = sum(e['w_rd'] is not None for e in calcul)
            res[v]['lectures'].append(n)
            res[v]['occupation'].append(100 * n / len(calcul))
    minimum = [len(layout.w_image(np.zeros((D, D), int))) for D in Ds]
    return list(Ds), res, minimum


def plot_lectures(ax, data):
    Ds, res, minimum = data
    ax.plot(Ds, minimum, color=REF, lw=4, alpha=0.35, label='minimum : chaque mot lu une fois (= WS)')
    for v, color in (('OS', C_OS), ('WS', C_WS)):
        ax.plot(Ds, res[v]['lectures'], 'o-', color=color, mec=SURFACE, mew=2, label=v)
    ax.set_yscale('log')
    ax.set_xticks(Ds)
    ax.set_xlabel('taille du GEMM D (D×D×D)')
    ax.set_ylabel('lectures du weight_buffer')
    ax.set_title('1. Lectures de W par GEMM', loc='left')
    ax.legend(loc='upper left')


def plot_occupation(ax, data):
    Ds, res, _ = data
    for v, color in (('OS', C_OS), ('WS', C_WS)):
        ax.plot(Ds, res[v]['occupation'], 'o-', color=color, mec=SURFACE, mew=2, label=v)
        ax.annotate(f'{res[v]["occupation"][-1]:.0f} %', (Ds[-1], res[v]['occupation'][-1]), xytext=(6, 0),
                    textcoords='offset points', va='center', fontsize=8, color=TEXT)
    ax.set_ylim(0, 100)
    ax.set_xticks(Ds)
    ax.set_xlabel('taille du GEMM D (D×D×D)')
    ax.set_ylabel('cycles de calcul avec une lecture (%)')
    ax.set_title('2. Occupation du port de lecture pendant le calcul', loc='left')
    ax.legend(loc='center right')


def chronologie(variant, D=16):
    """(cycle, adresse linéaire t·D + k) de chaque lecture de W pendant le calcul."""
    calcul = gemm_log(variant, D)
    pts = [(c, (e['w_rd'] >> layout.IW) * D + (e['w_rd'] & ((1 << layout.IW) - 1)))
           for c, e in enumerate(calcul) if e['w_rd'] is not None]
    return pts, len(calcul), D


def plot_chronologie(ax, data, variant, color, xmax):
    pts, n, D = data
    c, a = zip(*pts)
    ax.scatter(c, a, s=5, color=color, lw=0)
    for t in range(1, layout.tiles(D)):
        ax.axhline(t * D - 0.5, color=REF, lw=0.6, ls=':')
    ax.set_xlim(0, xmax)
    ax.set_ylim(-1, layout.tiles(D) * D)
    ax.set_yticks([t * D + D / 2 for t in range(layout.tiles(D))])
    ax.set_yticklabels([f'i_t = {t}' for t in range(layout.tiles(D))])
    ax.set_xlabel('cycle de calcul')
    ax.set_title(f'{3 if variant == "OS" else 4}. Lectures de W au fil du calcul, GEMM {D}³ — {variant} '
                 f'({len(pts)} lectures)', loc='left')
    ax.grid(axis='y', visible=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='enregistre la figure (png) au lieu de l\'afficher')
    args = ap.parse_args()

    print('lectures de W en fonction de la taille du GEMM...')
    lec = mesure_lectures()
    print('chronologie des lectures...')
    ch_os, ch_ws = chronologie('OS'), chronologie('WS')

    Ds, res, minimum = lec
    print('\nRésumé')
    for v in ('OS', 'WS'):
        print(f'  {v} : lectures ' + ', '.join(f'D={d} {n} ({n / mn:.1f}× le minimum)'
                                             for d, n, mn in zip(Ds, res[v]['lectures'], minimum)))
        print(f'       occupation du port ' + ', '.join(f'D={d} {o:.0f} %' for d, o in zip(Ds, res[v]['occupation'])))

    fig, axs = plt.subplots(2, 2, figsize=(14, 9.5))
    plot_lectures(axs[0, 0], lec)
    plot_occupation(axs[0, 1], lec)
    xmax = max(ch_os[1], ch_ws[1])
    plot_chronologie(axs[1, 0], ch_os, 'OS', C_OS, xmax)
    plot_chronologie(axs[1, 1], ch_ws, 'WS', C_WS, xmax)
    fig.suptitle('Benchmark du weight_buffer — accès du cœur de calcul à la mémoire des poids', fontsize=13,
                 fontweight='bold', color=TEXT)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f'\nfigure enregistrée : {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
