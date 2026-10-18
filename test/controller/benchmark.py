"""
Benchmark du controller : combien de cycles coûte un GEMM, et quelle variante choisir selon sa forme.

Le controller seul se simule vite (pas de chemin de données) : on mesure le registre `cycles` après `done`,
pour des GEMM jusqu'à 64×64×64. Le test vérifie par ailleurs que ce controller pilote correctement le NPU complet.

  1. Cycles par GEMM D×D×D, mesurés contre les formules de la spécification
  2. Carte de décision : quelle variante est la plus rapide selon (K, N), pour M = 32
  3. Utilisation des PE quand K varie (M = N = 32) : OS s'effondre quand K est petit
  4. Utilisation des PE quand N varie (M = K = 32) : WS s'effondre quand N est petit

Lancement : python test/controller/benchmark.py [--save fichier.png]
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
    """Lance un GEMM sur le controller seul et lit le registre cycles après done."""
    b = _benches.setdefault(variant, T.bench(variant))
    T.lancer(b, M, K, Nd)
    while not b.get('done'):
        b.step()
    b.step()
    return b.get('cycles')


def tiles(D):
    return -(-D // N)


def formule(variant, M, K, Nd):
    """Formules de src/controller/interface.md, avec les états de fin (TAIL et DONE)."""
    if variant == 'OS':
        return tiles(M) * tiles(Nd) * (K + 3 * N) + 4 + 1
    return tiles(M) * tiles(K) * (Nd + 3 * N - 3) + 2 * N + 4 + 1


def utilisation(variant, M, K, Nd):
    return 100 * M * K * Nd / (N * N * cycles(variant, M, K, Nd))


# --- 1. Cycles mesurés contre formule ---------------------------------------------------------------------------

def mesure_cycles(Ds=(1, 2, 4, 8, 12, 16, 24, 32, 48, 64)):
    res = {v: [cycles(v, D, D, D) for D in Ds] for v in ('OS', 'WS')}
    ecarts = {v: max(abs(c - formule(v, D, D, D)) for c, D in zip(res[v], Ds)) for v in ('OS', 'WS')}
    return list(Ds), res, ecarts


def plot_cycles(ax, data):
    Ds, res, ecarts = data
    d = np.array(Ds)
    ax.plot(d, d ** 3 / (N * N), color=REF, lw=1, ls='--', label='idéal : D³ / 16 MAC par cycle')
    for v, color in (('OS', C_OS), ('WS', C_WS)):
        ax.plot(Ds, [formule(v, D, D, D) for D in Ds], color=color, lw=4, alpha=0.25)
        ax.plot(Ds, res[v], 'o', color=color, mec=SURFACE, mew=2, label=f'{v} mesuré (écart à la formule : {ecarts[v]})')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(Ds)
    ax.set_xticklabels(Ds)
    ax.set_ylim(bottom=5)
    ax.set_xlabel('taille du GEMM D (D×D×D)')
    ax.set_ylabel('cycles (go → done)')
    ax.set_title('1. Cycles par GEMM : mesure (points) et formule (trait large)', loc='left')
    ax.legend(loc='upper left')


# --- 2 et 3. Asymétrie OS / WS ---------------------------------------------------------------------------------

AXE = (1, 2, 4, 8, 16, 32, 64)


def mesure_asymetrie(base=32):
    k = {v: [utilisation(v, base, x, base) for x in AXE] for v in ('OS', 'WS')}
    n = {v: [utilisation(v, base, base, x) for x in AXE] for v in ('OS', 'WS')}
    return k, n


def plot_asym(ax, res, label, titre):
    for v, color in (('OS', C_OS), ('WS', C_WS)):
        ax.plot(AXE, res[v], 'o-', color=color, mec=SURFACE, mew=2, label=v)
    ax.set_xscale('log', base=2)
    ax.set_xticks(AXE)
    ax.set_xticklabels(AXE)
    ax.set_ylim(0, 100)
    ax.set_xlabel(label)
    ax.set_ylabel('utilisation des PE (%)')
    ax.set_title(titre, loc='left')
    ax.legend(loc='upper left')


# --- 4. Carte de décision -----------------------------------------------------------------------------------------

def mesure_carte(M=32):
    r = np.array([[cycles('OS', M, k, n) / cycles('WS', M, k, n) for n in AXE] for k in AXE])
    return M, r


def plot_carte(ax, data):
    M, r = data
    cmap = LinearSegmentedColormap.from_list('div', [C_WS, '#f1f0ec', C_OS])
    lr = np.log2(r)
    lim = np.abs(lr).max()
    ax.imshow(-lr, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), origin='lower')
    for (i, j), v in np.ndenumerate(r):
        gagnant = 'OS' if v < 1 else 'WS'
        f = 1 / v if v < 1 else v
        ax.text(j, i, f'{gagnant}\n×{f:.1f}' if abs(v - 1) > 0.02 else '=', ha='center', va='center', fontsize=6.5,
                color='white' if abs(np.log2(v)) > lim * 0.55 else TEXT)
    ax.set_xticks(range(len(AXE)))
    ax.set_xticklabels(AXE)
    ax.set_yticks(range(len(AXE)))
    ax.set_yticklabels(AXE)
    ax.set_xlabel('N (colonnes de X et de Y)')
    ax.set_ylabel('K (dimension de réduction)')
    ax.grid(False)
    ax.set_title(f'2. Variante la plus rapide (et de combien), M = {M}', loc='left')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='enregistre la figure (png) au lieu de l\'afficher')
    args = ap.parse_args()

    print('cycles par GEMM...')
    cy = mesure_cycles()
    print('asymétrie OS / WS...')
    k, n = mesure_asymetrie()
    print('carte de décision...')
    ca = mesure_carte()

    Ds, res, ecarts = cy
    print('\nRésumé')
    for v in ('OS', 'WS'):
        print(f'  {v} : ' + ', '.join(f'D={d} {c}' for d, c in zip(Ds, res[v])) + f'  (écart max à la formule : {ecarts[v]})')
    print('  utilisation selon K (M = N = 32) : ' + ', '.join(f'K={x} OS {a:.0f} % / WS {b:.0f} %'
                                                         for x, a, b in zip(AXE, k['OS'], k['WS'])))
    print('  utilisation selon N (M = K = 32) : ' + ', '.join(f'N={x} OS {a:.0f} % / WS {b:.0f} %'
                                                         for x, a, b in zip(AXE, n['OS'], n['WS'])))
    M, r = ca
    print(f'  carte (M = {M}) : OS plus rapide sur {np.sum(r < 0.98)} formes, WS sur {np.sum(r > 1.02)}, '
          f'égalité sur {np.sum(abs(r - 1) <= 0.02)}')

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    plot_cycles(axs[0, 0], cy)
    plot_carte(axs[0, 1], ca)
    plot_asym(axs[1, 0], k, 'K (dimension de réduction)', '3. Utilisation quand K varie (M = N = 32)')
    plot_asym(axs[1, 1], n, 'N (colonnes de X et de Y)', '4. Utilisation quand N varie (M = K = 32)')
    fig.suptitle('Benchmark du controller — coût d\'un GEMM et choix de la variante', fontsize=13,
                 fontweight='bold', color=TEXT)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f'\nfigure enregistrée : {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
