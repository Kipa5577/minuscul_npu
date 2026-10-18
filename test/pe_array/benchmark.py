"""
Benchmark du pe_array : limites et compromis, sous forme de graphiques matplotlib.

  1. Front d'onde : nombre de PE qui font un MAC utile à chaque cycle (une tuile, N = 4)
  2. Cycles pour une tuile N×16×N en fonction de N (mesuré vs formule)
  3. Utilisation des PE en fonction de la dimension envoyée en flux (K en OS, J en WS)
  4. Vitesse de simulation py4hw en fonction de N

Pour une tuile, OS et WS font exactement le même travail : N·N·L MAC
(OS : W N×L, X L×N ; WS : W N×N préchargé, X N×L).

Lancement : python test/pe_array/benchmark.py [--save fichier.png]
"""

import sys
import time
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import pack, load_test_module

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

T = load_test_module('pe_array')
DW = T.DW


# --- Style (identique au benchmark du pe) ----------------------------------------------

SURFACE = '#fcfcfb'
TEXT = '#0b0b0b'
TEXT_2 = '#52514e'
GRID = '#e4e3df'
C_OS = '#2a78d6'
C_WS = '#eb6834'
REF = '#8a8984'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': TEXT_2, 'axes.titlecolor': TEXT,
    'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.labelsize': 9,
    'xtick.color': TEXT_2, 'ytick.color': TEXT_2, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 2, 'lines.markersize': 7,
    'legend.fontsize': 8, 'legend.frameon': False, 'text.color': TEXT,
})


# --- Exécution d'une tuile avec observation des PE --------------------------------------

def step_observe(b, actif, **values):
    """Présente les entrées, compte les PE actifs pendant ce cycle, puis avance d'un front."""
    b.set(**values)
    b.sim.propagateAll()
    n = sum(actif(pe) for row in b.dut.pes for pe in row)
    b.sim.clk(1)
    b.cycle += 1
    return n


actif_os = lambda pe: pe.w_in.get() != 0 and pe.x_in.get() != 0 and not pe.clear.get()
actif_ws = lambda pe: pe.x_in.get() != 0 and pe.w_r != 0 and not pe.load.get()


def tuile_os(N, L, rng):
    """Une tuile OS (K = L). Renvoie (PE actifs par cycle, nombre de cycles jusqu'au résultat complet)."""
    W = rng.integers(1, 128, (N, L))          # valeurs non nulles : tout MAC injecté est utile
    X = rng.integers(1, 128, (L, N))
    Y = W @ X
    b = T.bench_os(N)
    actifs, fin = [], None
    for c in range(L + 2 * N + 1):
        wl, xl = T.os_lanes(W, X, c)
        actifs.append(step_observe(b, actif_os, w_west=wl, x_north=xl, clear=0))
        if fin is None and (T.acc_matrix(b) == Y).all():
            fin = c + 1
    return actifs, fin


def tuile_ws(N, L, rng):
    """Préchargement + une tuile WS (J = L). Renvoie (PE actifs par cycle, cycles jusqu'à la dernière sortie)."""
    W = rng.integers(1, 128, (N, N))
    X = rng.integers(1, 128, (N, L))
    Y = W @ X
    b = T.bench_ws(N)
    actifs = []
    for k in reversed(range(N)):
        actifs.append(step_observe(b, actif_ws, w_north=pack(W[:, k], DW), x_west=0, load=1))
    fin, sortie = None, []
    for c in range(L + 2 * N + 1):
        actifs.append(step_observe(b, actif_ws, x_west=T.ws_x_lanes(X, c), w_north=0, load=0))
        sortie.append(T.psum_south(b)[N - 1])
        # fin = le cycle où la voie N−1 vient de sortir toute la ligne Y[N−1][0..L−1], dans l'ordre
        if fin is None and sortie[-L:] == list(Y[N - 1]):
            fin = N + c + 1
    return actifs, fin


# --- Mesures ---------------------------------------------------------------------------

def mesure_front_onde(N=4, L=16):
    rng = np.random.default_rng(20)
    return tuile_os(N, L, rng)[0], tuile_ws(N, L, rng)[0], N, L


def mesure_cycles_vs_N(L=16, Ns=range(1, 9)):
    rng = np.random.default_rng(21)
    return list(Ns), [tuile_os(N, L, rng)[1] for N in Ns], [tuile_ws(N, L, rng)[1] for N in Ns], L


def mesure_utilisation(Ls=(1, 2, 4, 8, 16, 32, 64), Ns=(4, 8)):
    rng = np.random.default_rng(22)
    res = {}
    for N in Ns:
        for var, f in (('OS', tuile_os), ('WS', tuile_ws)):
            res[(var, N)] = [100 * L / f(N, L, rng)[1] for L in Ls]      # N²·L MAC / (N²·cycles)
    return list(Ls), res


def mesure_vitesse(Ns=(2, 4, 8), cycles=1500):
    res = {'OS': [], 'WS': []}
    for N in Ns:
        for var, mk, v in (('OS', T.bench_os, dict(w_west=pack([3] * N, DW), x_north=pack([2] * N, DW), clear=0)),
                           ('WS', T.bench_ws, dict(x_west=pack([3] * N, DW), w_north=0, load=0))):
            b = mk(N)
            t0 = time.time()
            for _ in range(cycles):
                b.step(**v)
            res[var].append(cycles / (time.time() - t0))
    return list(Ns), res


# --- Graphiques ----------------------------------------------------------------------------

def plot_front_onde(ax, data):
    a_os, a_ws, N, L = data
    ax.step(range(len(a_os)), a_os, where='mid', color=C_OS, label='OS')
    ax.step(range(len(a_ws)), a_ws, where='mid', color=C_WS, label='WS')
    ax.axvspan(-0.5, N - 0.5, color=C_WS, alpha=0.08, lw=0)
    ax.text(N / 2 - 0.5, N * N + 0.6, 'préchargement\nWS', ha='center', fontsize=7, color=TEXT_2)
    ax.axhline(N * N, color=REF, lw=1, ls='--')
    ax.text(len(a_ws) - 0.5, N * N - 0.9, f'{N * N} PE', ha='right', fontsize=7, color=TEXT_2)
    ax.set_ylim(0, N * N + 2.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('cycle')
    ax.set_ylabel('PE faisant un MAC utile')
    ax.set_title(f'1. Front d\'onde d\'une tuile (N = {N}, flux de {L})', loc='left')
    ax.legend(loc='upper right')


def plot_cycles_vs_N(ax, data):
    Ns, c_os, c_ws, L = data
    n = np.array(Ns)
    ax.plot(n, L + 2 * n - 2, color=C_OS, lw=1, ls='--', alpha=0.7)
    ax.plot(n, L + 3 * n - 2, color=C_WS, lw=1, ls='--', alpha=0.7)
    ax.plot(Ns, c_os, 'o', color=C_OS, mec=SURFACE, mew=2, label='OS mesuré')
    ax.plot(Ns, c_ws, 'o', color=C_WS, mec=SURFACE, mew=2, label='WS mesuré')
    xo = Ns[-1] - 1.5
    ax.text(xo, L + 2 * xo - 2 - 2.5, f'OS : {L} + 2N − 2', ha='left', va='top', fontsize=8, color=TEXT_2)
    ax.annotate(f'WS : N + {L} + 2N − 2', (Ns[-1], L + 3 * Ns[-1] - 2), xytext=(-6, 6),
                textcoords='offset points', ha='right', fontsize=8, color=TEXT_2)
    ax.axhline(L, color=REF, lw=1, ls=':')
    ax.text(Ns[0], L + 0.5, f'idéal : {L} cycles', fontsize=7, color=TEXT_2)
    ax.set_xlabel('côté de la grille N')
    ax.set_ylabel('cycles jusqu\'au dernier résultat')
    ax.set_title(f'2. Cycles pour une tuile (flux de {L}), mesure vs formule', loc='left')
    ax.legend(loc='upper left')


def plot_utilisation(ax, data):
    Ls, res = data
    for (var, N), u in res.items():
        color = C_OS if var == 'OS' else C_WS
        ls = '-' if N == 4 else '--'
        ax.plot(Ls, u, marker='o', ls=ls, color=color, mec=SURFACE, mew=2, label=f'{var}, N = {N}')
    ax.set_xscale('log', base=2)
    ax.set_xticks(Ls)
    ax.set_xticklabels(Ls)
    ax.set_ylim(0, 100)
    ax.set_xlabel('longueur du flux (K en OS, J en WS)')
    ax.set_ylabel('utilisation des PE (%)')
    ax.set_title('3. Utilisation des PE pour une tuile isolée', loc='left')
    ax.legend(loc='upper left', handlelength=3.5)


def plot_vitesse(ax, data):
    Ns, res = data
    x = np.arange(len(Ns))
    for off, (var, color) in zip((-0.18, 0.18), (('OS', C_OS), ('WS', C_WS))):
        vals = [v / 1000 for v in res[var]]
        bars = ax.bar(x + off, vals, width=0.34, color=color, edgecolor=SURFACE, linewidth=2, label=var)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{v:.0f} k',
                    ha='center', va='bottom', fontsize=8, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels([f'N = {n}' for n in Ns])
    ax.set_ylabel('milliers de cycles simulés par seconde')
    ax.set_title('4. Vitesse de simulation py4hw', loc='left')
    ax.legend(loc='upper right')
    ax.grid(axis='x', visible=False)


# --- Principal ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='enregistre la figure (png) au lieu de l\'afficher')
    args = ap.parse_args()

    print('mesure du front d\'onde...')
    fo = mesure_front_onde()
    print('mesure des cycles en fonction de N...')
    cn = mesure_cycles_vs_N()
    print('mesure de l\'utilisation...')
    ut = mesure_utilisation()
    print('mesure de la vitesse de simulation...')
    vi = mesure_vitesse()

    Ns, c_os, c_ws, L = cn
    print('\nRésumé')
    print(f'  cycles (flux {L})  : OS {dict(zip(Ns, c_os))}')
    print(f'                     WS {dict(zip(Ns, c_ws))}')
    Ls, res = ut
    for (var, N), u in res.items():
        print(f'  utilisation {var} N={N} : ' + ', '.join(f'{l}→{x:.0f} %' for l, x in zip(Ls, u)))
    print('  vitesse            : ' + ', '.join(f'{v} N={n} {s:,.0f} c/s'.replace(',', ' ')
                                                 for v in vi[1] for n, s in zip(vi[0], vi[1][v])))

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    plot_front_onde(axs[0, 0], fo)
    plot_cycles_vs_N(axs[0, 1], cn)
    plot_utilisation(axs[1, 0], ut)
    plot_vitesse(axs[1, 1], vi)
    fig.suptitle('Benchmark du pe_array — output-stationary vs weight-stationary', fontsize=13,
                 fontweight='bold', color=TEXT)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f'\nfigure enregistrée : {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
