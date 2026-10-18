"""
Benchmark du data_feeder : limites et compromis, sous forme de graphiques matplotlib.

  1. Latence mesurée de chaque voie (réponse à une impulsion), N = 8
  2. Efficacité des voies selon la dimension : le masquage gaspille les voies quand D n'est pas multiple de N
  3. Réutilisation des données : MAC utiles par octet lu en SRAM (chaîne data_feeder → pe_array)
  4. Débit SRAM nécessaire : octets lus par cycle pendant une tuile

Lancement : python test/data_feeder/benchmark.py [--save fichier.png]
"""

import sys
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import pack, load_test_module
from viz import BENCH_RC, SURFACE, TEXT, TEXT_2, REF, C_OS, C_WS

import numpy as np
import matplotlib.pyplot as plt

T = load_test_module('data_feeder')
DW = T.DW
plt.rcParams.update(BENCH_RC)


# --- 1. Latence par voie --------------------------------------------------------------------------

def mesure_latences(N=8):
    """Impulsion sur toutes les voies à l'émission 0 ; latence = premier cycle où la voie sort la valeur."""
    res = {}
    w = [l + 1 for l in range(N)]
    for v in ('OS', 'WS'):
        h = T.run(T.bench(v, N), [dict(w=w, x=w, w_mask=(1 << N) - 1, x_mask=(1 << N) - 1, ctl=1)], extra=N + 2)
        for flux in ('w', 'x'):
            res[(v, flux)] = [next(c for c, s in enumerate(h) if s[flux][l] == w[l]) for l in range(N)]
        res[(v, 'ctl')] = next(c for c, s in enumerate(h) if s['ctl'])
    return N, res


def plot_latences(ax, data):
    N, res = data
    lanes = np.arange(N)
    styles = [(('OS', 'w'), C_OS, '-', 'OS : W'), (('OS', 'x'), C_OS, ':', 'OS : X'),
              (('WS', 'w'), C_WS, '-', 'WS : W (préchargement)'), (('WS', 'x'), C_WS, ':', 'WS : X')]
    for key, color, ls, label in styles:
        off = {('OS', 'w'): -0.09, ('OS', 'x'): -0.03, ('WS', 'w'): 0.03, ('WS', 'x'): 0.09}[key]
        ax.plot(lanes + off, res[key], marker='o', ls=ls, color=color, mec=SURFACE, mew=2, label=label)
    ax.plot(lanes, 1 + lanes, color=REF, lw=1, ls='--', zorder=0)
    ax.text(N - 1, N + 0.3, '1 + l', fontsize=8, color=TEXT_2, ha='right')
    ax.set_xticks(lanes)
    ax.set_xlabel('voie l')
    ax.set_ylabel('latence depuis l\'émission (cycles)')
    ax.set_title(f'1. Latence mesurée de chaque voie (N = {N})', loc='left')
    ax.legend(loc='upper left', handlelength=3)


# --- 2. Efficacité des voies (masquage) ---------------------------------------------------------------

def mesure_efficacite(Ns=(2, 4, 8), Dmax=64):
    """Envoie les tuiles d'une dimension D (un mot par tuile) et compte les voies non masquées en sortie."""
    res = {}
    for N in Ns:
        eff = []
        for D in range(1, Dmax + 1):
            tuiles = -(-D // N)
            em = [dict(w=[1] * N, x=[1] * N, w_mask=T.mask_of(D - N * t, N), x_mask=0, ctl=0) for t in range(tuiles)]
            h = T.run(T.bench('WS', N), em, extra=2)          # W non décalé : une tuile = un cycle de sortie
            utiles = sum(sum(1 for v in s['w'] if v) for s in h)
            eff.append(100 * utiles / (N * tuiles))
        res[N] = eff
    return list(range(1, Dmax + 1)), res


def plot_efficacite(ax, data):
    Ds, res = data
    for N, color in zip(res, ('#9085e9', '#6250d6', '#3a2d9a')):
        ax.plot(Ds, res[N], color=color, lw=1.6, label=f'N = {N}')
    ax.set_ylim(0, 105)
    ax.set_xlim(1, Ds[-1])
    ax.set_xlabel('dimension D de la matrice (M, K ou N)')
    ax.set_ylabel('voies utiles (%)')
    ax.set_title('2. Voies gaspillées par le masquage des bords', loc='left')
    ax.legend(loc='lower right')


# --- 3 et 4. Réutilisation et débit SRAM (chaîne data_feeder → pe_array) -----------------------------

def tuile(variant, N, L, rng):
    """Une tuile complète. Renvoie (MAC utiles, octets lus en SRAM, cycles de la première lecture au dernier MAC)."""
    b = T.chain(variant, N)
    full = (1 << N) - 1
    if variant == 'OS':
        W, X = rng.integers(1, 128, (N, L)), rng.integers(1, 128, (L, N))
        em = [dict(w=[0] * N, x=[0] * N, w_mask=0, x_mask=0, ctl=1)]
        em += [dict(w=list(W[:, k]), x=list(X[k]), w_mask=full, x_mask=full, ctl=0) for k in range(L)]
        octets = 2 * N * L
        actif = lambda pe: pe.w_in.get() != 0 and pe.x_in.get() != 0
    else:
        W, X = rng.integers(1, 128, (N, N)), rng.integers(1, 128, (N, L))
        em = [dict(w=list(W[:, k]), x=[0] * N, w_mask=full, x_mask=0, ctl=1) for k in reversed(range(N))]
        em += [dict(w=[0] * N, x=list(X[:, j]), w_mask=0, x_mask=full, ctl=0) for j in range(L)]
        octets = N * N + N * L
        actif = lambda pe: pe.x_in.get() != 0 and pe.w_r != 0 and not pe.load.get()
    idle = dict(w=[0] * N, x=[0] * N, w_mask=0, x_mask=0, ctl=0)
    prev, macs, dernier = idle, 0, 0
    first_read = 0 if variant == 'WS' else 1
    for e in range(len(em) + 3 * N):
        cur = em[e] if e < len(em) else idle
        b.set(w_rdata=pack(prev['w'], DW), x_rdata=pack(prev['x'], DW),
              w_mask=cur['w_mask'], x_mask=cur['x_mask'], ctl_in=cur['ctl'])
        b.sim.propagateAll()
        n = sum(actif(pe) for row in b.dut.pes for pe in row)
        if n:
            macs, dernier = macs + n, e
        b.sim.clk(1)
        prev = cur
    return macs, octets, dernier + 1 - first_read


def mesure_reutilisation(Ns=(2, 4, 8), L=64):
    rng = np.random.default_rng(40)
    res = {}
    for v in ('OS', 'WS'):
        for N in Ns:
            macs, octets, cycles = tuile(v, N, L, rng)
            assert macs == N * N * L, (v, N, macs)
            res[(v, N)] = dict(mac_par_octet=macs / octets, octets_par_cycle=octets / cycles,
                               macs=macs, octets=octets, cycles=cycles)
    return list(Ns), L, res


def plot_barres(ax, data, cle, ylabel, titre, fmt):
    Ns, L, res = data
    x = np.arange(len(Ns))
    for off, (v, color) in zip((-0.18, 0.18), (('OS', C_OS), ('WS', C_WS))):
        vals = [res[(v, N)][cle] for N in Ns]
        bars = ax.bar(x + off, vals, width=0.34, color=color, edgecolor=SURFACE, linewidth=2, label=v)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(val),
                    ha='center', va='bottom', fontsize=8, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels([f'N = {n}' for n in Ns])
    ax.set_ylabel(ylabel)
    ax.set_title(titre.format(L=L), loc='left')
    ax.legend(loc='upper left')
    ax.grid(axis='x', visible=False)


# --- Principal ------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='enregistre la figure (png) au lieu de l\'afficher')
    args = ap.parse_args()

    print('mesure des latences...')
    lat = mesure_latences()
    print('mesure de l\'efficacité des voies...')
    eff = mesure_efficacite()
    print('mesure de la réutilisation et du débit SRAM...')
    reu = mesure_reutilisation()

    N, r = lat
    print('\nRésumé')
    for k in (('OS', 'w'), ('OS', 'x'), ('WS', 'w'), ('WS', 'x')):
        print(f'  latence {k[0]} {k[1].upper()} (N={N}) : {r[k]}')
    print(f'  latence ctl : OS {r[("OS", "ctl")]}, WS {r[("WS", "ctl")]}')
    Ds, e = eff
    for n in e:
        print(f'  efficacité moyenne des voies N={n} (D = 1..{Ds[-1]}) : {np.mean(e[n]):.0f} %')
    Ns, L, res = reu
    for (v, n), d in res.items():
        print(f'  {v} N={n} : {d["macs"]} MAC, {d["octets"]} octets lus, {d["cycles"]} cycles '
              f'→ {d["mac_par_octet"]:.1f} MAC/octet, {d["octets_par_cycle"]:.1f} octets/cycle')

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    plot_latences(axs[0, 0], lat)
    plot_efficacite(axs[0, 1], eff)
    plot_barres(axs[1, 0], reu, 'mac_par_octet', 'MAC utiles par octet lu',
                '3. Réutilisation des données (tuile, flux de {L})', '{:.1f}')
    plot_barres(axs[1, 1], reu, 'octets_par_cycle', 'octets lus en SRAM par cycle',
                '4. Débit SRAM nécessaire (tuile, flux de {L})', '{:.1f}')
    fig.suptitle('Benchmark du data_feeder — output-stationary vs weight-stationary', fontsize=13,
                 fontweight='bold', color=TEXT)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f'\nfigure enregistrée : {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
