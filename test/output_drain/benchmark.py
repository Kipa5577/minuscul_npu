"""
Benchmark de l'output_drain : limites et compromis, sous forme de graphiques matplotlib.

Avec l'output_drain, le cœur de calcul (data_feeder → pe_array → output_drain) est complet :
les mesures 2 à 4 sont faites de bout en bout, sur des GEMM pilotés par le planificateur du test
(ce que la FSM du controller émet), et Y est vérifié à chaque fois.

  1. Latence mesurée : OS (drain_en → res_valid) et WS (émission de X → res_valid) en fonction de N
  2. Rythme de sortie des résultats sur un GEMM 16×16×16 : rafales (OS) contre flux continu (WS)
  3. Utilisation de la grille en fonction de la taille du GEMM (D×D×D)
  4. Répartition des cycles par phase de la FSM sur un GEMM 16×16×16

Lancement : python test/output_drain/benchmark.py [--save fichier.png]
"""

import sys
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import pack, load_test_module
from viz import BENCH_RC, SURFACE, TEXT, TEXT_2, REF, C_OS, C_WS

import numpy as np
import matplotlib.pyplot as plt

T = load_test_module('output_drain')
AW = T.AW
plt.rcParams.update(BENCH_RC)


# --- 1. Latence -----------------------------------------------------------------------------------------

def mesure_latences(Ns=range(1, 9)):
    os_lat, ws_lat = [], []
    for N in Ns:
        b = T.bench_os(N)
        b.step(acc=pack(range(1, N * N + 1), AW), drain_en=1, drain_row=0, tag_in=1)
        os_lat.append(1 if b.get('res_valid') else None)
        b = T.bench_ws(N)
        for c in range(4 * N + 4):
            south = [7 if c == 1 + i + N else 0 for i in range(N)]
            sb = dict(s_valid=1, s_j=0, s_first=1, s_last=1, tag_in=1) if c == 0 else dict(s_valid=0)
            b.step(psum_south=pack(south, AW), **sb)
            if b.get('res_valid'):
                ws_lat.append(c + 1)
                break
    return list(Ns), os_lat, ws_lat


def plot_latences(ax, data):
    Ns, os_lat, ws_lat = data
    n = np.array(Ns)
    ax.plot(n, 2 * n + 1, color=C_WS, lw=1, ls='--', alpha=0.7)
    ax.plot(Ns, ws_lat, 'o', color=C_WS, mec=SURFACE, mew=2, label='WS : émission de X → res_valid')
    ax.plot(Ns, os_lat, 'o-', color=C_OS, mec=SURFACE, mew=2, label='OS : drain_en → res_valid')
    ax.text(Ns[-1] - 0.2, 2 * Ns[-1] + 1 + 0.6, '2N + 1', ha='right', fontsize=8, color=TEXT_2)
    ax.set_xlabel('côté de la grille N')
    ax.set_ylabel('latence (cycles)')
    ax.set_title('1. Latence mesurée de l\'output_drain', loc='left')
    ax.legend(loc='upper left')


# --- GEMM de bout en bout ----------------------------------------------------------------------------------

def gemm(variant, D, seed):
    """GEMM D×D×D de bout en bout ; renvoie (plan, sorties, cycles jusqu'au dernier résultat)."""
    ref, Y, count, out, em = T.gemm_bout_en_bout(variant, D, D, D, seed)
    assert (count == 1).all() and (Y == ref).all(), (variant, D)
    return em, out, out[-1][0]


def mesure_rythme(D=16):
    res = {}
    for v in ('OS', 'WS'):
        em, out, fin = gemm(v, D, 80)
        cycles = [c for c, tag, _ in out]
        elems = np.cumsum([bin(T.tag_unpack(tag)[1]).count('1') for _, tag, _ in out])
        res[v] = (cycles, elems, fin, em)
    return D, res


def plot_rythme(ax, data):
    D, res = data
    for v, color in (('OS', C_OS), ('WS', C_WS)):
        cycles, elems, fin, _ = res[v]
        ax.step([0] + cycles, [0] + list(elems), where='post', color=color, label=f'{v} : fin au cycle {fin}')
    ax.axhline(D * D, color=REF, lw=1, ls='--')
    ax.text(0, D * D + 5, f'{D}×{D} = {D * D} éléments de Y', fontsize=8, color=TEXT_2)
    ax.set_xlabel('cycle')
    ax.set_ylabel('éléments de Y sortis (cumul)')
    ax.set_title(f'2. Rythme de sortie, GEMM {D}×{D}×{D}', loc='left')
    ax.legend(loc='center right')


def mesure_utilisation(Ds=(4, 8, 12, 16, 20, 24, 32), N=4):
    res = {}
    for v in ('OS', 'WS'):
        res[v] = [100 * D ** 3 / (N * N * gemm(v, D, 90 + D)[2]) for D in Ds]
    return list(Ds), res


def plot_utilisation(ax, data):
    Ds, res = data
    for v, color in (('OS', C_OS), ('WS', C_WS)):
        ax.plot(Ds, res[v], 'o-', color=color, mec=SURFACE, mew=2, label=v)
        ax.annotate(f'{res[v][-1]:.0f} %', (Ds[-1], res[v][-1]), xytext=(6, 0), textcoords='offset points',
                    va='center', fontsize=8, color=TEXT)
    ax.set_ylim(0, 100)
    ax.set_xticks(Ds)
    ax.set_xlabel('taille du GEMM D (D×D×D)')
    ax.set_ylabel('utilisation de la grille (%)')
    ax.set_title('3. Utilisation de bout en bout (cœur de calcul complet, N = 4)', loc='left')
    ax.legend(loc='lower right')


PHASES = [('flux', 'flux (MAC utiles possibles)', '#3a2d9a'), ('clear', 'clear', '#9085e9'),
          ('préchargement', 'préchargement', '#eda100'), ('vidange', 'vidange de la grille', '#b9b8b2'),
          ('lecture', 'lecture des accumulateurs', '#e87ba4'), ('fin', 'fin du pipeline', '#e4e3df')]


def plot_phases(ax, data):
    D, res = data
    y = {'OS': 1, 'WS': 0}
    for v in ('OS', 'WS'):
        em = res[v][3]
        left = 0
        for key, label, color in PHASES:
            n = sum(1 for e in em if e['phase'] == key)
            if n:
                ax.barh(y[v], n, left=left, color=color, edgecolor=SURFACE, linewidth=2, height=0.55,
                        label=label if v == 'OS' or key == 'préchargement' else None)
                if n / len(em) > 0.06:
                    ax.text(left + n / 2, y[v], f'{100 * n / len(em):.0f} %', ha='center', va='center',
                            fontsize=8, color='white' if key == 'flux' else TEXT)
                left += n
    ax.set_yticks([1, 0])
    ax.set_yticklabels(['OS', 'WS'])
    ax.set_xlabel('cycles')
    ax.set_title(f'4. Répartition des cycles par phase, GEMM {D}×{D}×{D}', loc='left')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncols=3)
    ax.grid(axis='y', visible=False)
    ax.set_ylim(-0.6, 1.6)


# --- Principal ------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='enregistre la figure (png) au lieu de l\'afficher')
    args = ap.parse_args()

    print('mesure des latences...')
    lat = mesure_latences()
    print('GEMM 16×16×16 de bout en bout...')
    ry = mesure_rythme()
    print('utilisation en fonction de la taille du GEMM...')
    ut = mesure_utilisation()

    Ns, ol, wl = lat
    print('\nRésumé')
    print(f'  latence OS : {dict(zip(Ns, ol))}')
    print(f'  latence WS : {dict(zip(Ns, wl))}')
    D, r = ry
    for v in ('OS', 'WS'):
        em, fin = r[v][3], r[v][2]
        phases = {k: sum(1 for e in em if e['phase'] == k) for k, _, _ in PHASES}
        print(f'  GEMM {D}³ {v} : dernier résultat au cycle {fin} ; phases {phases}')
    Ds, u = ut
    for v in ('OS', 'WS'):
        print(f'  utilisation {v} : ' + ', '.join(f'D={d} {x:.0f} %' for d, x in zip(Ds, u[v])))

    fig, axs = plt.subplots(2, 2, figsize=(14, 9.5))
    plot_latences(axs[0, 0], lat)
    plot_rythme(axs[0, 1], ry)
    plot_utilisation(axs[1, 0], ut)
    plot_phases(axs[1, 1], ry)
    fig.suptitle('Benchmark de l\'output_drain et du cœur de calcul — output-stationary vs weight-stationary',
                 fontsize=13, fontweight='bold', color=TEXT)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f'\nfigure enregistrée : {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
