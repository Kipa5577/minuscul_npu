"""
Benchmark du host_interface : ce que coûte le passage par le bus de 32 bits.

Toutes les mesures sont faites sur la puce complète (npu_top), pilotée uniquement par son bus par le logiciel
hôte de test/npu_top/test.py : un accès par cycle, un mot de 32 bits par accès.

  1. Où passent les cycles de bus pour une couche D×D×D : chargement de W, de X, des biais, configuration,
     attente du calcul (scrutation de STATUS), relecture de Y
  2. Débit effectif : MAC par cycle, hôte compris, comparé au calcul seul et au crête de la grille (16)
  3. Données échangées sur le bus par MAC calculé
  4. Un MLP 64 → 48 → 32 → 16 (lot de 16) sur la puce : chargement, recopie de Y vers ACT_IN entre couches,
     calcul et relecture finale

Lancement : python test/host_interface/benchmark.py [--save fichier.png]
"""

import sys
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_test_module
from viz import BENCH_RC, SURFACE, TEXT, TEXT_2, REF, C_OS, C_WS

import numpy as np
import matplotlib.pyplot as plt

T = load_test_module('npu_top')
plt.rcParams.update(BENCH_RC)

PHASES = (('W', 'chargement de W', '#6250d6'), ('X', 'chargement de X', '#1baf7a'), ('b', 'biais', '#9085e9'),
          ('cfg', 'configuration + GO', '#eda100'), ('calcul', 'calcul (scrutation de STATUS)', '#3a2d9a'),
          ('Y', 'relecture de Y', '#e87ba4'))


def couche_detaillee(npu, W, X, bias, mult=1, shift=10, relu=True, load_x=True):
    """Une couche, phase par phase ; renvoie (Y, cycles par phase, cycles de calcul lus dans CYCLES)."""
    M, K = W.shape
    Nd = X.shape[1]
    ph = {}

    def phase(nom, fn):
        c = npu.cycle
        r = fn()
        ph[nom] = npu.cycle - c
        return r

    phase('W', lambda: npu.load_w(W))
    phase('X', lambda: npu.load_x(X) if load_x else None)
    phase('b', lambda: npu.load_bias(bias[:M]))
    phase('cfg', lambda: (npu.configure(M, K, Nd, mult, shift), npu.start(relu)))
    phase('calcul', npu.wait_done)
    cycles = npu.read(T.CYCLES)
    Y = phase('Y', lambda: npu.read_y(M, Nd))
    return Y, ph, cycles


def mesure_couches(Ds=(4, 8, 16, 32, 64)):
    rng = np.random.default_rng(200)
    res = {}
    for v in ('OS', 'WS'):
        res[v] = []
        for D in Ds:
            npu = T.Npu(v)
            W, X = rng.integers(-128, 128, (D, D)), rng.integers(-128, 128, (D, D))
            _, ph, cyc = couche_detaillee(npu, W, X, [0] * D)
            res[v].append((ph, cyc))
    return list(Ds), res


def plot_phases(ax, data):
    Ds, res = data
    x = np.arange(len(Ds))
    for off, v in ((-0.2, 'OS'), (0.2, 'WS')):
        tot = np.array([sum(ph.values()) for ph, _ in res[v]], float)
        bottom = np.zeros(len(Ds))
        for key, label, color in PHASES:
            part = 100 * np.array([ph[key] for ph, _ in res[v]]) / tot
            ax.bar(x + off, part, width=0.36, bottom=bottom, color=color, edgecolor=SURFACE, linewidth=1.2,
                   label=label if v == 'OS' else None)
            bottom += part
        for xi, t in zip(x, tot):
            ax.text(xi + off, 101, f'{v}\n{int(t)}', ha='center', va='bottom', fontsize=6.5, color=TEXT_2)
    ax.set_xticks(x)
    ax.set_xticklabels([f'D = {d}' for d in Ds])
    ax.set_ylim(0, 118)
    ax.set_ylabel('part des cycles de bus (%)')
    ax.set_title('1. Où passent les cycles d\'une couche D×D×D (au-dessus : total)', loc='left')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncols=3)
    ax.grid(axis='x', visible=False)


def plot_debit(ax, data):
    Ds, res = data
    ax.axhline(16, color=REF, lw=1, ls='--')
    ax.text(Ds[0], 16.3, 'crête de la grille 4×4 : 16 MAC/cycle', fontsize=8, color=TEXT_2)
    for v, color in (('OS', C_OS), ('WS', C_WS)):
        total = [D ** 3 / sum(ph.values()) for D, (ph, _) in zip(Ds, res[v])]
        calcul = [D ** 3 / cyc for D, (_, cyc) in zip(Ds, res[v])]
        ax.plot(Ds, calcul, 'o--', color=color, mec=SURFACE, mew=2, label=f'{v} : calcul seul (CYCLES)')
        ax.plot(Ds, total, 'o-', color=color, mec=SURFACE, mew=2, label=f'{v} : hôte compris')
    ax.set_xscale('log', base=2)
    ax.set_xticks(Ds)
    ax.set_xticklabels(Ds)
    ax.set_ylim(0, 17.5)
    ax.set_xlabel('taille de la couche D (D×D×D)')
    ax.set_ylabel('MAC par cycle')
    ax.set_title('2. Débit effectif de la puce', loc='left')
    ax.legend(loc='center left', handlelength=3)


def plot_octets(ax, data):
    Ds, res = data
    ph0 = res['OS']
    octets = [4 * (ph['W'] + ph['X'] + ph['b'] + ph['Y']) / D ** 3 for D, (ph, _) in zip(Ds, ph0)]
    ax.plot(Ds, octets, 'o-', color='#3a2d9a', mec=SURFACE, mew=2)
    for D, o in zip(Ds, octets):
        ax.annotate(f'{o:.2f}', (D, o), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8, color=TEXT)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(Ds)
    ax.set_xticklabels(Ds)
    ax.set_xlabel('taille de la couche D (D×D×D)')
    ax.set_ylabel('octets échangés sur le bus par MAC')
    ax.set_title('3. Données transférées par l\'hôte, par MAC calculé (identique OS / WS)', loc='left')


def mesure_mlp(dims=(64, 48, 32, 16), batch=16):
    rng = np.random.default_rng(201)
    res = {}
    for v in ('OS', 'WS'):
        npu = T.Npu(v)
        X = rng.integers(-128, 128, (dims[0], batch))
        couches = []
        for li, (k, m) in enumerate(zip(dims[:-1], dims[1:])):
            W = rng.integers(-128, 128, (m, k))
            c0 = npu.cycle
            Y, ph, _ = couche_detaillee(npu, W, X, [0] * m, 1, 12, True, load_x=(li == 0))
            dern = li == len(dims) - 2
            c1 = npu.cycle
            if not dern:
                npu.copy_out_to_in(m, batch)
            couches.append(dict(chargement=ph['W'] + ph['X'] + ph['b'] + ph['cfg'], calcul=ph['calcul'],
                                recopie=npu.cycle - c1, relecture=ph['Y'] if dern else 0, nom=f'{k}→{m}'))
            X = Y
        res[v] = couches
    return res


def plot_mlp(ax, res):
    parts = (('chargement', 'chargement (W, X, b, config)', '#9085e9'), ('calcul', 'calcul', '#3a2d9a'),
             ('recopie', 'recopie Y → ACT_IN (relecture comprise)', '#e87ba4'), ('relecture', 'relecture finale', '#eda100'))
    y = 0
    labels = []
    for v in ('OS', 'WS'):
        for c in res[v]:
            left = 0
            for key, lab, color in parts:
                ax.barh(y, c[key], left=left, color=color, edgecolor=SURFACE, linewidth=1.2, height=0.7,
                        label=lab if y == 0 else None)
                left += c[key]
            ax.text(left + 60, y, str(left), va='center', fontsize=7.5, color=TEXT_2)
            labels.append(f'{v}  {c["nom"]}')
            y += 1
        y += 0.6
    ax.set_yticks([i if i < 3 else i + 0.6 for i in range(6)])
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('cycles')
    tot = {v: sum(sum(c[k] for k, _, _ in parts) for c in res[v]) for v in ('OS', 'WS')}
    ax.set_title(f'4. MLP 64→48→32→16, lot de 16 — total OS {tot["OS"]} / WS {tot["WS"]} cycles', loc='left')
    ax.legend(loc='lower right', fontsize=7.5)
    ax.grid(axis='y', visible=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='enregistre la figure (png) au lieu de l\'afficher')
    args = ap.parse_args()

    print('couches D×D×D sur la puce complète (jusqu\'à D = 64)...')
    co = mesure_couches()
    print('MLP à 3 couches...')
    ml = mesure_mlp()

    Ds, res = co
    print('\nRésumé')
    for v in ('OS', 'WS'):
        print(f'  {v} : ' + ', '.join(f'D={d} {D3 / sum(ph.values()):.1f} MAC/cycle (calcul seul {D3 / cyc:.1f})'
                                     for d, (ph, cyc), D3 in zip(Ds, res[v], [d ** 3 for d in Ds])))
    for v in ('OS', 'WS'):
        print(f'  MLP {v} : ' + ', '.join(f'{c["nom"]} charge {c["chargement"]} calcul {c["calcul"]} '
                                         f'recopie {c["recopie"]}' for c in ml[v]))

    fig, axs = plt.subplots(2, 2, figsize=(14, 10.5))
    plot_phases(axs[0, 0], co)
    plot_debit(axs[0, 1], co)
    plot_octets(axs[1, 0], co)
    plot_mlp(axs[1, 1], ml)
    fig.suptitle('Benchmark du host_interface — le coût du bus de 32 bits, mesuré sur la puce complète',
                 fontsize=13, fontweight='bold', color=TEXT)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f'\nfigure enregistrée : {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
