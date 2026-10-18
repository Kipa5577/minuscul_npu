"""
Benchmark de l'activation_buffer : la mémoire des activations dans le fonctionnement complet du NPU.

Toutes les mesures viennent du chemin de données complet avec ses mémoires, piloté comme le fera l'hôte
(chargement de W, X et b mot par mot, GEMM, relecture de Y) — voir run_layer dans test.py.

  1. Occupation des ports pendant le calcul : lecture du banc IN (X) et écriture du banc OUT (Y)
  2. Où passe le temps : chargement par l'hôte, calcul, relecture de Y, en fonction de la taille du GEMM
     (en v1 il n'y a ni DMA ni double tampon : l'hôte transfère un mot par cycle et le cœur attend)
  3. Enchaîner les couches d'un réseau : recopie de Y du banc OUT vers le banc IN, comparée au calcul
  4. Accès mémoire totaux par type, GEMM 32×32×32

Lancement : python test/activation_buffer/benchmark.py [--save fichier.png]
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

T = load_test_module('activation_buffer')
plt.rcParams.update(BENCH_RC)


def layer(variant, Mr, K, Nd, seed=160, relu=1):
    rng = np.random.default_rng(seed)
    W, X = rng.integers(-128, 128, (Mr, K)), rng.integers(-128, 128, (K, Nd))
    b = T.datapath(variant)
    log = []
    _, st = T.run_layer(b, variant, W, X, [0] * Mr, 1, 10, relu, log=log)
    return st, log


def compter(log):
    calcul = [e for e in log if e['phase'] == 'calcul']
    return dict(
        cycles_calcul=len(calcul),
        lect_x=sum(e['x_rd'] is not None for e in calcul),
        ecr_y=sum(e['y_wr'] is not None for e in calcul),
        lect_w=sum(e['w_rd'] is not None for e in calcul),
        hote_ecr=sum((e['wb_w'] is not None) + (e['ab_w'] is not None) for e in log),
        hote_lect=sum(e['ab_r'] is not None for e in log))


# --- 1. Occupation des ports ------------------------------------------------------------------------------------

def mesure_occupation(Ds=(4, 8, 16, 24, 32)):
    res = {}
    for v in ('OS', 'WS'):
        occ_in, occ_out = [], []
        for D in Ds:
            n = compter(layer(v, D, D, D)[1])
            occ_in.append(100 * n['lect_x'] / n['cycles_calcul'])
            occ_out.append(100 * n['ecr_y'] / n['cycles_calcul'])
        res[v] = (occ_in, occ_out)
    return list(Ds), res


def plot_occupation(ax, data):
    Ds, res = data
    for v, color in (('OS', C_OS), ('WS', C_WS)):
        occ_in, occ_out = res[v]
        ax.plot(Ds, occ_in, 'o-', color=color, mec=SURFACE, mew=2, label=f'{v} : lecture du banc IN (X)')
        ax.plot(Ds, occ_out, 'o--', color=color, mec=SURFACE, mew=2, label=f'{v} : écriture du banc OUT (Y)')
    ax.set_ylim(0, 100)
    ax.set_xticks(Ds)
    ax.set_xlabel('taille du GEMM D (D×D×D)')
    ax.set_ylabel('cycles de calcul avec un accès (%)')
    ax.set_title('1. Occupation des ports pendant le calcul', loc='left')
    ax.legend(loc='center right', handlelength=3)


# --- 2. Où passe le temps -------------------------------------------------------------------------------------------

PHASES = (('chargement', 'chargement par l\'hôte (W, X, b)', '#9085e9'),
          ('calcul', 'calcul', '#3a2d9a'),
          ('relecture', 'relecture de Y par l\'hôte', '#e87ba4'))


def mesure_temps(Ds=(4, 8, 16, 32, 64)):
    res = {}
    for v in ('OS', 'WS'):
        res[v] = [layer(v, D, D, D)[0] for D in Ds]
    return list(Ds), res


def plot_temps(ax, data):
    Ds, res = data
    x = np.arange(len(Ds))
    for off, v in ((-0.2, 'OS'), (0.2, 'WS')):
        tot = np.array([sum(st.values()) for st in res[v]], float)
        bottom = np.zeros(len(Ds))
        for key, label, color in PHASES:
            part = 100 * np.array([st[key] for st in res[v]]) / tot
            ax.bar(x + off, part, width=0.36, bottom=bottom, color=color, edgecolor=SURFACE, linewidth=1.5,
                   label=label if v == 'OS' else None)
            bottom += part
        for xi, t in zip(x, tot):
            ax.text(xi + off, 101, f'{v}\n{int(t)}', ha='center', va='bottom', fontsize=6.5, color=TEXT_2)
    ax.set_xticks(x)
    ax.set_xticklabels([f'D = {d}' for d in Ds])
    ax.set_ylim(0, 118)
    ax.set_ylabel('part du temps total (%)')
    ax.set_title('2. Où passe le temps (au-dessus : cycles totaux)', loc='left')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncols=3)
    ax.grid(axis='x', visible=False)


# --- 3. Enchaîner les couches ---------------------------------------------------------------------------------

def mesure_chainage(dims=(64, 48, 32, 16), batch=16):
    """MLP 64 → 48 → 32 → 16 sur un lot de 16 : calcul de chaque couche et recopie OUT → IN entre deux couches."""
    rng = np.random.default_rng(161)
    res = {}
    for v in ('OS', 'WS'):
        b = T.datapath(v)
        X = rng.integers(-128, 128, (dims[0], batch))
        couches = []
        for li, (k, m) in enumerate(zip(dims[:-1], dims[1:])):
            W = rng.integers(-128, 128, (m, k))
            Y, st = T.run_layer(b, v, W, X, [0] * m, 1, 12, 1, load_x=(li == 0))
            copie = T.copy_out_to_in(b, layout.y_addresses(m, batch, v)) if li < len(dims) - 2 else 0
            couches.append((f'{k}→{m}', st['calcul'], copie))
            X = Y
        res[v] = couches
    return res


def plot_chainage(ax, res):
    noms = [c[0] for c in res['OS']]
    x = np.arange(len(noms))
    for off, v, color in ((-0.27, 'OS', C_OS), (0, 'WS', C_WS)):
        ax.bar(x + off, [c[1] for c in res[v]], width=0.26, color=color, edgecolor=SURFACE, linewidth=1.5,
               label=f'calcul {v}')
    copies = [c[2] for c in res['OS']]
    ax.bar(x + 0.27, copies, width=0.26, color='#e87ba4', edgecolor=SURFACE, linewidth=1.5,
           label='recopie OUT → IN par l\'hôte')
    for xi, cp in zip(x, copies):
        if cp:
            ax.text(xi + 0.27, cp, str(cp), ha='center', va='bottom', fontsize=8, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels([f'couche {n}' for n in noms])
    ax.set_ylabel('cycles')
    ax.set_title('3. MLP 64→48→32→16 (lot de 16) : calcul contre recopie entre couches', loc='left')
    ax.legend(loc='upper right')
    ax.grid(axis='x', visible=False)


# --- 4. Accès totaux --------------------------------------------------------------------------------------------

TYPES = (('lect_w', 'lectures W\n(cœur)'), ('lect_x', 'lectures X\n(cœur)'), ('ecr_y', 'écritures Y\n(cœur)'),
         ('hote_ecr', 'écritures\nhôte'), ('hote_lect', 'lectures\nhôte'))


def mesure_acces(D=32):
    return D, {v: compter(layer(v, D, D, D)[1]) for v in ('OS', 'WS')}


def plot_acces(ax, data):
    D, res = data
    x = np.arange(len(TYPES))
    for off, v, color in ((-0.18, 'OS', C_OS), (0.18, 'WS', C_WS)):
        vals = [res[v][k] for k, _ in TYPES]
        bars = ax.bar(x + off, vals, width=0.34, color=color, edgecolor=SURFACE, linewidth=1.5, label=v)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(val), ha='center', va='bottom',
                    fontsize=7.5, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in TYPES])
    ax.set_ylabel('accès (mots de 32 bits)')
    ax.set_title(f'4. Accès mémoire par type, GEMM {D}×{D}×{D}', loc='left')
    ax.legend(loc='upper right')
    ax.grid(axis='x', visible=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='enregistre la figure (png) au lieu de l\'afficher')
    args = ap.parse_args()

    print('occupation des ports...')
    occ = mesure_occupation()
    print('répartition du temps (jusqu\'à D = 64)...')
    tps = mesure_temps()
    print('MLP à 3 couches...')
    ch = mesure_chainage()
    print('accès totaux...')
    acc = mesure_acces()

    print('\nRésumé')
    Ds, r = tps
    for v in ('OS', 'WS'):
        print(f'  {v} : ' + ', '.join(f'D={d} calcul {100 * st["calcul"] / sum(st.values()):.0f} % de {sum(st.values())} cycles'
                                     for d, st in zip(Ds, r[v])))
    for v in ('OS', 'WS'):
        print(f'  MLP {v} : ' + ', '.join(f'{n} calcul {c} + recopie {cp}' for n, c, cp in ch[v]))
    D, a = acc
    print(f'  accès GEMM {D}³ : OS {a["OS"]}, WS {a["WS"]}')

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    plot_occupation(axs[0, 0], occ)
    plot_temps(axs[0, 1], tps)
    plot_chainage(axs[1, 0], ch)
    plot_acces(axs[1, 1], acc)
    fig.suptitle('Benchmark de l\'activation_buffer — la mémoire dans le fonctionnement complet du NPU',
                 fontsize=13, fontweight='bold', color=TEXT)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f'\nfigure enregistrée : {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
