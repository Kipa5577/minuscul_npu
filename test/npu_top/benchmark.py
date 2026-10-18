"""
Benchmark de npu_top : la puce complète sur de vraies charges, pilotée uniquement par son bus.

  1. Un réseau de neurones en virgule flottante (MLP 64 → 32 → 16, lot de 8), quantifié en int8 et exécuté sur la
     puce couche par couche (Y recopié vers ACT_IN entre deux couches), comparé au calcul en virgule flottante
  2. Taille du lot : cycles totaux (hôte compris) d'une couche 64 → 64 selon le nombre d'échantillons, OS contre WS
  3. Activité de chaque bloc pendant une couche 32×32×32, chargement et relecture compris
  4. Chronogramme de cette activité pour une couche 16×16×16 (OS)

Lancement : python test/npu_top/benchmark.py [--save fichier.png]
"""

import sys
import math
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_test_module
from viz import BENCH_RC, SURFACE, TEXT, TEXT_2, REF, C_OS, C_WS

import numpy as np
import matplotlib.pyplot as plt

T = load_test_module('npu_top')
plt.rcParams.update(BENCH_RC)


def mult_shift(scale, mw=16):
    s = min(31, max(0, math.floor(math.log2(((1 << mw) - 1) / scale))))
    return min((1 << mw) - 1, max(1, round(scale * 2 ** s))), s


def quantifier(a):
    s = np.abs(a).max() / 127
    return np.clip(np.round(a / s), -127, 127).astype(int), s


# --- 1. Réseau en virgule flottante sur la puce -------------------------------------------------------------------

def mesure_reseau(seed=210):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (64, 8))
    W1, b1 = rng.normal(0, 1 / 8, (32, 64)), rng.normal(0, 0.1, 32)
    W2, b2 = rng.normal(0, 1 / 6, (16, 32)), rng.normal(0, 0.1, 16)
    H = np.maximum(0, W1 @ X + b1[:, None])
    Yf = W2 @ H + b2[:, None]
    Xq, sx = quantifier(X)
    W1q, sw1 = quantifier(W1)
    W2q, sw2 = quantifier(W2)
    sh, sy = H.max() / 127, np.abs(Yf).max() / 127
    res = {}
    for v in ('OS', 'WS'):
        npu = T.Npu(v)
        c0 = npu.cycle
        m1, s1 = mult_shift(sw1 * sx / sh)
        Hq, _, _ = npu.gemm(W1q, Xq, list(np.round(b1 / (sw1 * sx)).astype(int)), m1, s1, True)
        npu.copy_out_to_in(32, 8)
        m2, s2 = mult_shift(sw2 * sh / sy)
        Yq, _, _ = npu.gemm(W2q, Hq, list(np.round(b2 / (sw2 * sh)).astype(int)), m2, s2, False, load_x=False)
        res[v] = (Yq * sy, npu.cycle - c0)
    return Yf, res


def plot_reseau(ax, data):
    Yf, res = data
    lim = np.abs(Yf).max() * 1.1
    ax.plot([-lim, lim], [-lim, lim], color=REF, lw=1, ls='--')
    ax.scatter(Yf.flatten(), res['OS'][0].flatten(), s=24, color=C_OS, edgecolor=SURFACE, lw=0.6, zorder=3,
               label=f'OS ({res["OS"][1]} cycles de bus)')
    ax.scatter(Yf.flatten(), res['WS'][0].flatten(), s=8, color=C_WS, zorder=4,
               label=f'WS, identique ({res["WS"][1]} cycles de bus)')
    err = res['OS'][0] - Yf
    corr = np.corrcoef(Yf.flatten(), res['OS'][0].flatten())[0, 1]
    ax.text(0.03, 0.97, f'corrélation {corr:.4f}\nerreur RMS {np.sqrt(np.mean(err ** 2)):.3f} '
            f'(sorties ∈ ±{np.abs(Yf).max():.1f})', transform=ax.transAxes, va='top', fontsize=8, color=TEXT)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('sortie du réseau en virgule flottante')
    ax.set_ylabel('sortie de la puce (int8 déquantifié)')
    ax.set_title('1. MLP 64→32→16 exécuté sur la puce, par le bus', loc='left')
    ax.legend(loc='lower right')


# --- 2. Taille du lot ---------------------------------------------------------------------------------------------

def mesure_lot(lots=(1, 2, 4, 8, 16, 32, 64)):
    rng = np.random.default_rng(211)
    W = rng.integers(-128, 128, (64, 64))
    res = {}
    for v in ('OS', 'WS'):
        res[v] = []
        for n in lots:
            npu = T.Npu(v)
            c0 = npu.cycle
            npu.gemm(W, rng.integers(-128, 128, (64, n)), [0] * 64, 1, 12, True)
            res[v].append(npu.cycle - c0)
    return list(lots), res


def plot_lot(ax, data):
    lots, res = data
    for v, color in (('OS', C_OS), ('WS', C_WS)):
        par = [c / n for c, n in zip(res[v], lots)]
        ax.plot(lots, par, 'o-', color=color, mec=SURFACE, mew=2, label=v)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(lots)
    ax.set_xticklabels(lots)
    ax.set_xlabel('taille du lot (colonnes de X)')
    ax.set_ylabel('cycles par échantillon (hôte compris)')
    ax.set_title('2. Couche 64→64 : coût par échantillon selon la taille du lot', loc='left')
    ax.legend(loc='upper right')


# --- 3 et 4. Activité de chaque bloc ------------------------------------------------------------------------------

BLOCS = ['bus hôte', 'controller', 'weight_buffer', 'activation_buffer', 'data_feeder', 'pe_array',
         'output_drain', 'post_processing']


def sonde(npu):
    """Remplace le pas du banc pour relever, à chaque cycle, quels blocs de la puce travaillent."""
    b, d = npu.b, npu.b.dut
    trace = []

    def actif_pe(pe):
        if d.dataflow == 'OS':
            return pe.w_in.get() and pe.x_in.get()
        return pe.x_in.get() and pe.w_r          # un MAC peut avoir lieu pendant le premier cycle de rechargement

    def step(**v):
        b.set(**v)
        b.sim.propagateAll()
        wb, ab = d.weight_buffer, d.activation_buffer
        trace.append({
            'bus hôte': d.host.bus_we.get() or d.host.bus_re.get(),
            'controller': d.controller.busy.get(),
            'weight_buffer': (wb.busy.get() and wb.core_re.get()) or (not wb.busy.get() and (wb.host_we.get() or wb.host_re.get())),
            'activation_buffer': (ab.busy.get() and (ab.core_rd_re.get() or ab.core_wr_we.get()))
                                 or (not ab.busy.get() and (ab.host_we.get() or ab.host_re.get())),
            'data_feeder': d.data_feeder.w_arr.get() or d.data_feeder.x_arr.get(),
            'pe_array': any(actif_pe(pe) for row in d.pe_array.pes for pe in row),
            'output_drain': d.output_drain.res_valid.get(),
            'post_processing': d.post_processing.y_we.get()})
        b.sim.clk(1)
        b.cycle += 1

    b.step = step
    return trace


def mesure_activite(D=32):
    rng = np.random.default_rng(212)
    W, X = rng.integers(1, 128, (D, D)), rng.integers(1, 128, (D, D))
    res = {}
    for v in ('OS', 'WS'):
        npu = T.Npu(v)
        trace = sonde(npu)
        npu.gemm(W, X, [0] * D, 1, 12, True)
        res[v] = {k: 100 * np.mean([bool(t[k]) for t in trace]) for k in BLOCS}
    return D, res


def plot_activite(ax, data):
    D, res = data
    y = np.arange(len(BLOCS))
    for off, v, color in ((-0.2, 'OS', C_OS), (0.2, 'WS', C_WS)):
        vals = [res[v][k] for k in BLOCS]
        ax.barh(y + off, vals, height=0.38, color=color, edgecolor=SURFACE, linewidth=1, label=v)
        for yi, val in zip(y, vals):
            ax.text(val + 1, yi + off, f'{val:.0f} %', va='center', fontsize=7, color=TEXT)
    ax.set_yticks(y)
    ax.set_yticklabels(['bus hôte\n(scrutation comprise)'] + BLOCS[1:])
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel('cycles où le bloc travaille (%)')
    ax.set_title(f'3. Activité de chaque bloc, couche {D}×{D}×{D} (hôte compris)', loc='left')
    ax.legend(loc='lower right')
    ax.grid(axis='y', visible=False)


def mesure_chrono(D=16):
    rng = np.random.default_rng(213)
    npu = T.Npu('OS')
    trace = sonde(npu)
    npu.gemm(rng.integers(1, 128, (D, D)), rng.integers(1, 128, (D, D)), [0] * D, 1, 12, True)
    return D, trace


def plot_chrono(ax, data):
    D, trace = data
    couleurs = ['#52514e', '#9085e9', '#6250d6', '#1baf7a', '#eda100', '#3a2d9a', '#e87ba4', '#08402c']
    for i, (k, col) in enumerate(zip(BLOCS, couleurs)):
        on = np.array([bool(t[k]) for t in trace])
        debut = np.flatnonzero(np.diff(np.concatenate(([0], on.astype(int)))) == 1)
        fin = np.flatnonzero(np.diff(np.concatenate((on.astype(int), [0]))) == -1) + 1
        ax.broken_barh([(a, b - a) for a, b in zip(debut, fin)], (i - 0.35, 0.7), color=col)
    ax.set_yticks(range(len(BLOCS)))
    ax.set_yticklabels(BLOCS)
    ax.invert_yaxis()
    ax.set_xlim(0, len(trace))
    ax.set_xlabel('cycle')
    ax.set_title(f'4. Chronogramme de la puce, couche {D}×{D}×{D} en OS', loc='left')
    ax.grid(axis='y', visible=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='enregistre la figure (png) au lieu de l\'afficher')
    args = ap.parse_args()

    print('MLP flottant quantifié, sur la puce...')
    rs = mesure_reseau()
    print('taille du lot...')
    lot = mesure_lot()
    print('activité des blocs...')
    act = mesure_activite()
    chrono = mesure_chrono()

    Yf, r = rs
    print('\nRésumé')
    print(f'  MLP : OS et WS identiques = {bool((r["OS"][0] == r["WS"][0]).all())}, corrélation '
          f'{np.corrcoef(Yf.flatten(), r["OS"][0].flatten())[0, 1]:.4f}, cycles OS {r["OS"][1]} / WS {r["WS"][1]}')
    lots, l = lot
    print('  cycles par échantillon (couche 64→64) : ' + ', '.join(
        f'lot {n} OS {a / n:.0f} / WS {b / n:.0f}' for n, a, b in zip(lots, l['OS'], l['WS'])))
    D, a = act
    for v in ('OS', 'WS'):
        print(f'  activité {v} ({D}³) : ' + ', '.join(f'{k} {a[v][k]:.0f} %' for k in BLOCS))

    fig, axs = plt.subplots(2, 2, figsize=(15, 10.5))
    plot_reseau(axs[0, 0], rs)
    plot_lot(axs[0, 1], lot)
    plot_activite(axs[1, 0], act)
    plot_chrono(axs[1, 1], chrono)
    fig.suptitle('Benchmark de npu_top — la puce complète sur de vraies charges', fontsize=13, fontweight='bold',
                 color=TEXT)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f'\nfigure enregistrée : {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
