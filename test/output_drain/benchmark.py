"""
Benchmark of the output_drain: limits and trade-offs, as matplotlib charts.

With the output_drain, the compute core (data_feeder → pe_array → output_drain) is complete:
measurements 2 to 4 are made end to end, on GEMMs driven by the scheduler of the test
(what the controller FSM emits), and Y is checked every time.

  1. Measured latency: OS (drain_en → res_valid) and WS (emission of X → res_valid) as a function of N
  2. Output rhythm of the results on a 16×16×16 GEMM: bursts (OS) versus continuous stream (WS)
  3. Array utilisation as a function of the GEMM size (D×D×D)
  4. Breakdown of the cycles per FSM phase on a 16×16×16 GEMM

Run: python test/output_drain/benchmark.py [--save file.png]
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


# --- 1. Latency -----------------------------------------------------------------------------------------

def measure_latencies(Ns=range(1, 9)):
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


def plot_latencies(ax, data):
    Ns, os_lat, ws_lat = data
    n = np.array(Ns)
    ax.plot(n, 2 * n + 1, color=C_WS, lw=1, ls='--', alpha=0.7)
    ax.plot(Ns, ws_lat, 'o', color=C_WS, mec=SURFACE, mew=2, label='WS: emission of X → res_valid')
    ax.plot(Ns, os_lat, 'o-', color=C_OS, mec=SURFACE, mew=2, label='OS: drain_en → res_valid')
    ax.text(Ns[-1] - 0.2, 2 * Ns[-1] + 1 + 0.6, '2N + 1', ha='right', fontsize=8, color=TEXT_2)
    ax.set_xlabel('array side N')
    ax.set_ylabel('latency (cycles)')
    ax.set_title('1. Measured latency of the output_drain', loc='left')
    ax.legend(loc='upper left')


# --- End-to-end GEMM ---------------------------------------------------------------------------------------

def gemm(variant, D, seed):
    """End-to-end D×D×D GEMM; returns (plan, outputs, cycles until the last result)."""
    ref, Y, count, out, em = T.gemm_end_to_end(variant, D, D, D, seed)
    assert (count == 1).all() and (Y == ref).all(), (variant, D)
    return em, out, out[-1][0]


def measure_rhythm(D=16):
    res = {}
    for v in ('OS', 'WS'):
        em, out, end = gemm(v, D, 80)
        cycles = [c for c, tag, _ in out]
        elems = np.cumsum([bin(T.tag_unpack(tag)[1]).count('1') for _, tag, _ in out])
        res[v] = (cycles, elems, end, em)
    return D, res


def plot_rhythm(ax, data):
    D, res = data
    for v, color in (('OS', C_OS), ('WS', C_WS)):
        cycles, elems, end, _ = res[v]
        ax.step([0] + cycles, [0] + list(elems), where='post', color=color, label=f'{v}: ends at cycle {end}')
    ax.axhline(D * D, color=REF, lw=1, ls='--')
    ax.text(0, D * D + 5, f'{D}×{D} = {D * D} elements of Y', fontsize=8, color=TEXT_2)
    ax.set_xlabel('cycle')
    ax.set_ylabel('elements of Y output (cumulative)')
    ax.set_title(f'2. Output rhythm, {D}×{D}×{D} GEMM', loc='left')
    ax.legend(loc='center right')


def measure_utilisation(Ds=(4, 8, 12, 16, 20, 24, 32), N=4):
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
    ax.set_xlabel('GEMM size D (D×D×D)')
    ax.set_ylabel('array utilisation (%)')
    ax.set_title('3. End-to-end utilisation (complete compute core, N = 4)', loc='left')
    ax.legend(loc='lower right')


PHASES = [('stream', 'stream (useful MACs possible)', '#3a2d9a'), ('clear', 'clear', '#9085e9'),
          ('preload', 'preload', '#eda100'), ('flush', 'array flush', '#b9b8b2'),
          ('drain', 'reading the accumulators', '#e87ba4'), ('tail', 'pipeline tail', '#e4e3df')]


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
                        label=label if v == 'OS' or key == 'preload' else None)
                if n / len(em) > 0.06:
                    ax.text(left + n / 2, y[v], f'{100 * n / len(em):.0f} %', ha='center', va='center',
                            fontsize=8, color='white' if key == 'stream' else TEXT)
                left += n
    ax.set_yticks([1, 0])
    ax.set_yticklabels(['OS', 'WS'])
    ax.set_xlabel('cycles')
    ax.set_title(f'4. Breakdown of the cycles per phase, {D}×{D}×{D} GEMM', loc='left')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncols=3)
    ax.grid(axis='y', visible=False)
    ax.set_ylim(-0.6, 1.6)


# --- Main -----------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='saves the figure (png) instead of showing it')
    args = ap.parse_args()

    print('measuring the latencies...')
    lat = measure_latencies()
    print('end-to-end 16×16×16 GEMM...')
    ry = measure_rhythm()
    print('utilisation as a function of the GEMM size...')
    ut = measure_utilisation()

    Ns, ol, wl = lat
    print('\nSummary')
    print(f'  latency OS: {dict(zip(Ns, ol))}')
    print(f'  latency WS: {dict(zip(Ns, wl))}')
    D, r = ry
    for v in ('OS', 'WS'):
        em, end = r[v][3], r[v][2]
        phases = {k: sum(1 for e in em if e['phase'] == k) for k, _, _ in PHASES}
        print(f'  GEMM {D}³ {v}: last result at cycle {end}; phases {phases}')
    Ds, u = ut
    for v in ('OS', 'WS'):
        print(f'  utilisation {v}: ' + ', '.join(f'D={d} {x:.0f} %' for d, x in zip(Ds, u[v])))

    fig, axs = plt.subplots(2, 2, figsize=(14, 9.5))
    plot_latencies(axs[0, 0], lat)
    plot_rhythm(axs[0, 1], ry)
    plot_utilisation(axs[1, 0], ut)
    plot_phases(axs[1, 1], ry)
    fig.suptitle('Benchmark of the output_drain and of the compute core — output-stationary vs weight-stationary',
                 fontsize=13, fontweight='bold', color=TEXT)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f'\nfigure saved: {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
