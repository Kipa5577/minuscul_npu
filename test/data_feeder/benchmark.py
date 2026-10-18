"""
Benchmark of the data_feeder: limits and trade-offs, as matplotlib charts.

  1. Measured latency of each lane (impulse response), N = 8
  2. Lane efficiency according to the dimension: masking wastes lanes when D is not a multiple of N
  3. Data reuse: useful MACs per byte read from SRAM (data_feeder → pe_array chain)
  4. Required SRAM bandwidth: bytes read per cycle during a tile

Run: python test/data_feeder/benchmark.py [--save file.png]
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


# --- 1. Latency per lane --------------------------------------------------------------------------

def measure_latencies(N=8):
    """Impulse on all lanes at emission 0; latency = first cycle at which the lane outputs the value."""
    res = {}
    w = [l + 1 for l in range(N)]
    for v in ('OS', 'WS'):
        h = T.run(T.bench(v, N), [dict(w=w, x=w, w_mask=(1 << N) - 1, x_mask=(1 << N) - 1, ctl=1)], extra=N + 2)
        for stream in ('w', 'x'):
            res[(v, stream)] = [next(c for c, s in enumerate(h) if s[stream][l] == w[l]) for l in range(N)]
        res[(v, 'ctl')] = next(c for c, s in enumerate(h) if s['ctl'])
    return N, res


def plot_latencies(ax, data):
    N, res = data
    lanes = np.arange(N)
    styles = [(('OS', 'w'), C_OS, '-', 'OS: W'), (('OS', 'x'), C_OS, ':', 'OS: X'),
              (('WS', 'w'), C_WS, '-', 'WS: W (preload)'), (('WS', 'x'), C_WS, ':', 'WS: X')]
    for key, color, ls, label in styles:
        off = {('OS', 'w'): -0.09, ('OS', 'x'): -0.03, ('WS', 'w'): 0.03, ('WS', 'x'): 0.09}[key]
        ax.plot(lanes + off, res[key], marker='o', ls=ls, color=color, mec=SURFACE, mew=2, label=label)
    ax.plot(lanes, 1 + lanes, color=REF, lw=1, ls='--', zorder=0)
    ax.text(N - 1, N + 0.3, '1 + l', fontsize=8, color=TEXT_2, ha='right')
    ax.set_xticks(lanes)
    ax.set_xlabel('lane l')
    ax.set_ylabel('latency from emission (cycles)')
    ax.set_title(f'1. Measured latency of each lane (N = {N})', loc='left')
    ax.legend(loc='upper left', handlelength=3)


# --- 2. Lane efficiency (masking) ---------------------------------------------------------------------

def measure_efficiency(Ns=(2, 4, 8), Dmax=64):
    """Sends the tiles of a dimension D (one word per tile) and counts the unmasked lanes at the output."""
    res = {}
    for N in Ns:
        eff = []
        for D in range(1, Dmax + 1):
            tiles = -(-D // N)
            em = [dict(w=[1] * N, x=[1] * N, w_mask=T.mask_of(D - N * t, N), x_mask=0, ctl=0) for t in range(tiles)]
            h = T.run(T.bench('WS', N), em, extra=2)          # W not skewed: one tile = one output cycle
            useful = sum(sum(1 for v in s['w'] if v) for s in h)
            eff.append(100 * useful / (N * tiles))
        res[N] = eff
    return list(range(1, Dmax + 1)), res


def plot_efficiency(ax, data):
    Ds, res = data
    for N, color in zip(res, ('#9085e9', '#6250d6', '#3a2d9a')):
        ax.plot(Ds, res[N], color=color, lw=1.6, label=f'N = {N}')
    ax.set_ylim(0, 105)
    ax.set_xlim(1, Ds[-1])
    ax.set_xlabel('dimension D of the matrix (M, K or N)')
    ax.set_ylabel('useful lanes (%)')
    ax.set_title('2. Lanes wasted by edge masking', loc='left')
    ax.legend(loc='lower right')


# --- 3 and 4. Reuse and SRAM bandwidth (data_feeder → pe_array chain) --------------------------------

def tile(variant, N, L, rng):
    """One complete tile. Returns (useful MACs, bytes read from SRAM, cycles from the first read to the last MAC)."""
    b = T.chain(variant, N)
    full = (1 << N) - 1
    if variant == 'OS':
        W, X = rng.integers(1, 128, (N, L)), rng.integers(1, 128, (L, N))
        em = [dict(w=[0] * N, x=[0] * N, w_mask=0, x_mask=0, ctl=1)]
        em += [dict(w=list(W[:, k]), x=list(X[k]), w_mask=full, x_mask=full, ctl=0) for k in range(L)]
        nbytes = 2 * N * L
        active = lambda pe: pe.w_in.get() != 0 and pe.x_in.get() != 0
    else:
        W, X = rng.integers(1, 128, (N, N)), rng.integers(1, 128, (N, L))
        em = [dict(w=list(W[:, k]), x=[0] * N, w_mask=full, x_mask=0, ctl=1) for k in reversed(range(N))]
        em += [dict(w=[0] * N, x=list(X[:, j]), w_mask=0, x_mask=full, ctl=0) for j in range(L)]
        nbytes = N * N + N * L
        active = lambda pe: pe.x_in.get() != 0 and pe.w_r != 0 and not pe.load.get()
    idle = dict(w=[0] * N, x=[0] * N, w_mask=0, x_mask=0, ctl=0)
    prev, macs, last = idle, 0, 0
    first_read = 0 if variant == 'WS' else 1
    for e in range(len(em) + 3 * N):
        cur = em[e] if e < len(em) else idle
        b.set(w_rdata=pack(prev['w'], DW), x_rdata=pack(prev['x'], DW),
              w_mask=cur['w_mask'], x_mask=cur['x_mask'], ctl_in=cur['ctl'])
        b.sim.propagateAll()
        n = sum(active(pe) for row in b.dut.pes for pe in row)
        if n:
            macs, last = macs + n, e
        b.sim.clk(1)
        prev = cur
    return macs, nbytes, last + 1 - first_read


def measure_reuse(Ns=(2, 4, 8), L=64):
    rng = np.random.default_rng(40)
    res = {}
    for v in ('OS', 'WS'):
        for N in Ns:
            macs, nbytes, cycles = tile(v, N, L, rng)
            assert macs == N * N * L, (v, N, macs)
            res[(v, N)] = dict(macs_per_byte=macs / nbytes, bytes_per_cycle=nbytes / cycles,
                               macs=macs, nbytes=nbytes, cycles=cycles)
    return list(Ns), L, res


def plot_bars(ax, data, key, ylabel, title, fmt):
    Ns, L, res = data
    x = np.arange(len(Ns))
    for off, (v, color) in zip((-0.18, 0.18), (('OS', C_OS), ('WS', C_WS))):
        vals = [res[(v, N)][key] for N in Ns]
        bars = ax.bar(x + off, vals, width=0.34, color=color, edgecolor=SURFACE, linewidth=2, label=v)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(val),
                    ha='center', va='bottom', fontsize=8, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels([f'N = {n}' for n in Ns])
    ax.set_ylabel(ylabel)
    ax.set_title(title.format(L=L), loc='left')
    ax.legend(loc='upper left')
    ax.grid(axis='x', visible=False)


# --- Main -----------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='saves the figure (png) instead of showing it')
    args = ap.parse_args()

    print('measuring the latencies...')
    lat = measure_latencies()
    print('measuring the lane efficiency...')
    eff = measure_efficiency()
    print('measuring the reuse and the SRAM bandwidth...')
    reuse = measure_reuse()

    N, r = lat
    print('\nSummary')
    for k in (('OS', 'w'), ('OS', 'x'), ('WS', 'w'), ('WS', 'x')):
        print(f'  latency {k[0]} {k[1].upper()} (N={N}): {r[k]}')
    print(f'  latency ctl: OS {r[("OS", "ctl")]}, WS {r[("WS", "ctl")]}')
    Ds, e = eff
    for n in e:
        print(f'  average lane efficiency N={n} (D = 1..{Ds[-1]}): {np.mean(e[n]):.0f} %')
    Ns, L, res = reuse
    for (v, n), d in res.items():
        print(f'  {v} N={n}: {d["macs"]} MACs, {d["nbytes"]} bytes read, {d["cycles"]} cycles '
              f'→ {d["macs_per_byte"]:.1f} MACs/byte, {d["bytes_per_cycle"]:.1f} bytes/cycle')

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    plot_latencies(axs[0, 0], lat)
    plot_efficiency(axs[0, 1], eff)
    plot_bars(axs[1, 0], reuse, 'macs_per_byte', 'useful MACs per byte read',
                '3. Data reuse (tile, stream of {L})', '{:.1f}')
    plot_bars(axs[1, 1], reuse, 'bytes_per_cycle', 'bytes read from SRAM per cycle',
                '4. Required SRAM bandwidth (tile, stream of {L})', '{:.1f}')
    fig.suptitle('data_feeder benchmark — output-stationary vs weight-stationary', fontsize=13,
                 fontweight='bold', color=TEXT)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f'\nfigure saved: {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
