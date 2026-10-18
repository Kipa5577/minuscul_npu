"""
Benchmark of the pe_array: limits and trade-offs, as matplotlib charts.

  1. Wavefront: number of PEs doing a useful MAC at each cycle (one tile, N = 4)
  2. Cycles for an N×16×N tile as a function of N (measured vs formula)
  3. PE utilisation as a function of the streamed dimension (K in OS, J in WS)
  4. py4hw simulation speed as a function of N

For one tile, OS and WS do exactly the same work: N·N·L MACs
(OS: W N×L, X L×N; WS: W N×N preloaded, X N×L).

Run: python test/pe_array/benchmark.py [--save file.png]
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


# --- Style (same as the pe benchmark) --------------------------------------------------

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


# --- Running a tile while observing the PEs --------------------------------------------

def step_observe(b, active, **values):
    """Presents the inputs, counts the active PEs during this cycle, then advances by one edge."""
    b.set(**values)
    b.sim.propagateAll()
    n = sum(active(pe) for row in b.dut.pes for pe in row)
    b.sim.clk(1)
    b.cycle += 1
    return n


active_os = lambda pe: pe.w_in.get() != 0 and pe.x_in.get() != 0 and not pe.clear.get()
active_ws = lambda pe: pe.x_in.get() != 0 and pe.w_r != 0 and not pe.load.get()


def tile_os(N, L, rng):
    """One OS tile (K = L). Returns (active PEs per cycle, number of cycles until the complete result)."""
    W = rng.integers(1, 128, (N, L))          # non-zero values: every injected MAC is useful
    X = rng.integers(1, 128, (L, N))
    Y = W @ X
    b = T.bench_os(N)
    active, end = [], None
    for c in range(L + 2 * N + 1):
        wl, xl = T.os_lanes(W, X, c)
        active.append(step_observe(b, active_os, w_west=wl, x_north=xl, clear=0))
        if end is None and (T.acc_matrix(b) == Y).all():
            end = c + 1
    return active, end


def tile_ws(N, L, rng):
    """Preload + one WS tile (J = L). Returns (active PEs per cycle, cycles until the last output)."""
    W = rng.integers(1, 128, (N, N))
    X = rng.integers(1, 128, (N, L))
    Y = W @ X
    b = T.bench_ws(N)
    active = []
    for k in reversed(range(N)):
        active.append(step_observe(b, active_ws, w_north=pack(W[:, k], DW), x_west=0, load=1))
    end, out = None, []
    for c in range(L + 2 * N + 1):
        active.append(step_observe(b, active_ws, x_west=T.ws_x_lanes(X, c), w_north=0, load=0))
        out.append(T.psum_south(b)[N - 1])
        # end = the cycle at which lane N−1 has just output the whole row Y[N−1][0..L−1], in order
        if end is None and out[-L:] == list(Y[N - 1]):
            end = N + c + 1
    return active, end


# --- Measurements ----------------------------------------------------------------------

def measure_wavefront(N=4, L=16):
    rng = np.random.default_rng(20)
    return tile_os(N, L, rng)[0], tile_ws(N, L, rng)[0], N, L


def measure_cycles_vs_N(L=16, Ns=range(1, 9)):
    rng = np.random.default_rng(21)
    return list(Ns), [tile_os(N, L, rng)[1] for N in Ns], [tile_ws(N, L, rng)[1] for N in Ns], L


def measure_utilisation(Ls=(1, 2, 4, 8, 16, 32, 64), Ns=(4, 8)):
    rng = np.random.default_rng(22)
    res = {}
    for N in Ns:
        for var, f in (('OS', tile_os), ('WS', tile_ws)):
            res[(var, N)] = [100 * L / f(N, L, rng)[1] for L in Ls]      # N²·L MACs / (N²·cycles)
    return list(Ls), res


def measure_speed(Ns=(2, 4, 8), cycles=1500):
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


# --- Charts --------------------------------------------------------------------------------

def plot_wavefront(ax, data):
    a_os, a_ws, N, L = data
    ax.step(range(len(a_os)), a_os, where='mid', color=C_OS, label='OS')
    ax.step(range(len(a_ws)), a_ws, where='mid', color=C_WS, label='WS')
    ax.axvspan(-0.5, N - 0.5, color=C_WS, alpha=0.08, lw=0)
    ax.text(N / 2 - 0.5, N * N + 0.6, 'WS\npreload', ha='center', fontsize=7, color=TEXT_2)
    ax.axhline(N * N, color=REF, lw=1, ls='--')
    ax.text(len(a_ws) - 0.5, N * N - 0.9, f'{N * N} PEs', ha='right', fontsize=7, color=TEXT_2)
    ax.set_ylim(0, N * N + 2.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('cycle')
    ax.set_ylabel('PEs doing a useful MAC')
    ax.set_title(f'1. Wavefront of a tile (N = {N}, stream of {L})', loc='left')
    ax.legend(loc='upper right')


def plot_cycles_vs_N(ax, data):
    Ns, c_os, c_ws, L = data
    n = np.array(Ns)
    ax.plot(n, L + 2 * n - 2, color=C_OS, lw=1, ls='--', alpha=0.7)
    ax.plot(n, L + 3 * n - 2, color=C_WS, lw=1, ls='--', alpha=0.7)
    ax.plot(Ns, c_os, 'o', color=C_OS, mec=SURFACE, mew=2, label='OS measured')
    ax.plot(Ns, c_ws, 'o', color=C_WS, mec=SURFACE, mew=2, label='WS measured')
    xo = Ns[-1] - 1.5
    ax.text(xo, L + 2 * xo - 2 - 2.5, f'OS: {L} + 2N − 2', ha='left', va='top', fontsize=8, color=TEXT_2)
    ax.annotate(f'WS: N + {L} + 2N − 2', (Ns[-1], L + 3 * Ns[-1] - 2), xytext=(-6, 6),
                textcoords='offset points', ha='right', fontsize=8, color=TEXT_2)
    ax.axhline(L, color=REF, lw=1, ls=':')
    ax.text(Ns[0], L + 0.5, f'ideal: {L} cycles', fontsize=7, color=TEXT_2)
    ax.set_xlabel('array side N')
    ax.set_ylabel('cycles until the last result')
    ax.set_title(f'2. Cycles for a tile (stream of {L}), measurement vs formula', loc='left')
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
    ax.set_xlabel('stream length (K in OS, J in WS)')
    ax.set_ylabel('PE utilisation (%)')
    ax.set_title('3. PE utilisation for an isolated tile', loc='left')
    ax.legend(loc='upper left', handlelength=3.5)


def plot_speed(ax, data):
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
    ax.set_ylabel('thousands of simulated cycles per second')
    ax.set_title('4. py4hw simulation speed', loc='left')
    ax.legend(loc='upper right')
    ax.grid(axis='x', visible=False)


# --- Main -----------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='saves the figure (png) instead of showing it')
    args = ap.parse_args()

    print('measuring the wavefront...')
    fo = measure_wavefront()
    print('measuring the cycles as a function of N...')
    cn = measure_cycles_vs_N()
    print('measuring the utilisation...')
    ut = measure_utilisation()
    print('measuring the simulation speed...')
    vi = measure_speed()

    Ns, c_os, c_ws, L = cn
    print('\nSummary')
    print(f'  cycles (stream {L}): OS {dict(zip(Ns, c_os))}')
    print(f'                     WS {dict(zip(Ns, c_ws))}')
    Ls, res = ut
    for (var, N), u in res.items():
        print(f'  utilisation {var} N={N}: ' + ', '.join(f'{l}→{x:.0f} %' for l, x in zip(Ls, u)))
    print('  speed              : ' + ', '.join(f'{v} N={n} {s:,.0f} c/s'
                                                 for v in vi[1] for n, s in zip(vi[0], vi[1][v])))

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    plot_wavefront(axs[0, 0], fo)
    plot_cycles_vs_N(axs[0, 1], cn)
    plot_utilisation(axs[1, 0], ut)
    plot_speed(axs[1, 1], vi)
    fig.suptitle('pe_array benchmark — output-stationary vs weight-stationary', fontsize=13,
                 fontweight='bold', color=TEXT)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f'\nfigure saved: {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
