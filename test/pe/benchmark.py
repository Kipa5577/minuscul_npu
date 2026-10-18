"""
Benchmark of the pe: limits and trade-offs, as matplotlib charts.

  1. Impulse response: latency of each path, in cycles
  2. Overflow limit of the accumulator: theoretical vs measured max K
  3. Trade-off accumulator width / rate of wrong results on random data
  4. py4hw simulation speed

Run: python test/pe/benchmark.py [--save file.png]
"""

import sys
import time
import argparse
import pathlib
import importlib.util

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import common  # noqa: F401  (configure sys.path)

import numpy as np
import matplotlib.pyplot as plt

_spec = importlib.util.spec_from_file_location('pe_test', pathlib.Path(__file__).with_name('test.py'))
pe_test = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pe_test)
bench_os, bench_ws = pe_test.bench_os, pe_test.bench_ws


# --- Style ----------------------------------------------------------------------------

SURFACE = '#fcfcfb'
TEXT = '#0b0b0b'
TEXT_2 = '#52514e'
GRID = '#e4e3df'
C_OS = '#2a78d6'        # series 1: output-stationary
C_WS = '#eb6834'        # series 2: weight-stationary
C_INT4 = '#1baf7a'      # panel 2: int4
C_INT8 = '#6250d6'      # panel 2: int8
SEQ_K = ('#9085e9', '#6250d6', '#3a2d9a')   # panel 3: increasing K (light → dark)
REF = '#8a8984'         # reference lines

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': TEXT_2, 'axes.titlecolor': TEXT,
    'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.labelsize': 9,
    'xtick.color': TEXT_2, 'ytick.color': TEXT_2, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 2, 'lines.markersize': 7,
    'legend.fontsize': 8, 'legend.frameon': False, 'text.color': TEXT,
})


# --- 1. Impulse response ------------------------------------------------------------

def impulse_response(b, out, impulse, idle, cycles=5):
    """Applies `impulse` at cycle 0 then `idle`; returns the value of `out` after each edge."""
    trace = []
    b.step(**impulse)
    trace.append(b.get(out))
    for _ in range(cycles - 1):
        b.step(**idle)
        trace.append(b.get(out))
    return trace


def measure_latencies():
    paths = [
        ('OS', 'w_in → w_out', bench_os, 'w_out', dict(w_in=1, clear=0), dict(w_in=0)),
        ('OS', 'x_in → x_out', bench_os, 'x_out', dict(x_in=1, clear=0), dict(x_in=0)),
        ('OS', 'w·x → acc', bench_os, 'acc', dict(w_in=1, x_in=1, clear=0), dict(w_in=0, x_in=0)),
        ('WS', 'x_in → x_out', bench_ws, 'x_out', dict(x_in=1, load=0), dict(x_in=0)),
        ('WS', 'psum_in → psum_out', bench_ws, 'psum_out', dict(psum_in=1, load=0), dict(psum_in=0)),
        ('WS', 'w_load_in → w_load_out', bench_ws, 'w_load_out', dict(w_load_in=1, load=1), dict(w_load_in=0, load=0)),
    ]
    res = []
    for var, name, mk, out, imp, idle in paths:
        trace = impulse_response(mk(), out, imp, idle)
        lat = 1 + next(i for i, v in enumerate(trace) if v != 0)
        res.append((var, name, trace, lat))
    return res


def plot_latencies(ax, res):
    n = len(res)
    for row, (var, name, trace, lat) in enumerate(res):
        y0 = n - 1 - row
        color = C_OS if var == 'OS' else C_WS
        cycles = np.arange(len(trace) + 1)
        level = [0] + [1 if v else 0 for v in trace]          # 0 before edge 1
        ax.step(cycles, y0 + 0.6 * np.array(level), where='post', color=color)
        ax.text(-0.15, y0 + 0.2, f'{var}  {name}', ha='right', va='center', fontsize=8, color=TEXT)
        ax.text(lat + 0.1, y0 + 0.72, f'L = {lat}', fontsize=8, color=TEXT_2)
    ax.axvline(0, color=REF, lw=1, ls='--')
    ax.text(0.05, n - 0.2, 'input\nimpulse', fontsize=7, color=TEXT_2, va='top')
    ax.set_xlim(-0.2, len(res[0][2]))
    ax.set_ylim(-0.3, n)
    ax.set_yticks([])
    ax.set_xlabel('clock edge after the impulse')
    ax.set_title('1. Impulse response (latency of each path)', loc='left')
    ax.grid(axis='y', visible=False)


# --- 2. Overflow ------------------------------------------------------------------------

def k_max_theory(DW, AW):
    """Largest K such that no sequence of K products overflows (worst case)."""
    p_pos = (1 << (DW - 1)) ** 2                          # (-2^(DW-1))²
    p_neg = (1 << (DW - 1)) * ((1 << (DW - 1)) - 1)       # |(-2^(DW-1))·(2^(DW-1)-1)|
    return min(((1 << (AW - 1)) - 1) // p_pos, (1 << (AW - 1)) // p_neg)


def k_max_measured(DW, AW):
    """Accumulates the worst positive product until overflow; returns the last correct K."""
    b = bench_os(DW, AW)
    m = -(1 << (DW - 1))
    b.step(clear=1)
    k = 0
    while True:
        b.step(w_in=m, x_in=m, clear=0)
        if b.get('acc') != (k + 1) * m * m:
            return k
        k += 1


def measure_overflow():
    res = {}
    for DW, aws in [(4, range(8, 25, 2)), (8, range(16, 33, 2))]:
        th = [k_max_theory(DW, aw) for aw in aws]
        me = [k_max_measured(DW, aw) if k_max_theory(DW, aw) <= 200_000 else None for aw in aws]
        res[DW] = (list(aws), th, me)
    return res


def plot_overflow(ax, res, dim_max=64):
    for DW, color in [(8, C_INT8), (4, C_INT4)]:
        aws, th, me = res[DW]
        ax.plot(aws, th, color=color, label=f'int{DW}: theory')
        pts = [(a, m) for a, m in zip(aws, me) if m is not None]
        ax.plot(*zip(*pts), 'o', color=color, mec=SURFACE, mew=2, label=f'int{DW}: measured')
        ax.annotate(f'int{DW}', (aws[-1], th[-1]), xytext=(4, 0), textcoords='offset points',
                    fontsize=8, color=TEXT, va='center')
    ax.axhline(dim_max, color=REF, lw=1, ls='--')
    ax.text(8, dim_max * 1.4, f'K = DIM_MAX = {dim_max}', fontsize=7, color=TEXT_2)
    ax.axvline(32, color=REF, lw=1, ls=':')
    ax.text(31.6, 3, 'int32 (v1)', fontsize=7, color=TEXT_2, ha='right')
    ax.set_yscale('log', base=2)
    ax.set_xlabel('accumulator width AW (bits)')
    ax.set_ylabel('max K without overflow (worst case)')
    ax.set_title('2. Overflow limit of the accumulator', loc='left')
    ax.legend(loc='upper left')


# --- 3. Accumulator width trade-off -----------------------------------------------------

def aw_min_worst_case(DW, K):
    aw = 2 * DW
    while k_max_theory(DW, aw) < K:
        aw += 1
    return aw


def measure_acc_width(Ks=(16, 64, 256), aws=range(16, 25), trials=150):
    rng = np.random.default_rng(5)
    res = {}
    for K in Ks:
        W = rng.integers(-128, 128, (trials, K))
        X = rng.integers(-128, 128, (trials, K))
        exact = np.einsum('ij,ij->i', W, X)
        rates = []
        for AW in aws:
            b = bench_os(8, AW)
            wrong = 0
            for t in range(trials):
                b.step(clear=1)
                for k in range(K):
                    b.step(w_in=int(W[t, k]), x_in=int(X[t, k]), clear=0)
                wrong += b.get('acc') != int(exact[t])
            rates.append(100 * wrong / trials)
        res[K] = (list(aws), rates)
    return res


def plot_acc_width(ax, res):
    for (K, (aws, rates)), color in zip(res.items(), SEQ_K):
        ax.plot(aws, rates, 'o-', color=color, mec=SURFACE, mew=2, label=f'K = {K}')
        aw_pc = aw_min_worst_case(8, K)
        ax.axvline(aw_pc, color=color, lw=1, ls=':', alpha=0.8)
        ax.text(aw_pc - 0.1, 92, f'worst case\nK={K}', fontsize=7, color=TEXT_2, va='top', ha='right')
    ax.set_ylim(-3, 100)
    ax.set_xlabel('accumulator width AW (bits)')
    ax.set_ylabel('wrong results (%)')
    ax.set_title('3. Accumulator width vs errors (int8, uniform data)', loc='left')
    ax.legend(loc='center right')


# --- 4. Simulation speed ----------------------------------------------------------------

def measure_speed(cycles=20000):
    res = []
    for name, mk, v in [('OS', bench_os, dict(w_in=3, x_in=-2, clear=0)),
                       ('WS', bench_ws, dict(x_in=3, psum_in=7, load=0))]:
        b = mk()
        t0 = time.time()
        for _ in range(cycles):
            b.step(**v)
        res.append((name, cycles / (time.time() - t0)))
    return res


def plot_speed(ax, res):
    names = [n for n, _ in res]
    vals = [v / 1000 for _, v in res]
    bars = ax.barh(names, vals, color=[C_OS, C_WS], height=0.5, edgecolor=SURFACE, linewidth=2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() * 0.98, bar.get_y() + bar.get_height() / 2, f'{v:.0f} k',
                ha='right', va='center', fontsize=9, color='white', fontweight='bold')
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.set_xlabel('thousands of simulated cycles per second (1 PE)')
    ax.set_title('4. py4hw simulation speed', loc='left')
    ax.grid(axis='y', visible=False)


# --- Main --------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='saves the figure (png) instead of showing it')
    args = ap.parse_args()

    print('measuring the latencies...')
    lat = measure_latencies()
    print('measuring the overflow...')
    ovf = measure_overflow()
    print('measuring the accumulator width trade-off...')
    widths = measure_acc_width()
    print('measuring the simulation speed...')
    speed = measure_speed()

    print('\nSummary')
    print('  latencies      : ' + ', '.join(f'{v} {n} = {l}' for v, n, _, l in lat))
    aws, th, me = ovf[8]
    print(f'  max K int8/32  : theory {th[-1]}, measured {me[-1]}')
    for K, (a, t) in widths.items():
        ok = next(x for x, y in zip(a, t) if all(z == 0 for z in t[a.index(x):]))
        print(f'  K = {K:<4}       : 0 % error from AW = {ok} bits (worst case: {aw_min_worst_case(8, K)})')
    print('  speed          : ' + ', '.join(f'{n} {v:,.0f} cycles/s' for n, v in speed))

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    plot_latencies(axs[0, 0], lat)
    plot_overflow(axs[0, 1], ovf)
    plot_acc_width(axs[1, 0], widths)
    plot_speed(axs[1, 1], speed)
    fig.suptitle('PE benchmark — output-stationary vs weight-stationary', fontsize=13,
                 fontweight='bold', color=TEXT)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f'\nfigure saved: {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
