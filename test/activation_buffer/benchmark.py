"""
Benchmark of the activation_buffer: the activation memory in the complete operation of the NPU.

All the measurements come from the complete datapath with its memories, driven as the host will drive it
(loading W, X and b word by word, GEMM, reading Y back) — see run_layer in test.py.

  1. Port occupation during the computation: read of the IN bank (X) and write of the OUT bank (Y)
  2. Where the time goes: loading by the host, computation, reading Y back, as a function of the GEMM size
     (in v1 there is neither DMA nor double buffering: the host transfers one word per cycle and the core waits)
  3. Chaining the layers of a network: copy of Y from the OUT bank to the IN bank, compared with the computation
  4. Total memory accesses by type, 32×32×32 GEMM

Run: python test/activation_buffer/benchmark.py [--save file.png]
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


def count_accesses(log):
    compute = [e for e in log if e['phase'] == 'compute']
    return dict(
        cycles_compute=len(compute),
        rd_x=sum(e['x_rd'] is not None for e in compute),
        wr_y=sum(e['y_wr'] is not None for e in compute),
        rd_w=sum(e['w_rd'] is not None for e in compute),
        host_wr=sum((e['wb_w'] is not None) + (e['ab_w'] is not None) for e in log),
        host_rd=sum(e['ab_r'] is not None for e in log))


# --- 1. Port occupation ------------------------------------------------------------------------------------------

def measure_occupation(Ds=(4, 8, 16, 24, 32)):
    res = {}
    for v in ('OS', 'WS'):
        occ_in, occ_out = [], []
        for D in Ds:
            n = count_accesses(layer(v, D, D, D)[1])
            occ_in.append(100 * n['rd_x'] / n['cycles_compute'])
            occ_out.append(100 * n['wr_y'] / n['cycles_compute'])
        res[v] = (occ_in, occ_out)
    return list(Ds), res


def plot_occupation(ax, data):
    Ds, res = data
    for v, color in (('OS', C_OS), ('WS', C_WS)):
        occ_in, occ_out = res[v]
        ax.plot(Ds, occ_in, 'o-', color=color, mec=SURFACE, mew=2, label=f'{v}: read of the IN bank (X)')
        ax.plot(Ds, occ_out, 'o--', color=color, mec=SURFACE, mew=2, label=f'{v}: write of the OUT bank (Y)')
    ax.set_ylim(0, 100)
    ax.set_xticks(Ds)
    ax.set_xlabel('GEMM size D (D×D×D)')
    ax.set_ylabel('compute cycles with an access (%)')
    ax.set_title('1. Port occupation during the computation', loc='left')
    ax.legend(loc='center right', handlelength=3)


# --- 2. Where the time goes -----------------------------------------------------------------------------------------

PHASES = (('loading', 'loading by the host (W, X, b)', '#9085e9'),
          ('compute', 'computation', '#3a2d9a'),
          ('readback', 'reading Y back by the host', '#e87ba4'))


def measure_time(Ds=(4, 8, 16, 32, 64)):
    res = {}
    for v in ('OS', 'WS'):
        res[v] = [layer(v, D, D, D)[0] for D in Ds]
    return list(Ds), res


def plot_time(ax, data):
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
    ax.set_ylabel('share of the total time (%)')
    ax.set_title('2. Where the time goes (above: total cycles)', loc='left')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncols=3)
    ax.grid(axis='x', visible=False)


# --- 3. Chaining the layers ----------------------------------------------------------------------------------

def measure_chaining(dims=(64, 48, 32, 16), batch=16):
    """MLP 64 → 48 → 32 → 16 on a batch of 16: computation of each layer and OUT → IN copy between two layers."""
    rng = np.random.default_rng(161)
    res = {}
    for v in ('OS', 'WS'):
        b = T.datapath(v)
        X = rng.integers(-128, 128, (dims[0], batch))
        layers = []
        for li, (k, m) in enumerate(zip(dims[:-1], dims[1:])):
            W = rng.integers(-128, 128, (m, k))
            Y, st = T.run_layer(b, v, W, X, [0] * m, 1, 12, 1, load_x=(li == 0))
            copy = T.copy_out_to_in(b, layout.y_addresses(m, batch, v)) if li < len(dims) - 2 else 0
            layers.append((f'{k}→{m}', st['compute'], copy))
            X = Y
        res[v] = layers
    return res


def plot_chaining(ax, res):
    names = [c[0] for c in res['OS']]
    x = np.arange(len(names))
    for off, v, color in ((-0.27, 'OS', C_OS), (0, 'WS', C_WS)):
        ax.bar(x + off, [c[1] for c in res[v]], width=0.26, color=color, edgecolor=SURFACE, linewidth=1.5,
               label=f'computation {v}')
    copies = [c[2] for c in res['OS']]
    ax.bar(x + 0.27, copies, width=0.26, color='#e87ba4', edgecolor=SURFACE, linewidth=1.5,
           label='OUT → IN copy by the host')
    for xi, cp in zip(x, copies):
        if cp:
            ax.text(xi + 0.27, cp, str(cp), ha='center', va='bottom', fontsize=8, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels([f'layer {n}' for n in names])
    ax.set_ylabel('cycles')
    ax.set_title('3. 64→48→32→16 MLP (batch of 16): computation versus copy between layers', loc='left')
    ax.legend(loc='upper right')
    ax.grid(axis='x', visible=False)


# --- 4. Total accesses ------------------------------------------------------------------------------------------

TYPES = (('rd_w', 'reads of W\n(core)'), ('rd_x', 'reads of X\n(core)'), ('wr_y', 'writes of Y\n(core)'),
         ('host_wr', 'host\nwrites'), ('host_rd', 'host\nreads'))


def measure_accesses(D=32):
    return D, {v: count_accesses(layer(v, D, D, D)[1]) for v in ('OS', 'WS')}


def plot_accesses(ax, data):
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
    ax.set_ylabel('accesses (32-bit words)')
    ax.set_title(f'4. Memory accesses by type, {D}×{D}×{D} GEMM', loc='left')
    ax.legend(loc='upper right')
    ax.grid(axis='x', visible=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='saves the figure (png) instead of showing it')
    args = ap.parse_args()

    print('port occupation...')
    occ = measure_occupation()
    print('breakdown of the time (up to D = 64)...')
    times = measure_time()
    print('3-layer MLP...')
    ch = measure_chaining()
    print('total accesses...')
    acc = measure_accesses()

    print('\nSummary')
    Ds, r = times
    for v in ('OS', 'WS'):
        print(f'  {v}: ' + ', '.join(f'D={d} compute {100 * st["compute"] / sum(st.values()):.0f} % of {sum(st.values())} cycles'
                                     for d, st in zip(Ds, r[v])))
    for v in ('OS', 'WS'):
        print(f'  MLP {v}: ' + ', '.join(f'{n} compute {c} + copy {cp}' for n, c, cp in ch[v]))
    D, a = acc
    print(f'  accesses GEMM {D}³: OS {a["OS"]}, WS {a["WS"]}')

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    plot_occupation(axs[0, 0], occ)
    plot_time(axs[0, 1], times)
    plot_chaining(axs[1, 0], ch)
    plot_accesses(axs[1, 1], acc)
    fig.suptitle('activation_buffer benchmark — the memory in the complete operation of the NPU',
                 fontsize=13, fontweight='bold', color=TEXT)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f'\nfigure saved: {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
