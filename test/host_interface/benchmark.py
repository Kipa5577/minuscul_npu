"""
Benchmark of the host_interface: what going through the 32-bit bus costs.

All the measurements are made on the complete chip (npu_top), driven only through its bus by the host
software of test/npu_top/test.py: one access per cycle, one 32-bit word per access.

  1. Where the bus cycles go for a D×D×D layer: loading W, X, the biases, configuration,
     waiting for the computation (polling STATUS), reading Y back
  2. Effective throughput: MACs per cycle, host included, compared with the computation alone and the array peak (16)
  3. Data exchanged on the bus per computed MAC
  4. A 64 → 48 → 32 → 16 MLP (batch of 16) on the chip: loading, copy of Y to ACT_IN between layers,
     computation and final read-back

Run: python test/host_interface/benchmark.py [--save file.png]
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

PHASES = (('W', 'loading W', '#6250d6'), ('X', 'loading X', '#1baf7a'), ('b', 'biases', '#9085e9'),
          ('cfg', 'configuration + GO', '#eda100'), ('compute', 'computation (polling STATUS)', '#3a2d9a'),
          ('Y', 'reading Y back', '#e87ba4'))


def detailed_layer(npu, W, X, bias, mult=1, shift=10, relu=True, load_x=True):
    """One layer, phase by phase; returns (Y, cycles per phase, compute cycles read from CYCLES)."""
    M, K = W.shape
    Nd = X.shape[1]
    ph = {}

    def phase(name, fn):
        c = npu.cycle
        r = fn()
        ph[name] = npu.cycle - c
        return r

    phase('W', lambda: npu.load_w(W))
    phase('X', lambda: npu.load_x(X) if load_x else None)
    phase('b', lambda: npu.load_bias(bias[:M]))
    phase('cfg', lambda: (npu.configure(M, K, Nd, mult, shift), npu.start(relu)))
    phase('compute', npu.wait_done)
    cycles = npu.read(T.CYCLES)
    Y = phase('Y', lambda: npu.read_y(M, Nd))
    return Y, ph, cycles


def measure_layers(Ds=(4, 8, 16, 32, 64)):
    rng = np.random.default_rng(200)
    res = {}
    for v in ('OS', 'WS'):
        res[v] = []
        for D in Ds:
            npu = T.Npu(v)
            W, X = rng.integers(-128, 128, (D, D)), rng.integers(-128, 128, (D, D))
            _, ph, cyc = detailed_layer(npu, W, X, [0] * D)
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
    ax.set_ylabel('share of the bus cycles (%)')
    ax.set_title('1. Where the cycles of a D×D×D layer go (above: total)', loc='left')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncols=3)
    ax.grid(axis='x', visible=False)


def plot_throughput(ax, data):
    Ds, res = data
    ax.axhline(16, color=REF, lw=1, ls='--')
    ax.text(Ds[0], 16.3, 'peak of the 4×4 array: 16 MACs/cycle', fontsize=8, color=TEXT_2)
    for v, color in (('OS', C_OS), ('WS', C_WS)):
        total = [D ** 3 / sum(ph.values()) for D, (ph, _) in zip(Ds, res[v])]
        compute = [D ** 3 / cyc for D, (_, cyc) in zip(Ds, res[v])]
        ax.plot(Ds, compute, 'o--', color=color, mec=SURFACE, mew=2, label=f'{v}: computation alone (CYCLES)')
        ax.plot(Ds, total, 'o-', color=color, mec=SURFACE, mew=2, label=f'{v}: host included')
    ax.set_xscale('log', base=2)
    ax.set_xticks(Ds)
    ax.set_xticklabels(Ds)
    ax.set_ylim(0, 17.5)
    ax.set_xlabel('layer size D (D×D×D)')
    ax.set_ylabel('MACs per cycle')
    ax.set_title('2. Effective throughput of the chip', loc='left')
    ax.legend(loc='center left', handlelength=3)


def plot_bytes(ax, data):
    Ds, res = data
    ph0 = res['OS']
    nbytes = [4 * (ph['W'] + ph['X'] + ph['b'] + ph['Y']) / D ** 3 for D, (ph, _) in zip(Ds, ph0)]
    ax.plot(Ds, nbytes, 'o-', color='#3a2d9a', mec=SURFACE, mew=2)
    for D, o in zip(Ds, nbytes):
        ax.annotate(f'{o:.2f}', (D, o), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8, color=TEXT)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(Ds)
    ax.set_xticklabels(Ds)
    ax.set_xlabel('layer size D (D×D×D)')
    ax.set_ylabel('bytes exchanged on the bus per MAC')
    ax.set_title('3. Data transferred by the host, per computed MAC (identical OS / WS)', loc='left')


def measure_mlp(dims=(64, 48, 32, 16), batch=16):
    rng = np.random.default_rng(201)
    res = {}
    for v in ('OS', 'WS'):
        npu = T.Npu(v)
        X = rng.integers(-128, 128, (dims[0], batch))
        layers = []
        for li, (k, m) in enumerate(zip(dims[:-1], dims[1:])):
            W = rng.integers(-128, 128, (m, k))
            c0 = npu.cycle
            Y, ph, _ = detailed_layer(npu, W, X, [0] * m, 1, 12, True, load_x=(li == 0))
            last = li == len(dims) - 2
            c1 = npu.cycle
            if not last:
                npu.copy_out_to_in(m, batch)
            layers.append(dict(loading=ph['W'] + ph['X'] + ph['b'] + ph['cfg'], compute=ph['compute'],
                                copy=npu.cycle - c1, readback=ph['Y'] if last else 0, name=f'{k}→{m}'))
            X = Y
        res[v] = layers
    return res


def plot_mlp(ax, res):
    parts = (('loading', 'loading (W, X, b, config)', '#9085e9'), ('compute', 'computation', '#3a2d9a'),
             ('copy', 'copy Y → ACT_IN (read-back included)', '#e87ba4'), ('readback', 'final read-back', '#eda100'))
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
            labels.append(f'{v}  {c["name"]}')
            y += 1
        y += 0.6
    ax.set_yticks([i if i < 3 else i + 0.6 for i in range(6)])
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('cycles')
    tot = {v: sum(sum(c[k] for k, _, _ in parts) for c in res[v]) for v in ('OS', 'WS')}
    ax.set_title(f'4. 64→48→32→16 MLP, batch of 16 — total OS {tot["OS"]} / WS {tot["WS"]} cycles', loc='left')
    ax.legend(loc='lower right', fontsize=7.5)
    ax.grid(axis='y', visible=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='saves the figure (png) instead of showing it')
    args = ap.parse_args()

    print('D×D×D layers on the complete chip (up to D = 64)...')
    co = measure_layers()
    print('3-layer MLP...')
    ml = measure_mlp()

    Ds, res = co
    print('\nSummary')
    for v in ('OS', 'WS'):
        print(f'  {v}: ' + ', '.join(f'D={d} {D3 / sum(ph.values()):.1f} MACs/cycle (computation alone {D3 / cyc:.1f})'
                                     for d, (ph, cyc), D3 in zip(Ds, res[v], [d ** 3 for d in Ds])))
    for v in ('OS', 'WS'):
        print(f'  MLP {v}: ' + ', '.join(f'{c["name"]} loading {c["loading"]} compute {c["compute"]} '
                                         f'copy {c["copy"]}' for c in ml[v]))

    fig, axs = plt.subplots(2, 2, figsize=(14, 10.5))
    plot_phases(axs[0, 0], co)
    plot_throughput(axs[0, 1], co)
    plot_bytes(axs[1, 0], co)
    plot_mlp(axs[1, 1], ml)
    fig.suptitle('host_interface benchmark — the cost of the 32-bit bus, measured on the complete chip',
                 fontsize=13, fontweight='bold', color=TEXT)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f'\nfigure saved: {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
