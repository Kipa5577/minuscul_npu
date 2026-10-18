"""
Benchmark of npu_top: the complete chip on real workloads, driven only through its bus.

  1. A floating-point neural network (MLP 64 → 32 → 16, batch of 8), quantized to int8 and run on the
     chip layer by layer (Y copied to ACT_IN between two layers), compared with the floating-point computation
  2. Batch size: total cycles (host included) of a 64 → 64 layer according to the number of samples, OS versus WS
  3. Activity of each block during a 32×32×32 layer, loading and reading back included
  4. Timing diagram of this activity for a 16×16×16 layer (OS)

Run: python test/npu_top/benchmark.py [--save file.png]
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


def quantize(a):
    s = np.abs(a).max() / 127
    return np.clip(np.round(a / s), -127, 127).astype(int), s


# --- 1. Floating-point network on the chip ------------------------------------------------------------------------

def measure_network(seed=210):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (64, 8))
    W1, b1 = rng.normal(0, 1 / 8, (32, 64)), rng.normal(0, 0.1, 32)
    W2, b2 = rng.normal(0, 1 / 6, (16, 32)), rng.normal(0, 0.1, 16)
    H = np.maximum(0, W1 @ X + b1[:, None])
    Yf = W2 @ H + b2[:, None]
    Xq, sx = quantize(X)
    W1q, sw1 = quantize(W1)
    W2q, sw2 = quantize(W2)
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


def plot_network(ax, data):
    Yf, res = data
    lim = np.abs(Yf).max() * 1.1
    ax.plot([-lim, lim], [-lim, lim], color=REF, lw=1, ls='--')
    ax.scatter(Yf.flatten(), res['OS'][0].flatten(), s=24, color=C_OS, edgecolor=SURFACE, lw=0.6, zorder=3,
               label=f'OS ({res["OS"][1]} bus cycles)')
    ax.scatter(Yf.flatten(), res['WS'][0].flatten(), s=8, color=C_WS, zorder=4,
               label=f'WS, identical ({res["WS"][1]} bus cycles)')
    err = res['OS'][0] - Yf
    corr = np.corrcoef(Yf.flatten(), res['OS'][0].flatten())[0, 1]
    ax.text(0.03, 0.97, f'correlation {corr:.4f}\nRMS error {np.sqrt(np.mean(err ** 2)):.3f} '
            f'(outputs ∈ ±{np.abs(Yf).max():.1f})', transform=ax.transAxes, va='top', fontsize=8, color=TEXT)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('floating-point network output')
    ax.set_ylabel('chip output (dequantized int8)')
    ax.set_title('1. 64→32→16 MLP run on the chip, through the bus', loc='left')
    ax.legend(loc='lower right')


# --- 2. Batch size -------------------------------------------------------------------------------------------------

def measure_batch(batches=(1, 2, 4, 8, 16, 32, 64)):
    rng = np.random.default_rng(211)
    W = rng.integers(-128, 128, (64, 64))
    res = {}
    for v in ('OS', 'WS'):
        res[v] = []
        for n in batches:
            npu = T.Npu(v)
            c0 = npu.cycle
            npu.gemm(W, rng.integers(-128, 128, (64, n)), [0] * 64, 1, 12, True)
            res[v].append(npu.cycle - c0)
    return list(batches), res


def plot_batch(ax, data):
    batches, res = data
    for v, color in (('OS', C_OS), ('WS', C_WS)):
        per_sample = [c / n for c, n in zip(res[v], batches)]
        ax.plot(batches, per_sample, 'o-', color=color, mec=SURFACE, mew=2, label=v)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(batches)
    ax.set_xticklabels(batches)
    ax.set_xlabel('batch size (columns of X)')
    ax.set_ylabel('cycles per sample (host included)')
    ax.set_title('2. 64→64 layer: cost per sample according to the batch size', loc='left')
    ax.legend(loc='upper right')


# --- 3 and 4. Activity of each block -------------------------------------------------------------------------------

BLOCKS = ['host bus', 'controller', 'weight_buffer', 'activation_buffer', 'data_feeder', 'pe_array',
         'output_drain', 'post_processing']


def probe(npu):
    """Replaces the step of the bench to record, at each cycle, which blocks of the chip are working."""
    b, d = npu.b, npu.b.dut
    trace = []

    def pe_active(pe):
        if d.dataflow == 'OS':
            return pe.w_in.get() and pe.x_in.get()
        return pe.x_in.get() and pe.w_r          # a MAC can take place during the first reload cycle

    def step(**v):
        b.set(**v)
        b.sim.propagateAll()
        wb, ab = d.weight_buffer, d.activation_buffer
        trace.append({
            'host bus': d.host.bus_we.get() or d.host.bus_re.get(),
            'controller': d.controller.busy.get(),
            'weight_buffer': (wb.busy.get() and wb.core_re.get()) or (not wb.busy.get() and (wb.host_we.get() or wb.host_re.get())),
            'activation_buffer': (ab.busy.get() and (ab.core_rd_re.get() or ab.core_wr_we.get()))
                                 or (not ab.busy.get() and (ab.host_we.get() or ab.host_re.get())),
            'data_feeder': d.data_feeder.w_arr.get() or d.data_feeder.x_arr.get(),
            'pe_array': any(pe_active(pe) for row in d.pe_array.pes for pe in row),
            'output_drain': d.output_drain.res_valid.get(),
            'post_processing': d.post_processing.y_we.get()})
        b.sim.clk(1)
        b.cycle += 1

    b.step = step
    return trace


def measure_activity(D=32):
    rng = np.random.default_rng(212)
    W, X = rng.integers(1, 128, (D, D)), rng.integers(1, 128, (D, D))
    res = {}
    for v in ('OS', 'WS'):
        npu = T.Npu(v)
        trace = probe(npu)
        npu.gemm(W, X, [0] * D, 1, 12, True)
        res[v] = {k: 100 * np.mean([bool(t[k]) for t in trace]) for k in BLOCKS}
    return D, res


def plot_activity(ax, data):
    D, res = data
    y = np.arange(len(BLOCKS))
    for off, v, color in ((-0.2, 'OS', C_OS), (0.2, 'WS', C_WS)):
        vals = [res[v][k] for k in BLOCKS]
        ax.barh(y + off, vals, height=0.38, color=color, edgecolor=SURFACE, linewidth=1, label=v)
        for yi, val in zip(y, vals):
            ax.text(val + 1, yi + off, f'{val:.0f} %', va='center', fontsize=7, color=TEXT)
    ax.set_yticks(y)
    ax.set_yticklabels(['host bus\n(polling included)'] + BLOCKS[1:])
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel('cycles in which the block is working (%)')
    ax.set_title(f'3. Activity of each block, {D}×{D}×{D} layer (host included)', loc='left')
    ax.legend(loc='lower right')
    ax.grid(axis='y', visible=False)


def measure_timing(D=16):
    rng = np.random.default_rng(213)
    npu = T.Npu('OS')
    trace = probe(npu)
    npu.gemm(rng.integers(1, 128, (D, D)), rng.integers(1, 128, (D, D)), [0] * D, 1, 12, True)
    return D, trace


def plot_timing(ax, data):
    D, trace = data
    colors = ['#52514e', '#9085e9', '#6250d6', '#1baf7a', '#eda100', '#3a2d9a', '#e87ba4', '#08402c']
    for i, (k, col) in enumerate(zip(BLOCKS, colors)):
        on = np.array([bool(t[k]) for t in trace])
        starts = np.flatnonzero(np.diff(np.concatenate(([0], on.astype(int)))) == 1)
        ends = np.flatnonzero(np.diff(np.concatenate((on.astype(int), [0]))) == -1) + 1
        ax.broken_barh([(a, b - a) for a, b in zip(starts, ends)], (i - 0.35, 0.7), color=col)
    ax.set_yticks(range(len(BLOCKS)))
    ax.set_yticklabels(BLOCKS)
    ax.invert_yaxis()
    ax.set_xlim(0, len(trace))
    ax.set_xlabel('cycle')
    ax.set_title(f'4. Timing diagram of the chip, {D}×{D}×{D} layer in OS', loc='left')
    ax.grid(axis='y', visible=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='saves the figure (png) instead of showing it')
    args = ap.parse_args()

    print('quantized floating-point MLP, on the chip...')
    rs = measure_network()
    print('batch size...')
    batch = measure_batch()
    print('activity of the blocks...')
    act = measure_activity()
    timing = measure_timing()

    Yf, r = rs
    print('\nSummary')
    print(f'  MLP: OS and WS identical = {bool((r["OS"][0] == r["WS"][0]).all())}, correlation '
          f'{np.corrcoef(Yf.flatten(), r["OS"][0].flatten())[0, 1]:.4f}, cycles OS {r["OS"][1]} / WS {r["WS"][1]}')
    batches, l = batch
    print('  cycles per sample (64→64 layer): ' + ', '.join(
        f'batch {n} OS {a / n:.0f} / WS {b / n:.0f}' for n, a, b in zip(batches, l['OS'], l['WS'])))
    D, a = act
    for v in ('OS', 'WS'):
        print(f'  activity {v} ({D}³): ' + ', '.join(f'{k} {a[v][k]:.0f} %' for k in BLOCKS))

    fig, axs = plt.subplots(2, 2, figsize=(15, 10.5))
    plot_network(axs[0, 0], rs)
    plot_batch(axs[0, 1], batch)
    plot_activity(axs[1, 0], act)
    plot_timing(axs[1, 1], timing)
    fig.suptitle('npu_top benchmark — the complete chip on real workloads', fontsize=13, fontweight='bold',
                 color=TEXT)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f'\nfigure saved: {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
