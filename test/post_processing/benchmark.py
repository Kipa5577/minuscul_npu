"""
Benchmark of the post_processing: limits and trade-offs, as matplotlib charts.

  1. Precision of the requantization according to the width of the multiplier M
     (error with respect to the ideal round(x · scale), in int8 quantization steps)
  2. Saturation / resolution trade-off according to the chosen output scale, on real GEMM outputs
  3. 2-layer neural network run end to end on the simulated hardware
     (data_feeder → pe_array → output_drain → post_processing), compared with the same network in floating point
  4. Rhythm of the writes of Y into activation_buffer, 16×16×16 GEMM

Run: python test/post_processing/benchmark.py [--save file.png]
"""

import sys
import math
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import Bench, pack, unpack, load_test_module
from viz import BENCH_RC, SURFACE, TEXT, TEXT_2, REF, C_OS, C_WS, ERR

import numpy as np
import matplotlib.pyplot as plt
from src.post_processing.post_processing import PostProcessing_OS

T = load_test_module('post_processing')
OD = T.OD
N, AW, DW, IW, TAG_W, BAW = T.N, T.AW, T.DW, T.IW, T.TAG_W, T.BAW
plt.rcParams.update(BENCH_RC)


# --- Requantization on the hardware, with a parameterisable width of M ------------------------------------

def bench_mw(mw):
    b = Bench(
        inputs={'rst_n': 1, 'busy': 1, 'res': N * AW, 'res_valid': 1, 'tag_in': TAG_W, 'relu_en': 1,
                'req_mult': mw, 'req_shift': 5, 'bias_host_addr': IW, 'bias_host_wdata': AW,
                'bias_host_we': 1, 'bias_host_re': 1},
        outputs={'bias_host_rdata': AW, 'y_addr': BAW, 'y_wdata': N * DW, 'y_we': 1, 'y_wmask': N},
        build=lambda s, w: PostProcessing_OS(s, 'post', w['rst_n'], w['busy'], w['res'], w['res_valid'],
                                             w['tag_in'], w['relu_en'], w['req_mult'], w['req_shift'],
                                             w['bias_host_addr'], w['bias_host_wdata'], w['bias_host_we'],
                                             w['bias_host_re'], w['bias_host_rdata'], w['y_addr'], w['y_wdata'],
                                             w['y_we'], w['y_wmask']))
    b.step(rst_n=0)
    b.set(rst_n=1, relu_en=0)
    return b


def mult_shift(scale, mw):
    """Best approximation scale ≈ M / 2^s with M on mw bits and s ≤ 31."""
    s = min(31, max(0, math.floor(math.log2(((1 << mw) - 1) / scale))))
    return min((1 << mw) - 1, max(1, round(scale * 2 ** s))), s


def requant_hw(b, xs, M, s):
    """Passes the values xs (in packets of N) through the pipeline; returns the int8 values produced."""
    b.set(req_mult=M, req_shift=s)
    out = []
    xs = list(xs) + [0] * (-len(xs) % N)
    packets = [xs[i:i + N] for i in range(0, len(xs), N)]
    for c in range(len(packets) + 3):
        p = packets[c] if c < len(packets) else [0] * N
        b.step(res=pack(p, AW), res_valid=int(c < len(packets)), tag_in=0)
        if b.get('y_we'):
            out += unpack(b.w['y_wdata'].get(), N, DW)
    return np.array(out[:len(xs)])


def measure_precision(bits=(4, 6, 8, 10, 12, 14, 16), n_scales=24, n_x=64):
    rng = np.random.default_rng(120)
    scales = 10 ** rng.uniform(-4.5, -2, n_scales)
    res = {'rms': [], 'max': []}
    for mw in bits:
        b = bench_mw(mw)
        errs = []
        for sc in scales:
            M, s = mult_shift(sc, mw)
            xs = rng.integers(-int(127 / sc), int(127 / sc), n_x)          # outputs within the int8 range
            ideal = np.clip(np.round(xs * sc), -128, 127)
            errs += list(requant_hw(b, xs, M, s)[:n_x] - ideal)
        errs = np.array(errs, float)
        res['rms'].append(np.sqrt(np.mean(errs ** 2)))
        res['max'].append(np.max(np.abs(errs)))
    return list(bits), res


def plot_precision(ax, data):
    bits, res = data
    ax.plot(bits, res['max'], 'o-', color='#3a2d9a', mec=SURFACE, mew=2, label='maximum error')
    ax.plot(bits, res['rms'], 'o-', color='#9085e9', mec=SURFACE, mew=2, label='root mean square error')
    ax.axvline(16, color=REF, lw=1, ls=':')
    ax.text(15.85, 0.15, 'v1: 16 bits', ha='right', fontsize=8, color=TEXT_2)
    zero = [b for b, m in zip(bits, res['max']) if m == 0]
    if zero:
        ax.annotate('identical to the ideal', (zero[0], 0), xytext=(0, 18), textcoords='offset points',
                    ha='center', fontsize=8, color=TEXT, arrowprops=dict(arrowstyle='->', color=TEXT_2, lw=0.8))
    ax.set_ylim(-0.2, max(res['max']) * 1.15)
    ax.set_xticks(bits)
    ax.set_xlabel('width of the multiplier M (bits)')
    ax.set_ylabel('error (int8 quantization steps)')
    ax.set_title('1. Requantization precision according to the width of M', loc='left')
    ax.legend(loc='upper right')


# --- 2. Saturation versus resolution ------------------------------------------------------------------------

def measure_scale(n_scales=25):
    rng = np.random.default_rng(121)
    acc = (rng.integers(-128, 128, (32, 64)) @ rng.integers(-128, 128, (64, 32))).flatten()   # real GEMM outputs
    sigma = acc.std()
    b = bench_mw(16)
    # scale expressed as "how many standard deviations fit in the int8 range"
    coverage = np.geomspace(0.25, 40, n_scales)
    sat, err = [], []
    for k in coverage:
        sc = 127 / (k * sigma)
        M, s = mult_shift(sc, 16)
        y = requant_hw(b, acc, M, s)[:len(acc)]
        sat.append(100 * np.mean(np.abs(y) >= 127.5 - 1))
        err.append(100 * np.sqrt(np.mean((y / sc - acc) ** 2)) / sigma)
    return coverage, sat, err


def plot_scale(ax, data):
    k, sat, err = data
    ax.plot(k, err, color='#3a2d9a', label='reconstruction error (% of the standard deviation)')
    ax.plot(k, sat, color=ERR, label='saturated outputs (%)')
    best = int(np.argmin(err))
    ax.plot([k[best]], [err[best]], 'o', color='#3a2d9a', mec=SURFACE, mew=2)
    ax.annotate(f'optimum: ±{k[best]:.1f} σ → {err[best]:.1f} %', (k[best], err[best]), xytext=(8, 14),
                textcoords='offset points', fontsize=8, color=TEXT)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('int8 range covered, in standard deviations of the accumulators (output scale)')
    ax.set_ylabel('%')
    ax.set_title('2. Saturation versus resolution according to the output scale', loc='left')
    ax.legend(loc='upper center')


# --- 3. 2-layer network end to end ---------------------------------------------------------------------------

def quantize(a):
    s = np.abs(a).max() / 127
    return np.clip(np.round(a / s), -127, 127).astype(int), s


def layer_hw(variant, Wq, Xq, bq, M, s, relu, seed):
    Y, count, _ = T.gemm_full(variant, Wq, Xq, list(bq) + [0] * (64 - len(bq)), M, s, relu, seed)
    assert (count == 1).all()
    return Y


def measure_network(seed=122):
    """MLP 64 → 32 (ReLU) → 16 on a batch of 8 vectors, floating-point weights quantized to int8 per tensor."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (64, 8))
    W1, b1 = rng.normal(0, 1 / 8, (32, 64)), rng.normal(0, 0.1, 32)
    W2, b2 = rng.normal(0, 1 / 6, (16, 32)), rng.normal(0, 0.1, 16)
    H = np.maximum(0, W1 @ X + b1[:, None])
    Yf = W2 @ H + b2[:, None]                                       # floating-point reference

    Xq, sx = quantize(X)
    W1q, sw1 = quantize(W1)
    W2q, sw2 = quantize(W2)
    sh = H.max() / 127                                              # output scales (calibrated on the floating-point run)
    sy = np.abs(Yf).max() / 127
    res = {}
    for v in ('OS', 'WS'):
        M1, s1 = mult_shift(sw1 * sx / sh, 16)
        Hq = layer_hw(v, W1q, Xq, np.round(b1 / (sw1 * sx)).astype(int), M1, s1, 1, seed)
        M2, s2 = mult_shift(sw2 * sh / sy, 16)
        Yq = layer_hw(v, W2q, Hq, np.round(b2 / (sw2 * sh)).astype(int), M2, s2, 0, seed + 1)
        res[v] = Yq * sy
    return Yf, res


def plot_network(ax, data):
    Yf, res = data
    lim = np.abs(Yf).max() * 1.1
    ax.plot([-lim, lim], [-lim, lim], color=REF, lw=1, ls='--')
    ax.scatter(Yf.flatten(), res['OS'].flatten(), s=22, color=C_OS, edgecolor=SURFACE, lw=0.6, label='OS', zorder=3)
    ax.scatter(Yf.flatten(), res['WS'].flatten(), s=8, color=C_WS, label='WS (identical to OS)', zorder=4)
    err = res['OS'] - Yf
    corr = np.corrcoef(Yf.flatten(), res['OS'].flatten())[0, 1]
    ax.text(0.03, 0.97, f'correlation {corr:.4f}\nRMS error {np.sqrt(np.mean(err ** 2)):.3f} '
            f'(outputs ∈ ±{np.abs(Yf).max():.1f})', transform=ax.transAxes, va='top', fontsize=8, color=TEXT)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('floating-point network output')
    ax.set_ylabel('output of the simulated NPU (dequantized int8)')
    ax.set_title('3. 64→32→16 MLP end to end on the simulated NPU', loc='left')
    ax.legend(loc='lower right')


# --- 4. Rhythm of the writes ---------------------------------------------------------------------------------

def measure_writes(D=16):
    rng = np.random.default_rng(123)
    W, X = rng.integers(-128, 128, (D, D)), rng.integers(-128, 128, (D, D))
    res = {}
    for v in ('OS', 'WS'):
        _, _, writes = T.gemm_full(v, W, X, [0] * 64, 1, 8, 0)
        res[v] = [c for c, *_ in writes]
    return D, res


def plot_writes(ax, data):
    D, res = data
    for v, color in (('OS', C_OS), ('WS', C_WS)):
        c = res[v]
        ax.step([0] + c, range(len(c) + 1), where='post', color=color,
                label=f'{v}: {len(c)} words, last one at cycle {c[-1]}')
    ax.axhline(D * D / N, color=REF, lw=1, ls='--')
    ax.text(0, D * D / N + 1.5, f'{D * D // N} 4-byte words', fontsize=8, color=TEXT_2)
    ax.set_xlabel('cycle')
    ax.set_ylabel('words of Y written (cumulative)')
    ax.set_title(f'4. Writes of Y into activation_buffer, {D}×{D}×{D} GEMM', loc='left')
    ax.legend(loc='center right')


# --- Main --------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='saves the figure (png) instead of showing it')
    args = ap.parse_args()

    print('precision according to the width of M...')
    pr = measure_precision()
    print('saturation versus resolution...')
    ec = measure_scale()
    print('2-layer network end to end...')
    rs = measure_network()
    print('writes of Y...')
    wr = measure_writes()

    bits, r = pr
    print('\nSummary')
    print('  max error (int8 steps) according to the bits of M: ' + ', '.join(f'{b}b {m:.2f}' for b, m in zip(bits, r['max'])))
    k, sat, err = ec
    i = int(np.argmin(err))
    print(f'  best output scale: ±{k[i]:.1f} σ, error {err[i]:.1f} %, saturation {sat[i]:.2f} %')
    Yf, hw = rs
    print(f'  MLP: OS and WS identical = {bool((hw["OS"] == hw["WS"]).all())}, '
          f'RMS error {np.sqrt(np.mean((hw["OS"] - Yf) ** 2)):.3f}, '
          f'correlation {np.corrcoef(Yf.flatten(), hw["OS"].flatten())[0, 1]:.4f}')
    D, w = wr
    print(f'  writes GEMM {D}³: OS last word at cycle {w["OS"][-1]}, WS at cycle {w["WS"][-1]}')

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    plot_precision(axs[0, 0], pr)
    plot_scale(axs[0, 1], ec)
    plot_network(axs[1, 0], rs)
    plot_writes(axs[1, 1], wr)
    fig.suptitle('post_processing benchmark — requantization and complete chain', fontsize=13,
                 fontweight='bold', color=TEXT)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f'\nfigure saved: {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
