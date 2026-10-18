"""
Animation du post_processing : le pipeline de 3 étages, cycle par cycle.

La chaîne complète data_feeder → pe_array → output_drain → post_processing calcule
Y = requant(ReLU(W·X + b)) ; seul le post_processing est dessiné. Chaque voie est une ligne :
  res (int32, depuis l'output_drain) → S1 : + biais → S2 : ReLU puis × M → S3 : arrondi, >> s, saturation
puis écriture du mot de 4 octets dans le banc OUT de l'activation_buffer (les voies masquées ne sont pas écrites).

  - OS (en haut) : les 4 voies sont une même ligne de Y → un seul biais, le même pour les 4 voies
  - WS (en bas)  : les 4 voies sont 4 lignes de Y       → un biais différent par voie

L'animation commence deux cycles avant le premier résultat de l'output_drain.
Les valeurs sont celles de la simulation py4hw. Rendu ≥ 30 images/s (voir test/viz.py).

Lancement : python test/post_processing/animation.py [--save fichier.gif] [--fps 30] [--cycle-ms 900]
"""

import sys
import time
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import from_c2, unpack, load_test_module
from viz import (SURFACE, TEXT, TEXT_2, CELL, GRID, C_OS, C_WS, C_W, C_P, ERR,
                 ease, box_path, RECT, Layer, TextLayer, TokenLayers, title_axes)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

T = load_test_module('post_processing')
OD = T.OD
N, AW, DW, IW = T.N, T.AW, T.DW, T.IW
H = 8                               # mots écrits affichés
EDGE = '#b9b8b2'
C_Y = '#1baf7a'                     # octets int8 écrits
CFG = dict(M=5, s=2, relu=1)        # y = sat(round(ReLU(acc + b) × 1,25))


# --- Trace ---------------------------------------------------------------------------------------------

def capture(variant, Mr, K, Nd, seed):
    rng = np.random.default_rng(seed)
    W = rng.integers(-7, 8, (Mr, K))
    X = rng.integers(-7, 8, (K, Nd))
    bias = list(rng.integers(-40, 41, 64))
    b = OD.core(variant, N, post=True)
    for i, v in enumerate(bias):
        b.step(busy=0, bias_host_addr=i, bias_host_wdata=int(v) & 0xFFFFFFFF, bias_host_we=1)
    b.set(busy=1, bias_host_we=0, req_mult=CFG['M'], req_shift=CFG['s'], relu_en=CFG['relu'])
    plan = (OD.plan_os if variant == 'OS' else OD.plan_ws)(W, X, N, np.random.default_rng(seed))
    recs = []

    def hook(c, bb):
        pp = bb.dut['post']
        valid, tag = bb.get('res_valid'), bb.get('tag_out')
        row = tag & ((1 << IW) - 1)
        recs.append(dict(
            inp=(valid, unpack(bb.w['res'].get(), N, AW), tag),
            bias=[pp.bias[(row + l) % 64 if pp.BIAS_PER_LANE else row] for l in range(N)],
            bias_idx=[(row + l) % 64 if pp.BIAS_PER_LANE else row for l in range(N)],
            s1=pp.s1, s2=pp.s2, s3=pp.s3))

    OD.run_core(b, plan, on_cycle=hook)
    first = next(c for c, r in enumerate(recs) if r['inp'][0])
    last = max(c for c, r in enumerate(recs) if r['s3'][0])
    window = recs[max(0, first - 2):last + 3]
    ref = T.reference(W, X, bias, CFG['M'], CFG['s'], CFG['relu'])
    return window, ref


def describe(variant, tag):
    y_addr, mask, row = OD.tag_unpack(tag, N)
    t, idx = y_addr >> IW, y_addr & ((1 << IW) - 1)
    return (f'Y[{idx}, {N * t}–{N * t + N - 1}]' if variant == 'OS' else f'Y[{N * t}–{N * t + N - 1}, {idx}]'), mask


# --- Panneau ---------------------------------------------------------------------------------------------

class Panel:
    X_IN, X_B, X_S1, X_S2, X_S3 = 0.0, 1.35, 2.75, 4.45, 6.15
    X_MEM = 7.6

    def __init__(self, ax, variant, color):
        self.ax, self.variant, self.color = ax, variant, color
        ax.set_xlim(-1.1, self.X_MEM + H * 0.95 + 0.2)
        ax.set_ylim(N + 0.35, -1.85)
        ax.set_aspect('equal')
        ax.axis('off')
        titre = ('output-stationary — 4 voies = une ligne de Y, un seul biais' if variant == 'OS'
                 else 'weight-stationary — 4 voies = 4 lignes de Y, un biais par voie')
        ax.text(-1.05, -1.6, titre, fontsize=12, fontweight='bold', color=color, va='center')
        for x, lab in ((self.X_IN, 'res\n(int32)'), (self.X_B, 'biais'), (self.X_S1, 'S1\na = res + b'),
                       (self.X_S2, f'S2\nReLU, × {CFG["M"]}'), (self.X_S3, f'S3\n>> {CFG["s"]}, sat'),
                       (self.X_MEM + (H - 1) * 0.95 / 2, 'activation_buffer — banc OUT (mots écrits)')):
            ax.text(x, -0.85, lab, ha='center', va='center', fontsize=8, color=TEXT_2)
        for l in range(N):
            ax.text(-0.85, l, f'v{l}', ha='center', va='center', fontsize=8, color=TEXT_2)
            ax.plot([self.X_IN, self.X_S3], [l, l], color=GRID, lw=1, zorder=0)
            for x in (self.X_S1, self.X_S2, self.X_S3):
                ax.add_patch(plt.Rectangle((x - 0.68, l - 0.32), 1.36, 0.64, fc=SURFACE, ec=EDGE, lw=1.1))
        ax.plot([self.X_MEM - 0.6] * 2, [-0.5, N - 0.5], color=GRID, lw=1)
        self.mem = Layer(ax, 1, lw=0.8, edge='white')
        self.mem_txt = TextLayer(ax, 2)
        self.marks = Layer(ax, 5, lw=2.0, edge=ERR)
        self.tokens = TokenLayers(ax, 6)
        self.tag_txt = TextLayer(ax, 8)

    def wide(self, x, y, value, color, text_color='white', alpha=1.0, width=1.2, ec=None):
        v, c = box_path(width, 0.1, 0.5)
        self.tokens.boxes.add(v + (x, y), c, color, alpha, ec=ec)
        self.tokens.labels.text(x, y, str(value), text_color, alpha, pt=7.5)

    def draw(self, recs, c, s):
        for layer in (self.mem, self.mem_txt, self.marks, self.tokens, self.tag_txt):
            layer.reset()
        live = c < len(recs)
        r = recs[min(c, len(recs) - 1)]
        e = ease(s / 0.45)
        M, sh = CFG['M'], CFG['s']
        if live:
            # entrée : res et biais sélectionnés pendant ce cycle
            valid, res, tag = r['inp']
            if valid:
                for l in range(N):
                    self.wide(self.X_IN, l, res[l], C_P, TEXT, alpha=ease(s / 0.3), width=1.05)
                    self.wide(self.X_B, l, r['bias'][l], C_W, alpha=ease(s / 0.3), width=0.95)
                desc, _ = describe(self.variant, tag)
                idx = r['bias_idx']
                lab = f'b[{idx[0]}]' if len(set(idx)) == 1 else f'b[{idx[0]}…{idx[-1]}]'
                self.tag_txt.text(self.X_IN + 0.7, N - 0.35, f'{desc}   {lab}', TEXT_2, bold=False, pt=7)
            # étages : la valeur de l'étage k vient de l'étage k − 1 au cycle précédent
            prev = recs[c - 1] if c > 0 else None
            for k, (key, x0, x1) in enumerate((('s1', self.X_IN, self.X_S1), ('s2', self.X_S1, self.X_S2),
                                               ('s3', self.X_S2, self.X_S3))):
                v, vals, _ = r[key]
                if not v:
                    continue
                for l in range(N):
                    x = x0 + (x1 - x0) * e
                    if key == 's3':
                        # saturé si la valeur avant saturation (calculée depuis S2 au cycle précédent) sort de l'int8
                        p = prev['s2'][1][l] if prev and prev['s2'][0] else 0
                        q = (p + ((1 << (sh - 1)) if sh else 0)) >> sh
                        sat = q > 127 or q < -128
                        self.wide(x, l, vals[l], C_Y, width=1.2)
                        if sat and s > 0.45:
                            mv, mc = box_path(1.36, 0.1, 0.64)
                            self.marks.add(mv + (self.X_S3, l), mc, 'none')
                    elif key == 's2':
                        a = prev['s1'][1][l] if prev and prev['s1'][0] else None
                        clipped = CFG['relu'] and a is not None and a < 0
                        self.wide(x, l, vals[l], CELL if clipped else C_P, TEXT_2 if clipped else TEXT, width=1.2)
                    else:
                        self.wide(x, l, vals[l], C_P, TEXT, width=1.2)
        # mémoire : les mots écrits (y_we pendant le cycle → écrit au front)
        ecrits = []
        for k in range(min(c + 1, len(recs))):
            v, y, tag = recs[k]['s3']
            if v and (k < c or s >= 0.5 or not live):
                ecrits.append((y, tag))
        hv, hc = RECT
        for h in range(H):
            k = len(ecrits) - H + h
            x = self.X_MEM + h * 0.95
            if k < 0:
                for l in range(N):
                    self.mem.add(hv * 0.84 + (x, l), hc, CELL)
                continue
            y, tag = ecrits[k]
            desc, mask = describe(self.variant, tag)
            for l in range(N):
                on = (mask >> l) & 1
                self.mem.add(hv * 0.84 + (x, l), hc, C_Y if on else '#f4f3ef', 0.9 if on else 1)
                self.mem_txt.text(x, l, str(y[l]) if on else '—', 'white' if on else TEXT_2, pt=7)
            self.mem_txt.text(x, N - 0.35, desc, TEXT_2, bold=False, pt=5.5)
        return [self.mem.commit(), self.mem_txt.commit(), self.marks.commit()] + self.tokens.commit() + \
            [self.tag_txt.commit()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='enregistre l\'animation (gif) au lieu de l\'afficher')
    ap.add_argument('--fps', type=int, default=30, help='images par seconde')
    ap.add_argument('--cycle-ms', type=int, default=900, help='durée d\'un cycle d\'horloge à l\'écran (ms)')
    args = ap.parse_args()
    SUB = max(1, round(args.fps * args.cycle_ms / 1000))

    rec_os, ref_os = capture('OS', Mr=4, K=4, Nd=3, seed=8)     # voie 3 masquée (j = 3 ≥ N)
    rec_ws, ref_ws = capture('WS', Mr=3, K=4, Nd=8, seed=9)     # voie 3 masquée (i = 3 ≥ M)
    n_cycles = max(len(rec_os), len(rec_ws)) + 2
    n_frames = n_cycles * SUB

    plt.rcParams['figure.facecolor'] = SURFACE
    fig = plt.figure(figsize=(15, 8.2))
    gs = fig.add_gridspec(2, 1, hspace=0.05)
    P_os = Panel(fig.add_subplot(gs[0]), 'OS', C_OS)
    P_ws = Panel(fig.add_subplot(gs[1]), 'WS', C_WS)
    title = title_axes(fig, (0, 0.95, 1, 0.05)).text(0.5, 0.75, '', ha='center', va='top', fontsize=14,
                                                      fontweight='bold', color=TEXT, animated=True)
    fig.text(0.5, 0.935, f'y = sat( (ReLU(res + b) × {CFG["M"]} + 2^{CFG["s"] - 1}) >> {CFG["s"]} )   ·   '
             'jeton gris en S2 : valeur négative mise à 0 par la ReLU   ·   cadre rouge en S3 : saturation   ·   '
             '— : voie masquée, non écrite', ha='center', fontsize=9, color=TEXT_2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01)

    def frame(f):
        c, s = divmod(f % n_frames, SUB)
        s /= SUB
        title.set_text(f'post_processing N = {N}  —  cycle {c} (depuis 2 cycles avant le premier résultat)')
        return P_os.draw(rec_os, c, s) + P_ws.draw(rec_ws, c, s) + [title]

    if args.save:
        for a in fig.findobj(lambda a: a.get_animated()):
            a.set_animated(False)
        anim = FuncAnimation(fig, frame, frames=n_frames, blit=False)
        anim.save(args.save, writer=PillowWriter(fps=args.fps), dpi=72)
        print(f'animation enregistrée : {args.save}')
        return

    t0 = time.perf_counter()

    def clock_frames():
        while True:
            yield int((time.perf_counter() - t0) * args.fps)

    anim = FuncAnimation(fig, frame, frames=clock_frames, interval=1000 / args.fps, blit=True,
                         cache_frame_data=False)
    plt.show()
    return anim


if __name__ == '__main__':
    main()
