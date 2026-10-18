"""
Animation du data_feeder : comment un mot SRAM devient un escalier à l'entrée de la grille.

Quatre flux, une variante par colonne :
  - OS : W (voies i) et X (voies j) sont tous deux masqués puis décalés ;
  - WS : W (préchargement) est seulement masqué, X (voies k) est masqué puis décalé.

Pour chaque flux, de gauche à droite :
  SRAM → masque (aligné d'un cycle sur l'émission du controller) → registres de décalage → grille,
puis l'historique de ce qui est entré dans la grille : l'escalier systolique.
Les voies hors limites (dimensions non multiples de N) contiennent des données « sales » en mémoire
(jetons pâles) : le masque les force à 0.

Les valeurs (registres, masques, sorties) sont celles de la simulation py4hw.

Lancement : python test/data_feeder/animation.py [--save fichier.gif] [--fps 30] [--cycle-ms 800]
"""

import sys
import time
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import pack, unpack, from_c2, load_test_module
from viz import (SURFACE, TEXT, TEXT_2, CELL, C_OS, C_WS, C_W, C_X, ERR,
                 ease, box_path, RECT, Layer, TextLayer, TokenLayers, title_axes)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

T = load_test_module('data_feeder')
DW = T.DW
N = 4
H = 9                    # cycles d'historique affichés à l'entrée de la grille
QMAX = 4.2               # profondeur de la file d'attente SRAM affichée
X_OUT = N                # colonne « vers la grille »
X_HIST = N + 1.4         # début de l'historique
OK = '#1baf7a'
MASKED = '#c9c8c2'
C_CTL = '#52514e'


# --- Scénarios ---------------------------------------------------------------------------------------

def scenario_os(rng, M=3, K=5, Nd=4):
    """clear, puis K mots ; W a M = 3 lignes valides (voie 3 sale), X a Nd colonnes."""
    Wm = rng.choice([v for v in range(-9, 10) if v], (N, K))
    Xm = rng.choice([v for v in range(-9, 10) if v], (K, N))
    em = [dict(w=[0] * N, x=[0] * N, w_mask=0, x_mask=0, ctl=1)]
    em += [dict(w=list(Wm[:, k]), x=list(Xm[k]), w_mask=T.mask_of(M, N), x_mask=T.mask_of(Nd, N), ctl=0)
           for k in range(K)]
    return em


def scenario_ws(rng, M=3, K=3, J=5):
    """N mots de préchargement (k = N−1 d'abord ; le mot k = 3 ≥ K est entièrement masqué), puis J colonnes de X."""
    Wm = rng.choice([v for v in range(-9, 10) if v], (N, N))
    Xm = rng.choice([v for v in range(-9, 10) if v], (N, J))
    em = [dict(w=list(Wm[:, k]), x=[0] * N, w_mask=T.mask_of(M, N) if k < K else 0, x_mask=0, ctl=1)
          for k in reversed(range(N))]
    em += [dict(w=[0] * N, x=list(Xm[:, j]), w_mask=0, x_mask=T.mask_of(K, N), ctl=0) for j in range(J)]
    return em


# --- Simulation : état du data_feeder pendant chaque cycle ----------------------------------------------

def trace(variant, em, n_cycles):
    """
    Exécute le data_feeder et enregistre, pour chaque cycle : le mot rendu par la SRAM, le masque aligné,
    le contenu des registres de décalage et ce qui entre dans la grille.
    """
    b = T.bench(variant, N)
    f = b.dut
    idle = dict(w=[0] * N, x=[0] * N, w_mask=0, x_mask=0, ctl=0)
    recs, prev = [], idle
    for c in range(n_cycles):
        cur = em[c] if c < len(em) else idle
        b.set(w_rdata=pack(prev['w'], DW), x_rdata=pack(prev['x'], DW),
              w_mask=cur['w_mask'], x_mask=cur['x_mask'], ctl_in=cur['ctl'])
        b.sim.propagateAll()
        rec = {}
        for s, mask_r, chain, out in (('w', f.w_mask_r, f.w_chain, 'w_arr'), ('x', f.x_mask_r, f.x_chain, 'x_arr')):
            rec[s] = dict(raw=prev[s], mask=[(mask_r >> l) & 1 for l in range(N)],
                          chain=[[from_c2(v, DW) for v in chain[l]] for l in range(N)],
                          out=unpack(b.w[out].get(), N, DW))
        rec['ctl'] = b.w['ctl_arr'].get()
        recs.append(rec)
        b.sim.clk(1)
        prev = cur
    # files d'attente : chaque mot émis au cycle e arrive au masque au cycle e + 1
    queues = {s: [(e + 1, w[s], [(w[f'{s}_mask'] >> l) & 1 for l in range(N)]) for e, w in enumerate(em)
                  if w[f'{s}_mask'] or any(w[s])] for s in ('w', 'x')}
    queues['ctl'] = [e + 1 for e, w in enumerate(em) if w['ctl']]
    return recs, queues


# --- Panneau d'un flux --------------------------------------------------------------------------------------

class StreamPanel:

    def __init__(self, ax, title, color, skew, lane_name, ctl_label=None):
        self.ax, self.color, self.skew, self.ctl_label = ax, color, skew, ctl_label
        rows = N + (1.3 if ctl_label else 0.4)
        ax.set_xlim(-QMAX - 1.6, X_HIST + H + 0.2)
        ax.set_ylim(rows, -1.25)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=11, fontweight='bold', color=TEXT, loc='left', pad=2)
        # étiquettes de colonnes
        for x, lab in ((-QMAX / 2 - 0.5, 'SRAM'), (0, 'masque'), (X_OUT, 'grille'),
                       (X_HIST + (H - 1) / 2, 'entrée de la grille (historique)')):
            ax.text(x, -0.85, lab, ha='center', va='center', fontsize=8, color=TEXT_2)
        if skew:
            ax.text((1 + N - 1) / 2, -0.85, 'décalage', ha='center', va='center', fontsize=8, color=TEXT_2)
        # voies, étage de masque, registres de décalage (statiques)
        for l in range(N):
            ax.text(-QMAX - 1.5, l, f'{lane_name}{l}', ha='left', va='center', fontsize=8, color=TEXT_2)
            ax.plot([-0.5, X_OUT], [l, l], color='#e4e3df', lw=1, zorder=0)
            ax.add_patch(plt.Rectangle((-0.42, l - 0.42), 0.84, 0.84, fc='none', ec=REF_EC, lw=1, ls='--'))
            for r in range(1, (l if skew else 0) + 1):
                ax.add_patch(plt.Rectangle((r - 0.4, l - 0.4), 0.8, 0.8, fc=SURFACE, ec=REF_EC, lw=1.2))
            ax.annotate('', xy=(X_OUT + 0.42, l), xytext=(X_OUT - 0.42, l),
                        arrowprops=dict(arrowstyle='->', color=REF_EC, lw=1.2))
        ax.plot([X_HIST - 0.55, X_HIST - 0.55], [-0.5, N - 0.5], color='#e4e3df', lw=1)
        if ctl_label:
            ax.text(-QMAX - 1.5, N + 0.55, 'ctl', ha='left', va='center', fontsize=8, color=TEXT_2)
        self.hist = Layer(ax, 2, lw=0.8, edge='white')
        self.hist_txt = TextLayer(ax, 3)
        self.badges = Layer(ax, 4)
        self.tokens = TokenLayers(ax, 6)
        self.ctl_tokens = TokenLayers(ax, 8)

    def draw(self, recs, queue, ctl_queue, stream, c, s):
        for layer in (self.hist, self.hist_txt, self.badges, self.tokens, self.ctl_tokens):
            layer.reset()
        live = c < len(recs)
        r = recs[min(c, len(recs) - 1)][stream]
        prev = recs[c - 1][stream] if 0 < c <= len(recs) else None
        e = ease(s / 0.45)
        tok = self.tokens.token

        # 1. file SRAM : mots pas encore arrivés au masque (pâles si la voie est hors limites)
        for a, word, mask in queue:
            pos = (c - a) - 1 + e
            if a > c and pos > -QMAX:
                alpha = min(1.0, (pos + QMAX) / 1.0)
                for l in range(N):
                    if word[l]:
                        tok(pos, l, word[l], self.color, alpha=alpha * (1 if mask[l] else 0.35))
        # 2. étage de masque : le mot arrive, les voies masquées deviennent 0
        if live:
            for l in range(N):
                v, m = r['raw'][l], r['mask'][l]
                if v or m:
                    if m or s < 0.45:
                        tok(-1 + e, l, v, self.color, alpha=1 if m else 0.35)
                    else:
                        tok(0, l, 0, MASKED, text_color=TEXT_2, alpha=max(0.0, 1 - 2 * (s - 0.45)))
                bv, bc = box_path(0.22, 0.05)
                if any(r['raw']) or any(r['mask']):
                    self.badges.add(bv + (-0.55, l - 0.3), bc, OK if m else ERR)
        # 3. registres de décalage : le jeton du registre r vient du registre r − 1 au cycle précédent
        if live and self.skew:
            for l in range(1, N):
                for k, v in enumerate(r['chain'][l]):
                    if v:
                        tok(k + e, l, v, self.color, size=0.46)
        # 4. entrée de la grille : la sortie du cycle (escalier), puis l'historique
        if live:
            for l in range(N):
                if r['out'][l]:
                    src = 0 if not self.skew or l == 0 else l
                    x = X_OUT - 0.9 + 0.9 * e if src else X_OUT
                    tok(x, l, r['out'][l], self.color, size=0.46,
                        alpha=1.0 if s >= 0.15 or src else ease(s / 0.15))
        hv, hc = RECT
        for h in range(H):
            cc = c - (H - 1) + h                # cycle affiché dans cette colonne
            for l in range(N):
                v = recs[cc][stream]['out'][l] if 0 <= cc < len(recs) and (cc < c or s >= 0.5) else 0
                self.hist.add(hv * 0.86 + (X_HIST + h, l), hc, self.color if v else CELL, alpha=0.9 if v else 1)
                if v:
                    self.hist_txt.text(X_HIST + h, l, str(v), 'white', pt=7)
        # 5. ctl (clear / load) : émis avec les adresses, il entre dans la grille un cycle plus tard
        if self.ctl_label:
            cv, cc_ = box_path(0.88, 0.1, 0.42)
            for a in ctl_queue:
                pos = (c - a) - 1 + e
                if a > c and pos > -QMAX:
                    self.ctl_tokens.boxes.add(cv + (pos, N + 0.55), cc_, C_CTL, min(1.0, (pos + QMAX) / 1.0))
                    self.ctl_tokens.labels.text(pos, N + 0.55, self.ctl_label, 'white', pt=7)
            if live and recs[c]['ctl']:
                x = -1 + e if s < 0.5 else 0
                self.ctl_tokens.boxes.add(cv + (x, N + 0.55), cc_, C_CTL)
                self.ctl_tokens.labels.text(x, N + 0.55, self.ctl_label, 'white', pt=7)
                self.ctl_tokens.boxes.add(cv + (X_OUT, N + 0.55), cc_, C_CTL, ease(s / 0.3))
                self.ctl_tokens.labels.text(X_OUT, N + 0.55, self.ctl_label, 'white', ease(s / 0.3), pt=7)
        return ([self.hist.commit(), self.hist_txt.commit(), self.badges.commit()]
                + self.tokens.commit() + self.ctl_tokens.commit())


REF_EC = '#b9b8b2'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='enregistre l\'animation (gif) au lieu de l\'afficher')
    ap.add_argument('--fps', type=int, default=30, help='images par seconde')
    ap.add_argument('--cycle-ms', type=int, default=800, help='durée d\'un cycle d\'horloge à l\'écran (ms)')
    args = ap.parse_args()
    SUB = max(1, round(args.fps * args.cycle_ms / 1000))

    rng = np.random.default_rng(3)
    em_os, em_ws = scenario_os(rng), scenario_ws(rng)
    n_cycles = max(len(em_os), len(em_ws)) + N + H // 2
    rec_os, q_os = trace('OS', em_os, n_cycles)
    rec_ws, q_ws = trace('WS', em_ws, n_cycles)
    n_frames = n_cycles * SUB

    plt.rcParams['figure.facecolor'] = SURFACE
    fig = plt.figure(figsize=(16, 7.4))
    gs = fig.add_gridspec(2, 2, hspace=0.12, wspace=0.06)
    panels = [
        (StreamPanel(fig.add_subplot(gs[0, 0]), 'OS · W (voies i = lignes de W, M = 3)', C_W, True, 'i', 'clear'),
         rec_os, q_os, 'w'),
        (StreamPanel(fig.add_subplot(gs[1, 0]), 'OS · X (voies j = colonnes de X)', C_X, True, 'j'),
         rec_os, q_os, 'x'),
        (StreamPanel(fig.add_subplot(gs[0, 1]), 'WS · W préchargé, non décalé (M = 3, K = 3)', C_W, False, 'i', 'load'),
         rec_ws, q_ws, 'w'),
        (StreamPanel(fig.add_subplot(gs[1, 1]), 'WS · X (voies k = lignes de X, K = 3)', C_X, True, 'k'),
         rec_ws, q_ws, 'x'),
    ]
    fig.text(0.27, 0.875, 'output-stationary', ha='center', fontsize=13, fontweight='bold', color=C_OS)
    fig.text(0.73, 0.875, 'weight-stationary', ha='center', fontsize=13, fontweight='bold', color=C_WS)
    fig.text(0.5, 0.925, 'pastille verte / rouge : voie valide / masquée (masque aligné d\'un cycle)   ·   '
             'jeton pâle : donnée hors limites en mémoire   ·   ctl : clear (OS) / load (WS)',
             ha='center', fontsize=9, color=TEXT_2)
    title = title_axes(fig, (0, 0.945, 1, 0.055)).text(0.5, 0.8, '', ha='center', va='top', fontsize=14, fontweight='bold',
                                 color=TEXT, animated=True)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.01)

    def frame(f):
        c, s = divmod(f % n_frames, SUB)
        s /= SUB
        artists = []
        for P, recs, q, stream in panels:
            artists += P.draw(recs, q[stream], q['ctl'], stream, c, s)
        title.set_text(f'data_feeder N = {N}  —  cycle {c}')
        return artists + [title]

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
