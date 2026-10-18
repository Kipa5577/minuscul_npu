"""
Animation de l'output_drain : comment les résultats sortent de la grille, cycle par cycle.

Le cœur de calcul complet (data_feeder → pe_array → output_drain) exécute un GEMM, piloté par le
planificateur du test (ce que la FSM du controller émet). Seul l'output_drain est dessiné en détail.

  - OS (en haut) : les accumulateurs de la grille montent pendant le flux ; une fois la tuile terminée,
    l'output_drain lit une ligne par cycle et l'envoie vers le post_processing, avec son tag.
    Les résultats sortent par rafales de N lignes.
  - WS (en bas) : les sommes partielles sortent au sud de la grille en escalier ; les registres de
    redressement les réalignent ; le sideband du controller traverse sa ligne à retard de 2N cycles
    pour arriver en même temps. La première tuile de K écrit la mémoire d'accumulateurs (colonne j),
    la dernière additionne et émet le résultat.

En bas : le nombre d'éléments de Y sortis, pour les deux variantes.

Les valeurs sont celles de la simulation py4hw. Rendu ≥ 30 images/s (voir test/viz.py).

Lancement : python test/output_drain/animation.py [--save fichier.gif] [--fps 30] [--cycle-ms 650]
"""

import sys
import time
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import from_c2, unpack, load_test_module
from viz import (SURFACE, TEXT, TEXT_2, CELL, GRID, C_OS, C_WS, C_P, TINT_OS, TINT_WS,
                 ease, box_path, RECT, Layer, TextLayer, TokenLayers, title_axes)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import MaxNLocator

T = load_test_module('output_drain')
AW, IW = T.AW, T.IW
N = 4
H = 6                     # résultats affichés dans l'historique
C_SB = '#52514e'          # jetons du sideband
EDGE = '#b9b8b2'


# --- Traces ------------------------------------------------------------------------------------------------

def capture(variant, M, K, Nd, seed):
    rng = np.random.default_rng(seed)
    vals = [v for v in range(-5, 6) if v]
    W, X = rng.choice(vals, (M, K)), rng.choice(vals, (K, Nd))
    em = (T.plan_os if variant == 'OS' else T.plan_ws)(W, X, N, rng)
    b = T.core(variant, N)
    recs = []

    def hook(c, b):
        d, arr = b.dut['drain'], b.dut['array']
        r = dict(phase=em[c]['phase'], res=unpack(b.w['res'].get(), N, AW), valid=b.get('res_valid'),
                 tag=b.get('tag_out'))
        if variant == 'OS':
            r['acc'] = [[from_c2(pe.acc.get(), AW) for pe in row] for row in arr.pes]
            r['drain'] = (b.w['drain_en'].get(), b.w['drain_row'].get())
        else:
            r['south'] = [from_c2(pe.psum_out.get(), AW) for pe in arr.pes[N - 1]]
            r['deskew'] = [[from_c2(v, AW) for v in lane] for lane in d.deskew]
            r['sb'] = [dict(sb) if sb else None for sb in d.sideband]
            r['sb_in'] = dict(valid=b.w['s_valid'].get(), j=b.w['s_j'].get(), first=b.w['s_first'].get(),
                              last=b.w['s_last'].get())
            r['mem'] = [[from_c2(v, AW) for v in d.accmem[j]] for j in range(Nd)]
        recs.append(r)

    out = T.run_core(b, em, on_cycle=hook)
    Y, count = T.rebuild_y(variant, out, M, Nd, N)
    assert (count == 1).all() and (Y == W @ X).all(), variant
    elems = np.zeros(len(recs), dtype=int)
    for c, tag, _ in out:
        if c < len(recs):
            elems[c] += bin(T.tag_unpack(tag, N)[1]).count('1')
    return recs, np.cumsum(elems), Nd


def describe(variant, tag):
    y_addr, mask, row = T.tag_unpack(tag, N)
    t, idx = y_addr >> IW, y_addr & ((1 << IW) - 1)
    return f'Y[{idx}, {N * t}..{N * t + N - 1}]' if variant == 'OS' else f'Y[{N * t}..{N * t + N - 1}, {idx}]'


# --- Panneau OS ------------------------------------------------------------------------------------------

class PanelOS:
    X_RES = 5.3
    X_HIST = 6.9

    def __init__(self, ax):
        self.ax = ax
        ax.set_xlim(-1.4, self.X_HIST + H + 0.3)
        ax.set_ylim(N + 0.2, -1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.text(-1.35, -1.25, 'output-stationary', fontsize=13, fontweight='bold', color=C_OS, va='center')
        for x, lab in ((1.5, 'accumulateurs de la grille (Σ)'), (self.X_RES, 'res'),
                       (self.X_HIST + (H - 1) / 2, 'vers le post_processing')):
            ax.text(x, -0.75, lab, ha='center', va='center', fontsize=8, color=TEXT_2)
        for i in range(N):
            ax.text(-0.75, i, f'i{i}', ha='center', va='center', fontsize=8, color=TEXT_2)
            ax.add_patch(plt.Rectangle((self.X_RES - 0.42, i - 0.42), 0.84, 0.84, fc='none', ec=EDGE, lw=1.2))
        ax.plot([self.X_HIST - 0.6] * 2, [-0.5, N - 0.5], color=GRID, lw=1)
        self.cells = Layer(ax, 1, lw=1.5, edge='white')
        self.vals = TextLayer(ax, 2)
        self.band = Layer(ax, 3, lw=2.5, edge=C_OS)
        self.hist = Layer(ax, 2, lw=0.8, edge='white')
        self.hist_txt = TextLayer(ax, 3)
        self.tokens = TokenLayers(ax, 6)
        self.phase = ax.text(self.X_HIST + H - 0.2, -1.25, '', ha='right', va='center', fontsize=10,
                             fontweight='bold', color=C_OS, animated=True)
        self.cell_v, self.cell_c = box_path(0.9, 0.08)

    def draw(self, recs, c, s):
        for l in (self.cells, self.vals, self.band, self.hist, self.hist_txt, self.tokens):
            l.reset()
        live = c < len(recs)
        r = recs[min(c, len(recs) - 1)]
        e = ease(s / 0.6)
        draining, row = r['drain'] if live else (0, 0)
        for i in range(N):
            for j in range(N):
                lit = draining and i == row
                self.cells.add(self.cell_v + (j, i), self.cell_c, TINT_OS if lit else CELL)
                self.vals.text(j, i, f'{r["acc"][i][j]}', TEXT, pt=8)
        if draining:
            bv, bc = box_path(N - 0.02, 0.1, 0.98)
            self.band.add(bv + ((N - 1) / 2, row), bc, 'none', ec_alpha=1 - 0.6 * s)
            for j in range(N):                       # la ligne vole vers le registre res (front de fin de cycle)
                x = j + (self.X_RES - j) * e
                y = row + (j - row) * e
                self.tokens.token(x, y, r['acc'][row][j], C_P, text_color=TEXT, size=0.5)
        if live and r['valid']:
            for l in range(N):
                self.tokens.token(self.X_RES, l, r['res'][l], C_P, text_color=TEXT, size=0.56)
        # historique : les H derniers résultats sortis, le plus récent à droite
        sortis = [(k, recs[k]) for k in range(min(c, len(recs) - 1) + 1) if recs[k]['valid'] and (k < c or s >= 0.5)]
        hv, hc = RECT
        for h in range(H):
            k = len(sortis) - H + h
            for l in range(N):
                self.hist.add(hv * 0.86 + (self.X_HIST + h, l), hc, CELL)
            if k >= 0:
                _, rr = sortis[k]
                for l in range(N):
                    self.hist.add(hv * 0.86 + (self.X_HIST + h, l), hc, C_P, 0.85)
                    self.hist_txt.text(self.X_HIST + h, l, str(rr['res'][l]), TEXT, pt=7)
                self.hist_txt.text(self.X_HIST + h, N - 0.2, describe('OS', rr['tag']).replace('..', '–'),
                                   TEXT_2, bold=False, pt=6)
        self.phase.set_text(r['phase'] if live else 'terminé')
        return [self.cells.commit(), self.vals.commit(), self.band.commit(), self.hist.commit(),
                self.hist_txt.commit()] + self.tokens.commit() + [self.phase]


# --- Panneau WS ------------------------------------------------------------------------------------------

class PanelWS:

    def __init__(self, ax, J):
        self.ax, self.J = ax, J
        self.X_SUM = N + 0.8
        self.X_MEM = N + 2.1
        self.X_RES = self.X_MEM + J + 0.6
        self.X_HIST = self.X_RES + 1.4
        self.Y_SB = N + 0.75
        ax.set_xlim(-2 * N + N - 1.6, self.X_HIST + H + 0.3)
        ax.set_ylim(self.Y_SB + 0.8, -1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.text(-2 * N + N - 1.55, -1.25, 'weight-stationary', fontsize=13, fontweight='bold', color=C_WS,
                va='center')
        for x, lab in ((0, 'sud'), ((N - 1) / 2 + 0.5, 'redressement'), (self.X_SUM, 'Σ'),
                       (self.X_MEM + (J - 1) / 2, 'mémoire d\'accumulateurs'), (self.X_RES, 'res'),
                       (self.X_HIST + (H - 1) / 2, 'vers le post_processing')):
            ax.text(x, -0.75, lab, ha='center', va='center', fontsize=8, color=TEXT_2)
        for j in range(J):
            ax.text(self.X_MEM + j, N - 0.35, f'j{j}', ha='center', va='center', fontsize=7, color=TEXT_2)
        for i in range(N):
            ax.text(-0.9, i, f'i{i}', ha='center', va='center', fontsize=8, color=TEXT_2)
            ax.add_patch(plt.Rectangle((-0.42, i - 0.42), 0.84, 0.84, fc='none', ec=EDGE, lw=1, ls='--'))
            for r in range(1, N - i):
                ax.add_patch(plt.Rectangle((r - 0.4, i - 0.4), 0.8, 0.8, fc=SURFACE, ec=EDGE, lw=1.2))
            ax.add_patch(plt.Rectangle((self.X_SUM - 0.42, i - 0.42), 0.84, 0.84, fc='none', ec=EDGE, lw=1, ls=':'))
            ax.add_patch(plt.Rectangle((self.X_RES - 0.42, i - 0.42), 0.84, 0.84, fc='none', ec=EDGE, lw=1.2))
        # ligne à retard du sideband : 2N registres, le dernier est aligné sur la colonne Σ
        self.sb_x = [self.X_SUM - (2 * N - 1) + p for p in range(2 * N)]
        ax.text(self.sb_x[0] - 1, self.Y_SB - 0.62, 'sideband : ligne à retard de 2N cycles', fontsize=8,
                color=TEXT_2, va='center')
        for x in self.sb_x:
            ax.add_patch(plt.Rectangle((x - 0.45, self.Y_SB - 0.25), 0.9, 0.5, fc=SURFACE, ec=EDGE, lw=1))
        ax.plot([self.X_HIST - 0.6] * 2, [-0.5, N - 0.5], color=GRID, lw=1)
        self.mem = Layer(ax, 1, lw=1.2, edge='white')
        self.mem_txt = TextLayer(ax, 2)
        self.mem_hi = Layer(ax, 3, lw=2.2, edge=C_WS)
        self.hist = Layer(ax, 2, lw=0.8, edge='white')
        self.hist_txt = TextLayer(ax, 3)
        self.sb = TokenLayers(ax, 5)
        self.tokens = TokenLayers(ax, 6)
        self.phase = ax.text(self.X_HIST + H - 0.2, -1.25, '', ha='right', va='center', fontsize=10,
                             fontweight='bold', color=C_WS, animated=True)

    def sb_token(self, x, entry, alpha=1.0):
        if not entry or not entry['valid']:
            return
        lab = f'j{entry["j"]}' + (' F' if entry['first'] else '') + (' L' if entry['last'] else '')
        v, cds = box_path(0.86, 0.08, 0.42)
        self.sb.boxes.add(v + (x, self.Y_SB), cds, C_SB, alpha)
        self.sb.labels.text(x, self.Y_SB, lab, 'white', alpha, pt=6.5)

    def draw(self, recs, c, s):
        for l in (self.mem, self.mem_txt, self.mem_hi, self.hist, self.hist_txt, self.sb, self.tokens):
            l.reset()
        live = c < len(recs)
        r = recs[min(c, len(recs) - 1)]
        e = ease(s / 0.45)
        tok = self.tokens.token
        mv, mc = box_path(0.86, 0.06)
        # mémoire d'accumulateurs (contenu pendant ce cycle ; l'écriture a lieu au front)
        sb = r['sb'][-1] if live else None
        nxt = recs[c + 1]['mem'] if live and c + 1 < len(recs) else r['mem']
        for j in range(self.J):
            for i in range(N):
                written = sb and sb['valid'] and not sb['last'] and sb['j'] == j and s >= 0.5
                val = nxt[j][i] if written else r['mem'][j][i]
                used = any(recs[k]['sb'][-1] and recs[k]['sb'][-1]['valid'] and recs[k]['sb'][-1]['j'] == j
                           and not recs[k]['sb'][-1]['last'] for k in range(min(c + (s >= 0.5), len(recs))))
                self.mem.add(mv + (self.X_MEM + j, i), mc, TINT_WS if used or written else CELL)
                if used or written:
                    self.mem_txt.text(self.X_MEM + j, i, str(val), TEXT, pt=7)
        if live:
            # 1. sortie sud de la grille (escalier) et registres de redressement
            for i in range(N):
                if r['south'][i]:
                    tok(0, i - 0.6 + 0.6 * ease(s / 0.3), r['south'][i], C_P, text_color=TEXT, size=0.46,
                        alpha=ease(s / 0.3))
                for k, v in enumerate(r['deskew'][i]):
                    if v:
                        tok(k + e, i, v, C_P, text_color=TEXT, size=0.46)
            # 2. étage aligné : les N voies partent vers Σ ; lecture / écriture de la colonne j
            if sb and sb['valid']:
                j = sb['j']
                aligned = [r['deskew'][i][-1] if N - 1 - i else r['south'][i] for i in range(N)]
                for i in range(N):
                    x0 = N - 1 - i
                    if s < 0.5:
                        tok(x0 + (self.X_SUM - x0) * ease(s / 0.45), i, aligned[i], C_P, text_color=TEXT, size=0.46)
                    else:
                        total = aligned[i] + (0 if sb['first'] else r['mem'][j][i])
                        if sb['last']:
                            pass                                    # le résultat apparaît dans res au cycle suivant
                        else:
                            xe = self.X_SUM + (self.X_MEM + j - self.X_SUM) * ease((s - 0.5) / 0.45)
                            tok(xe, i, total, C_P, text_color=TEXT, size=0.46)
                bv, bc = box_path(0.98, 0.08, N - 0.02)
                if not sb['first'] and s < 0.55:
                    self.mem_hi.add(bv + (self.X_MEM + j, (N - 1) / 2), bc, 'none', ec_alpha=1.0)
                if not sb['last'] and s >= 0.5:
                    self.mem_hi.add(bv + (self.X_MEM + j, (N - 1) / 2), bc, 'none', ec_alpha=1 - (s - 0.5))
            # 3. registre res (visible le cycle suivant l'émission)
            if r['valid']:
                for i in range(N):
                    tok(self.X_RES, i, r['res'][i], C_P, text_color=TEXT, size=0.56)
            # 4. sideband : entrée du controller, puis 2N registres
            self.sb_token(self.sb_x[0] - 1 + 1 * e if False else self.sb_x[0] - 1, r['sb_in'], alpha=ease(s / 0.3))
            for p, entry in enumerate(r['sb']):
                self.sb_token(self.sb_x[p] - 1 + e, entry)
        # historique des résultats
        sortis = [recs[k] for k in range(min(c, len(recs) - 1) + 1) if recs[k]['valid'] and (k < c or s >= 0.5)]
        hv, hc = RECT
        for h in range(H):
            k = len(sortis) - H + h
            for i in range(N):
                self.hist.add(hv * 0.86 + (self.X_HIST + h, i), hc, C_P if k >= 0 else CELL, 0.85 if k >= 0 else 1)
                if k >= 0:
                    self.hist_txt.text(self.X_HIST + h, i, str(sortis[k]['res'][i]), TEXT, pt=7)
            if k >= 0:
                self.hist_txt.text(self.X_HIST + h, N - 0.2, describe('WS', sortis[k]['tag']).replace('..', '–'),
                                   TEXT_2, bold=False, pt=6)
        self.phase.set_text(r['phase'] if live else 'terminé')
        return [self.mem.commit(), self.mem_txt.commit(), self.mem_hi.commit(), self.hist.commit(),
                self.hist_txt.commit()] + self.sb.commit() + self.tokens.commit() + [self.phase]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='enregistre l\'animation (gif) au lieu de l\'afficher')
    ap.add_argument('--fps', type=int, default=30, help='images par seconde')
    ap.add_argument('--cycle-ms', type=int, default=650, help='durée d\'un cycle d\'horloge à l\'écran (ms)')
    args = ap.parse_args()
    SUB = max(1, round(args.fps * args.cycle_ms / 1000))

    rec_os, cum_os, _ = capture('OS', M=4, K=4, Nd=8, seed=5)          # 2 tuiles de sortie
    rec_ws, cum_ws, J = capture('WS', M=4, K=8, Nd=4, seed=6)          # 2 tuiles de K : écriture puis émission
    n_cycles = max(len(rec_os), len(rec_ws)) + 2
    n_frames = n_cycles * SUB

    plt.rcParams['figure.facecolor'] = SURFACE
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.15, 0.5], hspace=0.08)
    P_os = PanelOS(fig.add_subplot(gs[0]))
    P_ws = PanelWS(fig.add_subplot(gs[1]), J)

    ax_u = fig.add_subplot(gs[2])
    ax_u.set_xlim(-0.5, n_cycles - 0.5)
    ax_u.set_ylim(0, 17 * 1.12)
    ax_u.set_facecolor(SURFACE)
    for sp in ('top', 'right'):
        ax_u.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax_u.spines[sp].set_color(GRID)
    ax_u.grid(color=GRID, lw=0.8)
    ax_u.set_axisbelow(True)
    ax_u.set_ylabel('éléments de Y sortis', fontsize=9, color=TEXT_2)
    ax_u.set_xlabel('cycle', fontsize=9, color=TEXT_2)
    ax_u.tick_params(labelsize=8, colors=TEXT_2)
    ax_u.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_u.set_ylim(0, max(cum_os[-1], cum_ws[-1]) * 1.15)
    l_os, = ax_u.step([], [], where='post', color=C_OS, lw=2, label='OS (rafales)', animated=True)
    l_ws, = ax_u.step([], [], where='post', color=C_WS, lw=2, label='WS (à la dernière tuile de K)', animated=True)
    cursor = ax_u.axvline(0, ymin=0.02, ymax=0.95, color=TEXT_2, lw=1, animated=True)
    ax_u.legend(loc='upper left', fontsize=8, frameon=False, ncols=2)

    title = title_axes(fig, (0, 0.955, 1, 0.045)).text(0.5, 0.75, '', ha='center', va='top', fontsize=14,
                                                        fontweight='bold', color=TEXT, animated=True)
    fig.text(0.5, 0.94, 'jeton jaune : somme partielle / résultat int32   ·   jeton gris : sideband du controller '
             '(F = première tuile de K, L = dernière)   ·   cadre : lecture / écriture', ha='center', fontsize=9,
             color=TEXT_2)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.93, bottom=0.05)

    def frame(f):
        c, s = divmod(f % n_frames, SUB)
        s /= SUB
        artists = P_os.draw(rec_os, c, s) + P_ws.draw(rec_ws, c, s)
        k_os, k_ws = min(c + 1, len(cum_os)), min(c + 1, len(cum_ws))
        l_os.set_data(range(k_os), cum_os[:k_os])
        l_ws.set_data(range(k_ws), cum_ws[:k_ws])
        cursor.set_xdata([c + s])
        title.set_text(f'output_drain N = {N}  —  cycle {c}')
        return artists + [l_os, l_ws, cursor, title]

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
