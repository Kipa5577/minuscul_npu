"""
Animation du pe_array : les données circulent dans la grille, cycle par cycle, en mouvement continu.

Les deux variantes calculent un produit Y = W·X (N = 4, flux de longueur L) côte à côte :
  - OS : les jetons W (violet) entrent par l'ouest, les jetons X (vert) par le nord, en escalier.
         Chaque PE(i,j) accumule Y[i][j] ; un cadre lumineux signale chaque MAC.
  - WS : les poids W descendent d'abord dans la grille (préchargement), puis restent en place.
         Les jetons X entrent par l'ouest ; les sommes partielles (jaune) descendent et sortent au sud.

Les valeurs dans la grille sont celles de la simulation py4hw ; les jetons en attente
(hors de la grille) sont les entrées que le banc va injecter aux cycles suivants.

En bas : la matrice Y de chaque variante qui se remplit, et le nombre de PE actifs à chaque cycle.

Rendu : 30 images/s minimum. Les jetons, cases et chiffres sont regroupés en quelques collections
de tracés (un seul appel de dessin chacune), le fond fixe est mis en cache (blitting) et l'image
affichée est calée sur l'horloge réelle, pour une vitesse constante.

Lancement : python test/pe_array/animation.py [--save fichier.gif] [--fps 30] [--cycle-ms 700] [--L longueur]
"""

import sys
import time
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import pack, from_c2, load_test_module
from viz import (SURFACE, TEXT, TEXT_2, CELL, C_OS, C_WS, TINT_OS, TINT_WS, C_W, C_X, C_P, ERR,
                 ease, box_path, Layer, TextLayer, title_axes)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.path import Path
from matplotlib.ticker import MaxNLocator

T = load_test_module('pe_array')
DW, AW = T.DW, T.AW
N = 4
QMAX = 4.4               # distance maximale à laquelle on dessine les jetons en attente


# --- Simulation : une trace par cycle ---------------------------------------------------------

def observe(b, **values):
    """Présente les entrées, lit les entrées des PE pendant le cycle, puis avance d'un front."""
    b.set(**values)
    b.sim.propagateAll()
    ins = {n: [[from_c2(getattr(pe, n).get(), DW) for pe in row] for row in b.dut.pes]
           for n in ('w_in', 'x_in') if hasattr(b.dut.pes[0][0], n)}
    b.sim.clk(1)
    b.cycle += 1
    return ins


def trace_os(W, X):
    K = W.shape[1]
    b = T.bench_os(N)
    recs, acc_prev = [], np.zeros((N, N), dtype=int)
    for c in range(K + 2 * N - 1):
        wl, xl = T.os_lanes(W, X, c)
        ins = observe(b, w_west=wl, x_north=xl, clear=0)
        acc = T.acc_matrix(b)
        active = [[ins['w_in'][i][j] != 0 and ins['x_in'][i][j] != 0 for j in range(N)] for i in range(N)]
        done = np.array([[c >= i + j + K - 1 for j in range(N)] for i in range(N)])
        recs.append(dict(phase='calcul', w=ins['w_in'], x=ins['x_in'], acc=acc, acc_prev=acc_prev,
                         active=active, Y=acc.copy(), done=done))
        acc_prev = acc
    # jetons en attente : (voie, cycle d'injection, valeur)
    q_w = [(i, i + k, W[i][k]) for i in range(N) for k in range(K)]
    q_x = [(j, j + k, X[k][j]) for j in range(N) for k in range(K)]
    return recs, q_w, q_x


def trace_ws(W, X):
    J = X.shape[1]
    b = T.bench_ws(N)
    recs = []
    Y = np.zeros((N, J), dtype=int)
    done = np.zeros((N, J), dtype=bool)

    def state():
        w = np.array([[from_c2(pe.w_r, DW) for pe in row] for row in b.dut.pes])
        p = np.array([[from_c2(pe.psum_out.get(), AW) for pe in row] for row in b.dut.pes])
        return w, p

    for n, k in enumerate(reversed(range(N))):
        ins = observe(b, w_north=pack(W[:, k], DW), x_west=0, load=1)
        w, p = state()
        recs.append(dict(phase='préchargement', load=True, x=ins['x_in'], w=w, psum=p,
                         active=[[False] * N for _ in range(N)], Y=Y.copy(), done=done.copy()))
    for c in range(J + 2 * N - 1):
        ins = observe(b, x_west=T.ws_x_lanes(X, c), w_north=0, load=0)
        w, p = state()
        south = T.psum_south(b)
        for i in range(N):
            j = c - i - N + 1                     # la voie i sort Y[i][j] au pas d'indice j + i + N − 1
            if 0 <= j < J:
                Y[i][j], done[i][j] = south[i], True
        active = [[ins['x_in'][k][i] != 0 and w[k][i] != 0 for i in range(N)] for k in range(N)]
        recs.append(dict(phase='flux', load=False, x=ins['x_in'], w=w, psum=p,
                         active=active, Y=Y.copy(), done=done.copy()))
    q_w = [(i, n, W[i][N - 1 - n]) for i in range(N) for n in range(N)]          # nord, préchargement
    q_x = [(k, N + j + k, X[k][j]) for k in range(N) for j in range(J)]          # ouest
    return recs, q_w, q_x


# --- Panneaux -------------------------------------------------------------------------------------

class Panel:
    """Une grille animée : cases, cadres lumineux, jetons et chiffres."""

    def __init__(self, ax, title, color, tint, west_label, north_label):
        self.ax, self.color, self.tint = ax, color, tint
        ax.set_xlim(-QMAX - 0.6, N + 0.9)
        ax.set_ylim(N + 0.9, -QMAX - 0.6)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=13, fontweight='bold', color=color, pad=2)
        ax.add_patch(plt.Rectangle((-0.55, -0.55), N + 0.1, N + 0.1, fc='none', ec=color, lw=1.5, ls='--'))
        ax.text(-QMAX - 0.5, -0.75, west_label + '  →', fontsize=9, color=TEXT_2, va='bottom')
        ax.text(N + 0.8, -QMAX - 0.5, north_label + '\n↓', fontsize=9, color=TEXT_2, ha='right', va='top')
        self.cells = Layer(ax, 1, lw=1.5, edge='white')
        self.glow = Layer(ax, 4, lw=3.0, edge=color)
        self.vals = TextLayer(ax, 5)
        self.boxes = Layer(ax, 6, lw=1.2, edge='white')
        self.labels = TextLayer(ax, 7)
        self.phase = ax.text(N + 0.8, N + 0.8, '', fontsize=10, fontweight='bold', color=color,
                             ha='right', va='bottom', animated=True)
        self.cell_v, self.cell_c = box_path(0.86, 0.08)
        self.glow_v, self.glow_c = box_path(0.94, 0.1)

    def begin(self):
        for layer in (self.cells, self.glow, self.vals, self.boxes, self.labels):
            layer.reset()

    def token(self, x, y, value, color, text_color='white', alpha=1.0, size=0.5):
        if alpha <= 0:
            return
        v, codes = box_path(size, 0.1)
        self.boxes.add(v + (x, y), codes, color, alpha)
        self.labels.text(x, y, str(value), text_color, alpha)

    def flash(self, active, s):
        a = max(0.0, 1 - abs(s - 0.55) / 0.35) if s >= 0.2 else 0.0
        for r in range(N):
            for c in range(N):
                on = active[r][c]
                self.cells.add(self.cell_v + (c, r), self.cell_c, self.tint if on and s >= 0.45 else CELL)
                if on and a > 0:
                    self.glow.add(self.glow_v + (c, r), self.glow_c, 'none', ec_alpha=a)

    def end(self):
        return [self.cells.commit(), self.glow.commit(), self.vals.commit(),
                self.boxes.commit(), self.labels.commit(), self.phase]


def queue_alpha(pos):
    return min(1.0, max(0.0, (pos + QMAX) / 1.2))


def draw_queues(P, q_w, q_x, c, e, w_axis):
    """Jetons pas encore injectés : position le long de leur voie = (c − c0) − 1 + e."""
    for queue, color, axis in ((q_w, C_W, w_axis), (q_x, C_X, 'nord' if w_axis == 'ouest' else 'ouest')):
        for lane, c0, v in queue:
            pos = (c - c0) - 1 + e
            if c0 > c and pos > -QMAX:
                x, y = (pos, lane) if axis == 'ouest' else (lane, pos)
                P.token(x, y, v, color, alpha=queue_alpha(pos))


NO_ACTIVITY = [[False] * N for _ in range(N)]


def draw_os(P, recs, q_w, q_x, c, s):
    P.begin()
    live = c < len(recs)
    r = recs[min(c, len(recs) - 1)]
    e = ease(s / 0.45)
    draw_queues(P, q_w, q_x, c, e, 'ouest')
    if live:
        for i in range(N):
            for j in range(N):
                if r['w'][i][j]:
                    P.token(j - 1 + e - 0.16, i - 0.12, r['w'][i][j], C_W, size=0.38)
                if r['x'][i][j]:
                    P.token(j + 0.16, i - 1 + e - 0.12, r['x'][i][j], C_X, size=0.38)
    # jetons qui sortent de la grille (est et sud) après le dernier PE
    if 0 < c <= len(recs):
        prev = recs[c - 1]
        for i in range(N):
            if prev['w'][i][N - 1]:
                P.token(N - 1 + e - 0.16, i - 0.12, prev['w'][i][N - 1], C_W, size=0.38, alpha=1 - e)
        for j in range(N):
            if prev['x'][N - 1][j]:
                P.token(j + 0.16, N - 1 + e - 0.12, prev['x'][N - 1][j], C_X, size=0.38, alpha=1 - e)
    P.flash(r['active'] if live else NO_ACTIVITY, s)
    acc = r['acc'] if (s >= 0.5 or not live) else r['acc_prev']
    for i in range(N):
        for j in range(N):
            P.vals.text(j, i + 0.27, f'Σ {acc[i][j]}', TEXT)
    P.phase.set_text(r['phase'] if live else 'terminé')
    return P.end()


def draw_ws(P, recs, q_w, q_x, c, s):
    P.begin()
    live = c < len(recs)
    r = recs[min(c, len(recs) - 1)]
    e = ease(s / 0.45)
    draw_queues(P, q_w, q_x, c, e, 'nord')
    for k in range(N):
        for i in range(N):
            if live and r['load']:                    # poids qui descendent d'une ligne
                if r['w'][k][i]:
                    P.token(i - 0.2, k - 1 + e - 0.2, r['w'][k][i], C_W, size=0.36)
            elif r['w'][k][i]:                        # poids en place
                P.token(i - 0.22, k - 0.22, r['w'][k][i], C_W, size=0.34)
            if live and r['x'][k][i]:
                P.token(i - 1 + e + 0.18, k - 0.2, r['x'][k][i], C_X, size=0.36)
    # sommes partielles : naissent au front (s = 0.5) et descendent d'une ligne
    # pendant la demi-période suivante ; celles de la dernière ligne sortent au sud
    if live and s >= 0.5:
        for k in range(N):
            for i in range(N):
                if r['psum'][k][i]:
                    P.token(i, k + 0.2 + (s - 0.5), r['psum'][k][i], C_P, text_color=TEXT, size=0.4)
    if 0 < c <= len(recs) and s < 0.5:
        prev = recs[c - 1]
        for k in range(N):
            for i in range(N):
                if prev['psum'][k][i]:
                    alpha = 1 - 2 * s if k == N - 1 else 1.0
                    P.token(i, k + 0.7 + s, prev['psum'][k][i], C_P, text_color=TEXT, size=0.4, alpha=alpha)
    P.flash(r['active'] if live else NO_ACTIVITY, s)
    P.phase.set_text(r['phase'] if live else 'terminé')
    return P.end()


class ResultPanel:
    def __init__(self, ax, title, tint, Y):
        self.Y, self.tint = Y, tint
        rows, cols = Y.shape
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=9, color=TEXT_2)
        self.cells = Layer(ax, 1, lw=1.0, edge='white')
        self.txt = TextLayer(ax, 2)
        self.rect = (np.array([[-0.46, -0.46], [0.46, -0.46], [0.46, 0.46], [-0.46, 0.46], [-0.46, -0.46]]),
                     [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])

    def update(self, done, Yhw):
        self.cells.reset()
        self.txt.reset()
        rows, cols = self.Y.shape
        for i in range(rows):
            for j in range(cols):
                ok = done[i][j]
                self.cells.add(self.rect[0] + (j, i), self.rect[1], self.tint if ok else CELL)
                if ok:
                    self.txt.text(j, i, f'{Yhw[i][j]}', TEXT if Yhw[i][j] == self.Y[i][j] else ERR,
                                  bold=False, pt=7.5)
        return [self.cells.commit(), self.txt.commit()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='enregistre l\'animation (gif) au lieu de l\'afficher')
    ap.add_argument('--fps', type=int, default=30, help='images par seconde')
    ap.add_argument('--cycle-ms', type=int, default=700, help='durée d\'un cycle d\'horloge à l\'écran (ms)')
    ap.add_argument('--L', type=int, default=5, help='longueur du flux (K en OS, J en WS)')
    args = ap.parse_args()
    L = args.L
    SUB = max(1, round(args.fps * args.cycle_ms / 1000))         # images par cycle d'horloge

    rng = np.random.default_rng(7)
    nz = [v for v in range(-9, 10) if v]               # valeurs non nulles : chaque jeton est visible
    W_os, X_os = rng.choice(nz, (N, L)), rng.choice(nz, (L, N))
    W_ws, X_ws = rng.choice(nz, (N, N)), rng.choice(nz, (N, L))

    os_recs, os_qw, os_qx = trace_os(W_os, X_os)
    ws_recs, ws_qw, ws_qx = trace_ws(W_ws, X_ws)
    n_cycles = max(len(os_recs), len(ws_recs)) + 2
    n_frames = n_cycles * SUB

    plt.rcParams['figure.facecolor'] = SURFACE
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 6, height_ratios=[4.2, 1.3], hspace=0.12, wspace=0.3)
    P_os = Panel(fig.add_subplot(gs[0, 0:3]), f'Output-stationary  ·  W {N}×{L}  ·  X {L}×{N}',
                 C_OS, TINT_OS, 'W en attente\n(voie i = ligne)', 'X en attente\n(voie j = colonne)')
    P_ws = Panel(fig.add_subplot(gs[0, 3:6]), f'Weight-stationary  ·  W {N}×{N}  ·  X {N}×{L}',
                 C_WS, TINT_WS, 'X en attente\n(voie k = ligne)', 'W à précharger\n(voie i = colonne)')
    R_os = ResultPanel(fig.add_subplot(gs[1, 0]), f'Y OS ({N}×{N})', TINT_OS, W_os @ X_os)
    R_ws = ResultPanel(fig.add_subplot(gs[1, 5]), f'Y WS ({N}×{L})', TINT_WS, W_ws @ X_ws)

    ax_u = fig.add_subplot(gs[1, 1:5])
    u_os = [sum(map(sum, r['active'])) for r in os_recs]
    u_ws = [sum(map(sum, r['active'])) for r in ws_recs]
    ax_u.set_xlim(-0.5, n_cycles - 0.5)
    ax_u.set_ylim(0, N * N + 1.5)
    ax_u.set_facecolor(SURFACE)
    for sp in ('top', 'right'):
        ax_u.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax_u.spines[sp].set_color('#e4e3df')
    ax_u.grid(color='#e4e3df', lw=0.8)
    ax_u.set_axisbelow(True)
    ax_u.set_ylabel('PE actifs', fontsize=9, color=TEXT_2)
    ax_u.set_xlabel('cycle', fontsize=9, color=TEXT_2)
    ax_u.tick_params(labelsize=8, colors=TEXT_2)
    ax_u.axhline(N * N, color='#8a8984', lw=1, ls='--')
    ax_u.yaxis.set_major_locator(MaxNLocator(integer=True))
    l_os, = ax_u.step([], [], where='mid', color=C_OS, lw=2, label='OS', animated=True)
    l_ws, = ax_u.step([], [], where='mid', color=C_WS, lw=2, label='WS', animated=True)
    cursor = ax_u.axvline(0, ymin=0.02, ymax=0.95, color=TEXT_2, lw=1, animated=True)
    ax_u.legend(loc='upper left', fontsize=8, frameon=False, ncols=2)

    # le titre change à chaque cycle : il doit appartenir à un axe pour que le blitting le redessine
    ax_t = title_axes(fig)
    title = ax_t.text(0.5, 0.75, '', ha='center', va='top', fontsize=14, fontweight='bold', color=TEXT,
                      transform=ax_t.transAxes, animated=True)
    fig.text(0.5, 0.952, 'violet : poids W   ·   vert : activation X   ·   jaune : somme partielle   ·   '
             'cadre lumineux : MAC pendant ce cycle', ha='center', fontsize=9, color=TEXT_2)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.06)

    def result_state(recs, c, s):
        if c == 0 and s < 0.5:
            r = recs[0]
            return np.zeros_like(r['done']), r['Y']
        r = recs[min(c, len(recs) - 1)]
        return r['done'], r['Y']

    def frame(f):
        c, s = divmod(f % n_frames, SUB)
        s /= SUB
        artists = draw_os(P_os, os_recs, os_qw, os_qx, c, s)
        artists += draw_ws(P_ws, ws_recs, ws_qw, ws_qx, c, s)
        artists += R_os.update(*result_state(os_recs, c, s))
        artists += R_ws.update(*result_state(ws_recs, c, s))
        k_os, k_ws = min(c + 1, len(u_os)), min(c + 1, len(u_ws))
        l_os.set_data(range(k_os), u_os[:k_os])
        l_ws.set_data(range(k_ws), u_ws[:k_ws])
        cursor.set_xdata([c + s - 0.5])
        title.set_text(f'pe_array {N}×{N}  —  cycle {c}')
        return artists + [l_os, l_ws, cursor, title]

    if args.save:
        # un enregistrement redessine toute la figure : les artistes « animés » doivent y être inclus
        for a in fig.findobj(lambda a: a.get_animated()):
            a.set_animated(False)
        anim = FuncAnimation(fig, frame, frames=n_frames, blit=False)
        anim.save(args.save, writer=PillowWriter(fps=args.fps), dpi=72)
        print(f'animation enregistrée : {args.save}')
        return

    # affichage : l'image montrée dépend de l'horloge réelle -> vitesse constante, même si une image est lente
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
