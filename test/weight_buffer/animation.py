"""
Animation du weight_buffer : la mémoire des poids vue comme une carte, mot par mot.

Le chemin de données complet (avec ses mémoires) exécute un GEMM 8×12×8, piloté comme le fera l'hôte.
Chaque case est un mot de 32 bits à l'adresse {i_t, k} (4 poids W[4·i_t + l][k]) :
  - pendant le chargement, l'hôte écrit les mots un par un (jetons venant de la gauche) ;
  - pendant le calcul, chaque lecture du cœur part vers le data_feeder (jetons vers la droite)
    et la case fonce un peu plus à chaque lecture.

  - OS (en haut) : chaque ligne de tuiles de W est relue pour chaque tuile de colonnes de X
  - WS (en bas)  : chaque mot n'est lu qu'une fois, par rafales de 4 (préchargement, k décroissant)

La vitesse dépend de la phase (le chargement passe vite). Rendu ≥ 30 images/s (voir test/viz.py).
Ce fichier fournit aussi la carte mémoire et la ligne de temps réutilisées par l'animation de l'activation_buffer.

Lancement : python test/weight_buffer/animation.py [--save fichier.gif] [--fps 30]
"""

import sys
import time
import bisect
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_test_module
from viz import (SURFACE, TEXT, TEXT_2, CELL, GRID, C_OS, C_WS, C_W,
                 ease, box_path, RECT, Layer, TextLayer, TokenLayers, title_axes)
import layout

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgb

AB = load_test_module('activation_buffer')
EDGE = '#b9b8b2'
VITESSE = {'chargement': 40, 'calcul': 9, 'relecture': 30}      # cycles d'horloge par seconde affichée
VOL = 3                                                        # durée d'un transfert à l'écran (cycles)


# --- Ligne de temps à vitesse variable ------------------------------------------------------------------------

class Timeline:
    """Associe un temps affiché (secondes) à un cycle du journal et à une fraction s ∈ [0, 1) de ce cycle."""

    def __init__(self, log, vitesse=VITESSE):
        self.debut = [0.0]
        for e in log:
            self.debut.append(self.debut[-1] + 1 / vitesse[e['phase']])
        self.duree = self.debut[-1]
        self.n = len(log)

    def locate(self, t):
        if t >= self.duree:
            return self.n, 0.0
        c = bisect.bisect_right(self.debut, t) - 1
        return c, (t - self.debut[c]) / (self.debut[c + 1] - self.debut[c])


def melange(c1, c2, a):
    r1, r2 = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple(r1 + (r2 - r1) * min(1.0, max(0.0, a)))


# --- Carte mémoire ----------------------------------------------------------------------------------------------

class MemGrid:
    """
    Carte d'une zone mémoire : une case par mot, ligne = indice de tuile t, colonne = indice u (adresse {t, u}).
    Les cases chargées prennent la couleur `color`, de plus en plus foncée à chaque lecture par le cœur.
    """

    def __init__(self, ax, x0, y0, rows, cols, color, titre, row_label, col_label, dark=None):
        self.ax, self.x0, self.y0, self.rows, self.cols = ax, x0, y0, rows, cols
        self.color, self.dark = color, dark or color
        ax.text(x0 - 0.5, y0 - 1.05, titre, fontsize=9, fontweight='bold', color=TEXT, va='center')
        for r in range(rows):
            ax.text(x0 - 0.65, y0 + r, f'{row_label} = {r}', ha='right', va='center', fontsize=7, color=TEXT_2)
        for c in range(cols):
            ax.text(x0 + c, y0 + rows - 0.35, str(c), ha='center', va='center', fontsize=6, color=TEXT_2)
        ax.text(x0 + (cols - 1) / 2, y0 + rows + 0.15, col_label, ha='center', va='center', fontsize=7, color=TEXT_2)
        self.cells = Layer(ax, 1, lw=1.0, edge='white')
        self.glow = Layer(ax, 3, lw=2.2, edge=TEXT)
        self.counts = TextLayer(ax, 2)

    def pos(self, a):
        return self.x0 + (a & ((1 << layout.IW) - 1)), self.y0 + (a >> layout.IW)

    def draw(self, charges, lectures, actifs, s, vmax, chiffres=True):
        for l in (self.cells, self.glow, self.counts):
            l.reset()
        cv, cc = box_path(0.86, 0.08)
        for r in range(self.rows):
            for c in range(self.cols):
                a = (r << layout.IW) | c
                x, y = self.x0 + c, self.y0 + r
                n = lectures.get(a, 0)
                if a in charges:
                    fc = melange('#f1f0ec', self.color, 0.35 + 0.65 * (n / vmax if vmax else 0))
                    if n:
                        fc = melange(self.color, self.dark, (n - 1) / max(1, vmax - 1))
                else:
                    fc = CELL
                self.cells.add(cv + (x, y), cc, fc)
                if n and chiffres:
                    self.counts.text(x, y, str(n), 'white', pt=6.5)
                if a in actifs:
                    self.glow.add(cv * 1.12 + (x, y), cc, 'none', ec_alpha=max(0.0, 1 - s))
        return [self.cells.commit(), self.glow.commit(), self.counts.commit()]


# --- Trace ------------------------------------------------------------------------------------------------------

def capture(variant, Mr=8, K=12, Nd=8, seed=170):
    rng = np.random.default_rng(seed)
    W, X = rng.integers(-128, 128, (Mr, K)), rng.integers(-128, 128, (K, Nd))
    b = AB.datapath(variant)
    log = []
    AB.run_layer(b, variant, W, X, [0] * Mr, 1, 10, 1, log=log)
    return log, (Mr, K, Nd)


def etats_cumules(log, maj):
    """
    État avant chaque cycle, précalculé une fois (sinon chaque image relirait tout le journal) :
    etats[c] = état après les cycles 0..c−1. `maj(etat, e)` applique un enregistrement du journal à l'état.
    """
    import copy
    etats, courant = [], None
    for e in log + [None]:
        etats.append(copy.deepcopy(courant) if courant is not None else None)
        if e is not None:
            courant = maj(courant, e)
    return etats


def maj_w(etat, e):
    charges, lectures = etat or (set(), {})
    if e['wb_w'] is not None:
        charges.add(e['wb_w'])
    if e['w_rd'] is not None:
        lectures[e['w_rd']] = lectures.get(e['w_rd'], 0) + 1
    return charges, lectures


class Panel:
    def __init__(self, ax, variant, color, dims, log):
        self.ax, self.variant, self.color, self.log = ax, variant, color, log
        Mr, K, Nd = dims
        rows, cols = layout.tiles(Mr), K
        self.grid = MemGrid(ax, 0, 0, rows, cols, C_W, 'weight_buffer : W rangé en mots {i_t, k}', 'i_t',
                            'k', dark='#241a6b')
        self.x_host, self.x_port = -3.4, cols + 1.6
        ax.set_xlim(self.x_host - 1.2, self.x_port + 2.6)
        ax.set_ylim(rows + 0.9, -2.3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.text(self.x_host - 1.1, -1.95, f'{"output-stationary" if variant == "OS" else "weight-stationary"}',
                fontsize=12, fontweight='bold', color=color, va='center')
        ax.text(self.x_host, rows / 2 - 0.5, 'hôte', ha='center', va='center', fontsize=9, color=TEXT_2)
        ax.text(self.x_port + 0.4, rows / 2 - 0.5, 'data_\nfeeder', ha='left', va='center', fontsize=9, color=TEXT_2)
        self.minimum = len(layout.w_image(np.zeros((Mr, K), int)))
        self.tokens = TokenLayers(ax, 6)
        self.info = ax.text(self.x_port + 2.5, -1.95, '', ha='right', va='center', fontsize=9, color=TEXT,
                            animated=True)
        self.etats = etats_cumules(log, maj_w)
        fin = self.etats[-1] or (set(), {})
        self.vmax = max([1] + list(fin[1].values()))

    def draw(self, c, s):
        self.tokens.reset()
        log = self.log
        cc = min(c, len(log))
        st = self.etats[cc] or (set(), {})
        charges, lectures = set(st[0]), dict(st[1])
        e = log[c] if c < len(log) else None
        actifs = set()
        if e is not None:
            if e['wb_w'] is not None and s > 0.8:
                charges.add(e['wb_w'])
            if e['w_rd'] is not None:
                actifs.add(e['w_rd'])
                if s > 0.5:
                    lectures[e['w_rd']] = lectures.get(e['w_rd'], 0) + 1
        # jetons en vol : chaque transfert dure VOL cycles, ce qui rend visible le flux de mots
        for back in range(VOL):
            k = c - back
            if not 0 <= k < len(log):
                continue
            f, u = log[k], ease((back + s) / VOL)
            if f['wb_w'] is not None:                                   # écriture par l'hôte
                x, y = self.grid.pos(f['wb_w'])
                self.tokens.token(self.x_host + (x - self.x_host) * u, y, 'W', C_W, size=0.55, alpha=1 - u ** 8)
            if f['w_rd'] is not None:                                   # lecture par le cœur
                x, y = self.grid.pos(f['w_rd'])
                self.tokens.token(x + (self.x_port - x) * u, y, 'W', C_W, size=0.55, alpha=1 - u ** 8)
        arts = self.grid.draw(charges, lectures, actifs, s, self.vmax)
        phase = e['phase'] if e else 'terminé'
        self.info.set_text(f'{phase}  ·  lectures de W : {sum(lectures.values())} (minimum {self.minimum})')
        return arts + self.tokens.commit() + [self.info]


def run_animation(fig, panels, timelines, title, label, fps, save):
    """Boucle d'animation commune : temps réel, ≥ 30 images/s, blitting."""
    duree = max(t.duree for t in timelines) + 1.5

    def frame(f):
        t = (f / fps) % duree
        arts = []
        for P, tl in zip(panels, timelines):
            c, s = tl.locate(t)
            arts += P.draw(c, s)
        title.set_text(label(t))
        return arts + [title]

    if save:
        for a in fig.findobj(lambda a: a.get_animated()):
            a.set_animated(False)
        anim = FuncAnimation(fig, frame, frames=int(duree * fps), blit=False)
        anim.save(save, writer=PillowWriter(fps=fps), dpi=72)
        print(f'animation enregistrée : {save}')
        return None

    t0 = time.perf_counter()

    def clock_frames():
        while True:
            yield int((time.perf_counter() - t0) * fps)

    anim = FuncAnimation(fig, frame, frames=clock_frames, interval=1000 / fps, blit=True, cache_frame_data=False)
    plt.show()
    return anim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='enregistre l\'animation (gif) au lieu de l\'afficher')
    ap.add_argument('--fps', type=int, default=30, help='images par seconde')
    args = ap.parse_args()

    log_os, dims = capture('OS')
    log_ws, _ = capture('WS')
    log_os = [e for e in log_os if e['phase'] != 'relecture']          # le weight_buffer ne sert plus ensuite
    log_ws = [e for e in log_ws if e['phase'] != 'relecture']

    plt.rcParams['figure.facecolor'] = SURFACE
    fig = plt.figure(figsize=(14, 7.6))
    gs = fig.add_gridspec(2, 1, hspace=0.05)
    P_os = Panel(fig.add_subplot(gs[0]), 'OS', C_OS, dims, log_os)
    P_ws = Panel(fig.add_subplot(gs[1]), 'WS', C_WS, dims, log_ws)
    title = title_axes(fig, (0, 0.95, 1, 0.05)).text(0.5, 0.75, '', ha='center', va='top', fontsize=14,
                                                      fontweight='bold', color=TEXT, animated=True)
    fig.text(0.5, 0.93, f'GEMM {dims[0]}×{dims[1]}×{dims[2]}   ·   chiffre dans une case : nombre de lectures du mot '
             'par le cœur   ·   le chargement est accéléré', ha='center', fontsize=9, color=TEXT_2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.01)
    tls = [Timeline(log_os), Timeline(log_ws)]
    return run_animation(fig, [P_os, P_ws], tls, title, lambda t: f'weight_buffer — {t:4.1f} s', args.fps, args.save)


if __name__ == '__main__':
    main()
