"""
Animation de l'activation_buffer : le parcours complet d'une couche à travers les deux bancs.

Le chemin de données complet (avec ses mémoires) exécute un GEMM 8×12×8, piloté comme le fera l'hôte :
  hôte → banc IN (X, mot par mot) → cœur de calcul (lectures de X) → banc OUT (écriture des mots de Y)
  → hôte (relecture de Y).
Chaque case est un mot de 32 bits (4 valeurs int8) ; le chiffre indique combien de fois le cœur l'a lu ou écrit.

  - OS (en haut) : X est rangé en mots {j_t, k}, Y est écrit en mots {j_t, i}
  - WS (en bas)  : X est rangé en mots {k_t, j}, Y est écrit en mots {i_t, j}
Dans les deux cas, Y sort au format X de la variante : il peut servir tel quel d'entrée à la couche suivante.

La vitesse dépend de la phase (chargement et relecture accélérés). Rendu ≥ 30 images/s (voir test/viz.py).

Lancement : python test/activation_buffer/animation.py [--save fichier.gif] [--fps 30]
"""

import sys
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_test_module
from viz import SURFACE, TEXT, TEXT_2, C_OS, C_WS, C_X, ease, TokenLayers, title_axes
import layout

import matplotlib.pyplot as plt

WA = load_test_module('weight_buffer', 'animation.py')
C_Y = '#1baf7a'
VOL = WA.VOL


VIDE = (set(), {}, set(), {}, 0)


def maj_ab(etat, e):
    charges_in, lus_in, charges_out, ecrits_out, relus = etat or (set(), {}, set(), {}, 0)
    if e['ab_w'] is not None and e['ab_w'][0] == 0:
        charges_in.add(e['ab_w'][1])
    if e['x_rd'] is not None:
        lus_in[e['x_rd']] = lus_in.get(e['x_rd'], 0) + 1
    if e['y_wr'] is not None:
        charges_out.add(e['y_wr'][0])
        ecrits_out[e['y_wr'][0]] = ecrits_out.get(e['y_wr'][0], 0) + 1
    return charges_in, lus_in, charges_out, ecrits_out, relus + (e['ab_r'] is not None)


class Panel:
    def __init__(self, ax, variant, color, dims, log):
        self.ax, self.variant, self.log = ax, variant, log
        Mr, K, Nd = dims
        T = layout.tiles
        if variant == 'OS':
            xin, yout = (T(Nd), K, 'j_t', 'k', '{j_t, k}'), (T(Nd), Mr, 'j_t', 'i', '{j_t, i}')
        else:
            xin, yout = (T(K), Nd, 'k_t', 'j', '{k_t, j}'), (T(Mr), Nd, 'i_t', 'j', '{i_t, j}')
        rows = max(xin[0], yout[0])
        self.x_host_in = -3.6
        self.in_grid = WA.MemGrid(ax, 0, 0, xin[0], xin[1], C_X, f'banc IN : X en mots {xin[4]}', xin[2], xin[3],
                                  dark='#08402c')
        self.x_core = xin[1] + 1.9
        x_out = self.x_core + 2.9
        self.out_grid = WA.MemGrid(ax, x_out, 0, yout[0], yout[1], C_Y, f'banc OUT : Y en mots {yout[4]}',
                                   yout[2], yout[3], dark='#08402c')
        self.x_host_out = x_out + yout[1] + 1.6
        ax.set_xlim(self.x_host_in - 1.1, self.x_host_out + 1.4)
        ax.set_ylim(rows + 0.9, -2.4)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.text(self.x_host_in - 1.0, -2.05, 'output-stationary' if variant == 'OS' else 'weight-stationary',
                fontsize=12, fontweight='bold', color=color, va='center')
        for x, lab in ((self.x_host_in, 'hôte'), (self.x_core, 'cœur de\ncalcul'), (self.x_host_out, 'hôte')):
            ax.text(x, rows / 2 - 0.5, lab, ha='center', va='center', fontsize=9, color=TEXT_2)
        self.tokens = TokenLayers(ax, 6)
        self.info = ax.text(self.x_host_out + 1.3, -2.05, '', ha='right', va='center', fontsize=9, color=TEXT,
                            animated=True)
        self.etats = WA.etats_cumules(log, maj_ab)
        fin = self.etats[-1] or VIDE
        self.vmax = max([1] + list(fin[1].values()))

    def draw(self, c, s):
        self.tokens.reset()
        log = self.log
        cc = min(c, len(log))
        st = self.etats[cc] or VIDE
        charges_in, lus_in, charges_out, ecrits_out, relus = set(st[0]), dict(st[1]), set(st[2]), dict(st[3]), st[4]
        act_in, act_out = set(), set()
        e = log[c] if c < len(log) else None
        if e is not None:
            if e['x_rd'] is not None:
                act_in.add(e['x_rd'])
            if e['y_wr'] is not None:
                act_out.add(e['y_wr'][0])
                if s > 0.5:
                    charges_out.add(e['y_wr'][0])
                    ecrits_out[e['y_wr'][0]] = ecrits_out.get(e['y_wr'][0], 0) + 1
            if e['ab_w'] is not None and e['ab_w'][0] == 0 and s > 0.8:
                charges_in.add(e['ab_w'][1])
            if e['ab_r'] is not None:
                act_out.add(e['ab_r'][1])
        # jetons en vol (un transfert dure VOL cycles à l'écran)
        for back in range(VOL):
            k = c - back
            if not 0 <= k < len(log):
                continue
            f, u = log[k], ease((back + s) / VOL)
            fade = 1 - u ** 8
            if f['ab_w'] is not None and f['ab_w'][0] == 0:
                x, y = self.in_grid.pos(f['ab_w'][1])
                self.tokens.token(self.x_host_in + (x - self.x_host_in) * u, y, 'X', C_X, size=0.55, alpha=fade)
            if f['x_rd'] is not None:
                x, y = self.in_grid.pos(f['x_rd'])
                self.tokens.token(x + (self.x_core - x) * u, y, 'X', C_X, size=0.55, alpha=fade)
            if f['y_wr'] is not None:
                x, y = self.out_grid.pos(f['y_wr'][0])
                self.tokens.token(self.x_core + (x - self.x_core) * u, y, 'Y', C_Y, size=0.55, alpha=fade)
            if f['ab_r'] is not None and f['ab_r'][0] == 1:
                x, y = self.out_grid.pos(f['ab_r'][1])
                self.tokens.token(x + (self.x_host_out - x) * u, y, 'Y', C_Y, size=0.55, alpha=fade)
        arts = self.in_grid.draw(charges_in, lus_in, act_in, s, self.vmax)
        arts += self.out_grid.draw(charges_out, ecrits_out, act_out, s, 1, chiffres=False)   # toujours 1 écriture
        phase = e['phase'] if e else 'terminé'
        self.info.set_text(f'{phase}  ·  lectures de X : {sum(lus_in.values())}  ·  mots de Y écrits : '
                           f'{len(charges_out)}  ·  relus par l\'hôte : {relus}')
        return arts + self.tokens.commit() + [self.info]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='enregistre l\'animation (gif) au lieu de l\'afficher')
    ap.add_argument('--fps', type=int, default=30, help='images par seconde')
    args = ap.parse_args()

    log_os, dims = WA.capture('OS')
    log_ws, _ = WA.capture('WS')
    # le chargement de W et des biais ne concerne pas l'activation_buffer : on ne garde que ses accès
    garder = lambda e: e['phase'] != 'chargement' or (e['ab_w'] is not None)
    log_os, log_ws = [e for e in log_os if garder(e)], [e for e in log_ws if garder(e)]

    plt.rcParams['figure.facecolor'] = SURFACE
    fig = plt.figure(figsize=(15, 7.8))
    gs = fig.add_gridspec(2, 1, hspace=0.05)
    P_os = Panel(fig.add_subplot(gs[0]), 'OS', C_OS, dims, log_os)
    P_ws = Panel(fig.add_subplot(gs[1]), 'WS', C_WS, dims, log_ws)
    title = title_axes(fig, (0, 0.95, 1, 0.05)).text(0.5, 0.75, '', ha='center', va='top', fontsize=14,
                                                      fontweight='bold', color=TEXT, animated=True)
    fig.text(0.5, 0.93, f'GEMM {dims[0]}×{dims[1]}×{dims[2]}   ·   chiffre dans une case du banc IN : nombre '
             'de lectures du mot par le cœur (chaque mot de Y est écrit une seule fois)   ·   chargement et relecture accélérés', ha='center', fontsize=9,
             color=TEXT_2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.01)
    tls = [WA.Timeline(log_os), WA.Timeline(log_ws)]
    return WA.run_animation(fig, [P_os, P_ws], tls, title, lambda t: f'activation_buffer — {t:4.1f} s',
                            args.fps, args.save)


if __name__ == '__main__':
    main()
