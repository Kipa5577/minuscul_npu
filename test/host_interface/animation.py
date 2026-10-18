"""
Animation du host_interface : le bus de 32 bits vu de l'intérieur, transaction par transaction.

Le logiciel hôte pilote la puce complète pour une couche (GEMM 8×6×8, variante OS) :
chargement de W, X et b, configuration, GO, attente de DONE (lecture répétée de STATUS), relecture de Y.
Pour chaque cycle :
  - la transaction en cours (écriture ou lecture, adresse, donnée) et sa fenêtre de destination,
    choisie par bus_addr[12:10] : registres, WEIGHT, ACT_IN, ACT_OUT ou BIAS ;
  - une lecture rend sa donnée au cycle suivant (latence 1) ;
  - la table des registres en direct, et les voyants go (impulsion), occupé (BUSY) et irq (DONE).

La vitesse dépend de la phase. Rendu ≥ 30 images/s (voir test/viz.py).

Lancement : python test/host_interface/animation.py [--save fichier.gif] [--fps 30]
"""

import sys
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_test_module
from viz import (SURFACE, TEXT, TEXT_2, CELL, GRID, C_OS, C_W, C_X, ERR,
                 ease, box_path, Layer, TextLayer, TokenLayers, title_axes)

import matplotlib.pyplot as plt

TOP = load_test_module('npu_top', 'animation.py')
WA = load_test_module('weight_buffer', 'animation.py')
C_Y = '#1baf7a'
C_REG = '#eda100'
C_B = '#9085e9'
VITESSE = {'chargement': 12, 'configuration': 2.5, 'calcul': 14, 'relecture': 5}
FENETRES = [('REG', 'registres', '0x0000', C_REG), ('WEIGHT', 'WEIGHT', '0x0400', C_W),
            ('ACT_IN', 'ACT_IN', '0x0800', C_X), ('ACT_OUT', 'ACT_OUT', '0x0C00', C_Y), ('BIAS', 'BIAS', '0x1000', C_B)]
NOMS_REG = ['ID', 'CTRL', 'STATUS', 'DIM_M', 'DIM_K', 'DIM_N', 'REQ_MULT', 'REQ_SHIFT', 'CYCLES']
ID_OS = 4 | (64 << 8) | (1 << 24)


def valeurs_registres(t):
    r = t['regs']
    return [ID_OS, t['relu'] << 1, t['occupe'] | (t['irq'] << 1), r[3], r[4], r[5], r[6], r[7], t['cycles']]


class Vue:
    X_HOTE, X_DEC, X_FEN, X_REG = 0.0, 4.2, 8.2, 12.6

    def __init__(self, ax, trace):
        self.ax, self.trace = ax, trace
        ax.set_xlim(-1.4, 19.6)
        ax.set_ylim(6.3, -1.6)
        ax.set_aspect('equal')
        ax.axis('off')
        self.y_fen = {nom: 0.3 + i * 1.15 for i, (nom, *_) in enumerate(FENETRES)}
        ax.text(self.X_HOTE, 1.0, 'hôte', ha='center', fontsize=10, fontweight='bold', color=TEXT_2)
        ax.add_patch(plt.Rectangle((self.X_HOTE - 1.1, 1.4), 2.2, 2.6, fc='none', ec=GRID, lw=1.2))
        ax.text(self.X_DEC, 0.9, 'décodage\nbus_addr[12:10]', ha='center', fontsize=8, color=TEXT_2)
        ax.add_patch(plt.Rectangle((self.X_DEC - 0.9, 1.4), 1.8, 2.6, fc='none', ec=GRID, lw=1.2))
        ax.annotate('', xy=(self.X_DEC - 0.9, 2.7), xytext=(self.X_HOTE + 1.1, 2.7),
                    arrowprops=dict(arrowstyle='<->', color=GRID, lw=1.6))
        for nom, lab, base, col in FENETRES:
            y = self.y_fen[nom]
            ax.annotate('', xy=(self.X_FEN - 1.4, y), xytext=(self.X_DEC + 0.9, 2.7),
                        arrowprops=dict(arrowstyle='<->', color=GRID, lw=1.0))
        ax.text(self.X_REG + 2.6, -1.2, 'registres', ha='center', fontsize=9, fontweight='bold', color=TEXT_2)
        for i, n in enumerate(NOMS_REG):
            ax.text(self.X_REG, -0.6 + i * 0.62, n, fontsize=8, color=TEXT_2, va='center')
        for i, lab in enumerate(('go', 'occupé (BUSY)', 'irq (DONE)')):
            ax.text(self.X_HOTE - 0.6, 4.75 + i * 0.5, lab, fontsize=8, color=TEXT_2, va='center')
        self.boxes = Layer(ax, 1, lw=1.2, edge='white')
        self.glow = Layer(ax, 2, lw=2.2, edge=C_OS)
        self.lamps = Layer(ax, 3, lw=0.8, edge='white')
        self.tokens = TokenLayers(ax, 6)
        self.txt = TextLayer(ax, 8)

    def draw(self, c, s):
        for l in (self.boxes, self.glow, self.lamps, self.tokens, self.txt):
            l.reset()
        tr = self.trace
        live = c < len(tr)
        t = tr[min(c, len(tr) - 1)]
        prev = tr[c - 1] if 0 < c <= len(tr) else None
        bv, bc = box_path(2.8, 0.1, 0.85)
        cible = TOP.cible(t['addr']) if live and (t['bus_we'] or t['bus_re']) else None
        bloque = cible not in (None, 'REG') and t['occupe']
        for nom, lab, base, col in FENETRES:
            y = self.y_fen[nom]
            on = nom == cible
            self.boxes.add(bv + (self.X_FEN, y), bc, col if on else CELL, 0.9 if on else 1)
            self.txt.text(self.X_FEN - 0.3, y, lab, 'white' if on else TEXT_2, pt=8)
            self.txt.text(self.X_FEN + 0.95, y, base, 'white' if on else TEXT_2, bold=False, pt=6)
            if on:
                self.glow.add(bv * 1.06 + (self.X_FEN, y), bc, 'none', ec=ERR if bloque else C_OS,
                              ec_alpha=1 - 0.5 * s)
        u = ease(s / 0.85)
        # transaction de ce cycle : du hôte vers la fenêtre
        if cible:
            yf = self.y_fen.get(cible, 2.7) if cible != '—' else 2.7
            x0, x1 = self.X_HOTE + 1.1, self.X_FEN - 1.4
            col = dict((n, c_) for n, _, _, c_ in FENETRES).get(cible, TEXT_2)
            self.tokens.token(x0 + (x1 - x0) * u, 2.7 + (yf - 2.7) * u, 'W' if t['bus_we'] else 'R', col, size=0.45)
        # donnée lue au cycle précédent : elle revient vers l'hôte (latence 1)
        if prev and prev['bus_re']:
            cp = TOP.cible(prev['addr'])
            yf = self.y_fen.get(cp, 2.7)
            x0, x1 = self.X_FEN - 1.4, self.X_HOTE + 1.1
            self.tokens.token(x0 + (x1 - x0) * u, yf + (2.7 - yf) * u, 'D', C_OS, size=0.45)
        # carte de la transaction
        if live:
            if t['bus_we']:
                l1, l2 = 'écriture', f'{self._nom(t["addr"])}\n← 0x{t["wdata"]:08X}'
            elif t['bus_re']:
                l1, l2 = 'lecture', f'{self._nom(t["addr"])}'
            else:
                l1, l2 = 'repos', ''
            self.txt.text(self.X_HOTE, 1.9, l1, TEXT, pt=8.5)
            for i, ligne in enumerate(l2.split('\n')):
                self.txt.text(self.X_HOTE, 2.45 + i * 0.45, ligne, TEXT_2, bold=False, pt=7)
            if prev and prev['bus_re']:
                self.txt.text(self.X_HOTE, 3.6, f'→ 0x{t["rdata"]:08X}', C_OS, pt=7.5)
            if bloque:
                self.txt.text(self.X_FEN, -0.75, 'accès ignoré : NPU occupé', ERR, pt=8)
        # registres : la valeur écrite apparaît au front d'horloge (milieu du cycle à l'écran)
        apres = tr[c + 1] if live and c + 1 < len(tr) and s >= 0.5 else t
        vals = valeurs_registres(apres)
        reg_actif = (t['addr'] & 0xF) if cible == 'REG' else None
        for i, v in enumerate(vals):
            y = -0.6 + i * 0.62
            if i == reg_actif:
                rv, rc = box_path(5.6, 0.08, 0.5)
                self.glow.add(rv + (self.X_REG + 2.75, y), rc, 'none', ec=C_REG, ec_alpha=1 - 0.5 * s)
            txt = f'{v}' if i not in (0, 1, 2) else f'0x{v:08X}'
            if i == 1:
                txt += f'  RELU_EN={apres["relu"]}'
            if i == 2:
                txt = f'BUSY={apres["occupe"]}  DONE={apres["irq"]}'
            self.txt.text(self.X_REG + 3.6, y, txt, TEXT, bold=False, pt=7.5)
        # voyants
        for i, (on, col) in enumerate(((t['go'], C_OS), (t['occupe'], C_REG), (t['irq'], C_Y))):
            lv, lc = box_path(0.3, 0.15)
            self.lamps.add(lv + (self.X_HOTE + 2.4, 4.75 + i * 0.5), lc, col if (on and live) else CELL)
        self.txt.text(self.X_FEN, 6.0, f'phase : {t["phase"]}', TEXT, pt=9)
        return [self.boxes.commit(), self.glow.commit(), self.lamps.commit()] + self.tokens.commit() + \
            [self.txt.commit()]

    @staticmethod
    def _nom(addr):
        cb = TOP.cible(addr)
        if cb == 'REG':
            r = addr & 0xF
            return NOMS_REG[r] if r < len(NOMS_REG) else f'REG[{r}]'
        if cb == '—':
            return f'0x{addr:04X} (vide)'
        base = {'WEIGHT': 0x400, 'ACT_IN': 0x800, 'ACT_OUT': 0xC00, 'BIAS': 0x1000}[cb]
        return f'{cb}[{addr - base}]'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='enregistre l\'animation (gif) au lieu de l\'afficher')
    ap.add_argument('--fps', type=int, default=30, help='images par seconde')
    args = ap.parse_args()

    trace = TOP.trace_puce('OS')
    plt.rcParams['figure.facecolor'] = SURFACE
    fig = plt.figure(figsize=(14, 6.6))
    vue = Vue(fig.add_axes((0.01, 0.01, 0.98, 0.86)), trace)
    title = title_axes(fig, (0, 0.93, 1, 0.07)).text(0.5, 0.75, '', ha='center', va='top', fontsize=14,
                                                      fontweight='bold', color=TEXT, animated=True)
    fig.text(0.5, 0.905, 'W : écriture  ·  R : demande de lecture  ·  D : donnée rendue au cycle suivant  ·  '
             'GEMM 8×6×8 en OS  ·  chargement et attente accélérés', ha='center', fontsize=9, color=TEXT_2)
    tl = WA.Timeline(trace, VITESSE)

    def label(t):
        c, _ = tl.locate(t)
        return f'host_interface — cycle {min(c, len(trace) - 1)}'

    return WA.run_animation(fig, [vue], [tl], title, label, args.fps, args.save)


if __name__ == '__main__':
    main()
