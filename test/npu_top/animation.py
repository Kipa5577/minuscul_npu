"""
Animation de npu_top : la puce complète pendant une couche, vue comme un plan de puce.

Le logiciel hôte pilote la puce par son bus (test/npu_top/test.py) pour une couche Y = requant(ReLU(W·X + b)),
GEMM 8×6×8 : chargement de W, X et b, configuration, GO, attente de DONE, relecture de Y.

  - à gauche, l'hôte et le journal des transactions du bus ;
  - au centre, les mémoires mot par mot : les cases se remplissent quand l'hôte écrit, s'illuminent quand le cœur
    lit, et le banc OUT se remplit des mots de Y ;
  - à droite, la grille 4×4 : les valeurs entrent par ses bords, chaque MAC s'allume, et chaque case montre son
    accumulateur (OS) ou son poids (WS) ; puis output_drain et post_processing ;
  - en haut, l'état du controller ; en bas, la frise des phases et les compteurs.
Les paquets de données circulent sur les liens. Toutes les valeurs viennent de la simulation py4hw.

La vitesse dépend de la phase (le chargement et l'attente sont accélérés). Rendu ≥ 30 images/s (voir test/viz.py).
Ce fichier fournit aussi trace_puce(), réutilisée par l'animation du host_interface.

Lancement : python test/npu_top/animation.py [--variant OS|WS] [--save fichier.gif] [--fps 30]
"""

import sys
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import from_c2, unpack, load_test_module
from viz import (SURFACE, TEXT, TEXT_2, CELL, C_OS, C_WS, C_W, C_X,
                 ease, box_path, Layer, TextLayer, TokenLayers, title_axes)
import layout

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

T = load_test_module('npu_top')
WA = load_test_module('weight_buffer', 'animation.py')
N, IW = 4, 6
C_Y = '#1baf7a'
C_P = '#eda100'
C_CFG = '#52514e'
C_B = '#9085e9'
FOND = '#f4f3ef'
BORD = '#dcdbd5'
FIL = '#e2e1db'
DIMS = (8, 6, 8)
VITESSE = {'chargement': 18, 'configuration': 3, 'calcul': 6, 'relecture': 9}
PHASES = [('chargement', '#9085e9'), ('configuration', C_P), ('calcul', '#3a2d9a'), ('relecture', '#e87ba4')]


# --- Trace de la puce ---------------------------------------------------------------------------------------------

def trace_puce(variant, dims=DIMS, seed=220):
    """
    Exécute une couche par le bus et relève, pour chaque cycle : la phase du pilote, l'accès au bus, l'état des
    registres et du controller, les accès mémoire, ce qui entre dans la grille et l'activité de chaque PE.
    """
    M, K, Nd = dims
    rng = np.random.default_rng(seed)
    nz = [v for v in range(-9, 10) if v]               # valeurs non nulles : chaque MAC est visible
    W, X = rng.choice(nz, (M, K)), rng.choice(nz, (K, Nd))
    bias = list(rng.integers(-20, 20, M))
    npu = T.Npu(variant)
    b, d = npu.b, npu.b.dut
    h, c = d.host, d.controller
    wb, ab, pp = d.weight_buffer, d.activation_buffer, d.post_processing
    trace, phase = [], ['chargement']

    def actif_pe(pe):
        if variant == 'OS':
            return bool(pe.w_in.get() and pe.x_in.get())
        return bool(pe.x_in.get() and pe.w_r)          # un MAC peut avoir lieu pendant le 1er cycle de rechargement

    def step(**v):
        b.set(**v)
        b.sim.propagateAll()
        occupe = h._occupe()
        libre = not wb.busy.get()
        if variant == 'OS':
            valeurs = [[from_c2(pe.acc.get(), 32) for pe in row] for row in d.pe_array.pes]
        else:
            valeurs = [[from_c2(pe.w_r, 8) for pe in row] for row in d.pe_array.pes]
        trace.append(dict(
            phase=phase[0],
            bus_we=h.bus_we.get(), bus_re=h.bus_re.get(), addr=h.bus_addr.get(), wdata=h.bus_wdata.get(),
            rdata=h.bus_rdata.get(), go=h.go.get(), busy=c.busy.get(), irq=h.irq.get(), occupe=occupe,
            etat=c.state, regs=dict(h.regs), relu=h.relu_r, cycles=c.cycles.get(),
            t_out=c.t_out, t_mid=c.t_mid, cnt=c.cnt,
            # accès mémoire de l'hôte (acceptés) et du cœur
            wb_hw=wb.host_addr.get() if libre and wb.host_we.get() else None,
            ab_hw=ab.host_addr.get() if libre and ab.host_we.get() else None,
            ab_hr=ab.host_addr.get() if libre and ab.host_re.get() else None,
            bias_hw=pp.bias_host_addr.get() if libre and pp.bias_host_we.get() else None,
            w_rd=wb.busy.get() and wb.core_re.get(), x_rd=ab.busy.get() and ab.core_rd_re.get(),
            w_addr=wb.core_addr.get(), x_addr=ab.core_rd_addr.get(),
            y_wr=ab.busy.get() and ab.core_wr_we.get(), y_addr=ab.core_wr_addr.get(),
            # chemin de données
            west=unpack(d.data_feeder.w_arr.get() if variant == 'OS' else d.data_feeder.x_arr.get(), N, 8),
            north=unpack(d.data_feeder.x_arr.get() if variant == 'OS' else d.data_feeder.w_arr.get(), N, 8),
            feeder=bool(d.data_feeder.w_arr.get() or d.data_feeder.x_arr.get()),
            pes=[[actif_pe(pe) for pe in row] for row in d.pe_array.pes], valeurs=valeurs,
            drain=d.output_drain.res_valid.get(), res=unpack(d.output_drain.res.get(), N, 32),
            post=pp.y_we.get(), y=unpack(pp.y_wdata.get(), N, 8)))
        b.sim.clk(1)
        b.cycle += 1

    b.step = step
    npu.load_w(W)
    npu.load_x(X)
    npu.load_bias(bias)
    phase[0] = 'configuration'
    npu.configure(M, K, Nd, 5, 2)
    npu.start(True)
    phase[0] = 'calcul'
    npu.wait_done()
    npu.read(T.CYCLES)
    phase[0] = 'relecture'
    Y = npu.read_y(M, Nd)
    assert (Y == T.PP.reference(W, X, bias + [0] * 64, 5, 2, 1)).all(), variant
    return trace


def cible(addr):
    return {0: 'REG', 1: 'WEIGHT', 2: 'ACT_IN', 3: 'ACT_OUT', 4: 'BIAS'}.get(addr >> 10, '—')


NOMS_REG = ['ID', 'CTRL', 'STATUS', 'DIM_M', 'DIM_K', 'DIM_N', 'REQ_MULT', 'REQ_SHIFT', 'CYCLES']


def nom_adresse(addr):
    cb = cible(addr)
    if cb == 'REG':
        r = addr & 0xF
        return NOMS_REG[r] if r < len(NOMS_REG) else f'REG[{r}]'
    base = {'WEIGHT': 0x400, 'ACT_IN': 0x800, 'ACT_OUT': 0xC00, 'BIAS': 0x1000}.get(cb)
    return f'{cb}[{addr - base}]' if base is not None else f'0x{addr:04X}'


# --- Géométrie du plan de puce ----------------------------------------------------------------------------------------

def chemin(*pts):
    """Polyligne : renvoie une fonction u ∈ [0, 1] → point, à vitesse constante le long du tracé."""
    pts = np.array(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate(([0], np.cumsum(seg)))

    def at(u):
        dist = min(max(u, 0.0), 1.0) * cum[-1]
        i = min(np.searchsorted(cum, dist, side='right') - 1, len(seg) - 1)
        f = (dist - cum[i]) / seg[i] if seg[i] else 0
        return pts[i] + (pts[i + 1] - pts[i]) * f

    at.pts = pts
    return at


class Plan:
    """Plan de la puce : positions des blocs, des cartes mémoire et des liens."""

    def __init__(self, variant):
        M, K, Nd = DIMS
        T_ = layout.tiles
        self.variant = variant
        # cartes mémoire : (lignes, colonnes) des mots utiles
        self.dim_w = (T_(M), K)
        self.dim_in = (T_(Nd), K) if variant == 'OS' else (T_(K), Nd)
        self.dim_out = (T_(Nd), M) if variant == 'OS' else (T_(M), Nd)
        self.cs = 1.0                                                  # pas des cases mémoire
        self.x_mem = 11.1
        self.y_w, self.y_in, self.y_out = 4.1, 8.4, 12.2
        self.pe0 = (22.4, 5.2)                                          # centre de la case PE(0,0)
        self.pe_pas = 1.25
        self.blocs = {  # nom : (x0, y0, x1, y1)
            'hôte': (0.4, 2.6, 6.6, 14.2),
            'host_interface': (7.4, 2.6, 9.2, 14.2),
            'weight_buffer': (10.2, 2.6, 20.0, 5.9),
            'activation_buffer': (10.2, 6.6, 20.0, 14.2),
            'data_feeder': (20.6, 3.0, 21.5, 10.4),
            'pe_array': (21.7, 3.3, 27.6, 10.6),
            'output_drain': (28.3, 5.0, 31.5, 7.0),
            'post_processing': (28.3, 9.0, 31.5, 11.4),
            'controller': (10.2, 0.0, 31.5, 1.6),
        }
        y_mid_w = self.y_w + (self.dim_w[0] - 1) * self.cs / 2
        y_mid_in = self.y_in + (self.dim_in[0] - 1) * self.cs / 2
        y_mid_out = self.y_out + (self.dim_out[0] - 1) * self.cs / 2
        x_fin = lambda dims: self.x_mem + (dims[1] - 0.5) * self.cs + 0.2
        pe_y = self.pe0[1] + 1.5 * self.pe_pas
        self.liens = {
            'bus': chemin((6.6, 8.4), (7.4, 8.4)),
            'hi_wb': chemin((9.2, 4.4), (10.2, 4.4), (self.x_mem - 0.5, y_mid_w)),
            'hi_in': chemin((9.2, 8.4), (10.2, 8.4), (self.x_mem - 0.5, y_mid_in)),
            'out_hi': chemin((self.x_mem - 0.5, y_mid_out), (10.2, 12.6), (9.2, 12.6)),
            'hi_ctrl': chemin((8.3, 2.6), (8.3, 0.8), (10.2, 0.8)),
            'hi_bias': chemin((8.3, 14.2), (8.3, 15.1), (29.9, 15.1), (29.9, 11.4)),
            'wb_df': chemin((x_fin(self.dim_w), y_mid_w), (19.4, y_mid_w), (20.6, y_mid_w + 0.4)),
            'in_df': chemin((x_fin(self.dim_in), y_mid_in), (19.4, y_mid_in), (20.6, 8.6)),
            'df_pe': chemin((21.5, pe_y), (21.7, pe_y)),
            'pe_od': chemin((27.6, 6.0), (28.3, 6.0)),
            'od_pp': chemin((29.9, 7.0), (29.9, 9.0)),
            'pp_out': chemin((28.3, 10.8), (27.9, 10.8), (27.9, 13.4), (20.0, 13.4), (x_fin(self.dim_out), y_mid_out)),
        }


# --- Animation ----------------------------------------------------------------------------------------------------------

def melange(c1, c2, a):
    r1, r2 = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple(r1 + (r2 - r1) * min(1.0, max(0.0, a)))


class Puce:
    def __init__(self, ax, axt, variant, trace):
        self.ax, self.axt, self.variant, self.tr = ax, axt, variant, trace
        self.color = C_OS if variant == 'OS' else C_WS
        P = self.P = Plan(variant)
        ax.set_xlim(0, 32)
        ax.set_ylim(15.8, -0.4)
        ax.set_aspect('equal')
        ax.axis('off')
        # fond : blocs et liens (statiques)
        for nom, (x0, y0, x1, y1) in P.blocs.items():
            v, _ = box_path(x1 - x0, 0.25, y1 - y0)
            ax.add_patch(plt.Polygon(v + ((x0 + x1) / 2, (y0 + y1) / 2), closed=True, fc=FOND, ec=BORD, lw=1.2,
                                     zorder=0))
            if nom not in ('controller', 'data_feeder'):
                ax.text(x0 + 0.25, y0 + 0.42, nom, fontsize=8.5, fontweight='bold', color=TEXT_2, va='center')
        ax.text(21.05, 2.75, 'data_\nfeeder', fontsize=7, fontweight='bold', color=TEXT_2, ha='center', va='center')
        for lien in P.liens.values():
            ax.plot(lien.pts[:, 0], lien.pts[:, 1], color=FIL, lw=3.2, solid_capstyle='round',
                    solid_joinstyle='round', zorder=0.5)
        ax.text(P.x_mem - 0.4, P.y_in - 0.95, 'banc IN : X ' + ('{j_t, k}' if variant == 'OS' else '{k_t, j}'),
                fontsize=7.5, color=TEXT_2, va='center')
        ax.text(P.x_mem - 0.4, P.y_out - 0.95, 'banc OUT : Y ' + ('{j_t, i}' if variant == 'OS' else '{i_t, j}'),
                fontsize=7.5, color=TEXT_2, va='center')
        # couches dynamiques
        self.cells = Layer(ax, 2, lw=1.0, edge='white')
        self.glow = Layer(ax, 3, lw=2.4, edge=self.color)
        self.vals = TextLayer(ax, 4)
        self.tokens = TokenLayers(ax, 6)
        self.txt = TextLayer(ax, 8)
        # précalculs : état cumulé des mémoires, compteurs, journal du bus
        self.etats = WA.etats_cumules(trace, self._maj)
        self.macs = np.cumsum([sum(map(sum, t['pes'])) for t in trace])
        self.nb_y = np.cumsum([bool(t['y_wr']) for t in trace])
        self.journal = self._journal()
        self._frise()

    # --- état cumulé ---------------------------------------------------------------------------

    @staticmethod
    def _maj(etat, t):
        w, xin, out, lus_w, lus_x = etat or (set(), set(), set(), {}, {})
        if t['wb_hw'] is not None:
            w.add(t['wb_hw'])
        if t['ab_hw'] is not None and not (t['ab_hw'] >> 10):
            xin.add(t['ab_hw'])
        if t['w_rd']:
            lus_w[t['w_addr']] = lus_w.get(t['w_addr'], 0) + 1
        if t['x_rd']:
            lus_x[t['x_addr']] = lus_x.get(t['x_addr'], 0) + 1
        if t['y_wr']:
            out.add(t['y_addr'])
        return w, xin, out, lus_w, lus_x

    def _journal(self):
        lignes, res = [], []
        for k, t in enumerate(self.tr):
            if t['bus_we']:
                lignes.append(('W', f'{nom_adresse(t["addr"])} ← {t["wdata"] & 0xFFFFFFFF:08X}', k))
            elif t['bus_re']:
                lignes.append(('R', f'{nom_adresse(t["addr"])}', k))
            res.append(list(lignes[-12:]))
        return res

    def _frise(self):
        ax, tr = self.axt, self.tr
        n = len(tr)
        ax.set_xlim(-0.5, n + 0.5)
        ax.set_ylim(1.7, -0.9)
        ax.axis('off')
        self.segments = []
        debut = 0
        for k in range(1, n + 1):
            if k == n or tr[k]['phase'] != tr[debut]['phase']:
                col = dict(PHASES)[tr[debut]['phase']]
                ax.add_patch(plt.Rectangle((debut - 0.5, 0), k - debut, 0.55, color=col, alpha=0.22, lw=0))
                self.segments.append((debut, k, col))
                ax.text((debut + k) / 2 - 0.5, -0.45, tr[debut]['phase'], ha='center', fontsize=8.5, color=TEXT_2)
                debut = k
        self.prog = Layer(ax, 2)
        self.curseur = ax.axvline(0, ymin=0.15, ymax=0.62, color=TEXT, lw=1.4, animated=True)
        self.ftxt = TextLayer(ax, 3)

    # --- dessin --------------------------------------------------------------------------------

    def paquet(self, lien, s, lettre, couleur):
        u = ease(s / 0.9)
        x, y = self.P.liens[lien](u)
        self.tokens.token(x, y, lettre, couleur, size=0.52, alpha=1 - max(0.0, u - 0.85) / 0.15)

    def carte(self, x0, y0, dims, remplis, lus, couleur, actif, s, sombre):
        P = self.P
        cv, cc = box_path(P.cs * 0.86, 0.06)
        for r in range(dims[0]):
            for col in range(dims[1]):
                a = (r << IW) | col
                x, y = x0 + col * P.cs, y0 + r * P.cs
                n = lus.get(a, 0)
                if a in remplis:
                    fc = melange(couleur, sombre, min(1.0, n / 4)) if n else melange('#ffffff', couleur, 0.55)
                else:
                    fc = CELL
                self.cells.add(cv + (x, y), cc, fc)
                if a == actif:
                    self.glow.add(cv * 1.25 + (x, y), cc, 'none', ec=TEXT, ec_alpha=1 - 0.6 * s)

    def draw(self, c, s):
        for l in (self.cells, self.glow, self.vals, self.tokens, self.txt, self.prog, self.ftxt):
            l.reset()
        tr, P = self.tr, self.P
        live = c < len(tr)
        k = min(c, len(tr) - 1)
        t = tr[k]
        w_ch, in_ch, out_ch, lus_w, lus_x = self.etats[k] or (set(), set(), set(), {}, {})

        # mémoires : l'écriture prend effet au front (milieu du cycle à l'écran)
        apres = live and s > 0.5
        wb_hw = t['wb_hw'] if live else None
        ab_hw = t['ab_hw'] if live and t['ab_hw'] is not None and not (t['ab_hw'] >> 10) else None
        ab_hr = t['ab_hr'] & 0x3FF if live and t['ab_hr'] is not None and t['ab_hr'] >> 10 else None
        self.carte(P.x_mem, P.y_w, P.dim_w, w_ch | ({wb_hw} if apres and wb_hw is not None else set()), lus_w, C_W,
                   t['w_addr'] if live and t['w_rd'] else wb_hw, s, '#241a6b')
        self.carte(P.x_mem, P.y_in, P.dim_in, in_ch | ({ab_hw} if apres and ab_hw is not None else set()), lus_x, C_X,
                   t['x_addr'] if live and t['x_rd'] else ab_hw, s, '#08402c')
        self.carte(P.x_mem, P.y_out, P.dim_out, out_ch | ({t['y_addr']} if apres and t['y_wr'] else set()), {}, C_Y,
                   t['y_addr'] if live and t['y_wr'] else ab_hr, s, '#08402c')

        # grille : valeur de chaque PE, MAC allumés, valeurs entrantes sur les bords
        pv, pc = box_path(P.pe_pas * 0.84, 0.1)
        for i in range(N):
            for j in range(N):
                x, y = P.pe0[0] + j * P.pe_pas, P.pe0[1] + i * P.pe_pas
                on = live and t['pes'][i][j]
                self.cells.add(pv + (x, y), pc, melange('#ffffff', self.color, 0.3) if on else '#ffffff')
                if on:
                    self.glow.add(pv * 1.08 + (x, y), pc, 'none', ec_alpha=1 - 0.7 * s)
                v = t['valeurs'][i][j]
                if v:
                    self.vals.text(x, y + 0.28, ('Σ ' if self.variant == 'OS' else 'w ') + str(v), TEXT, pt=7.5)
        if live:
            u = ease(s / 0.5)
            cw, cn = (C_W, C_X) if self.variant == 'OS' else (C_X, C_W)
            for l in range(N):
                if t['west'][l]:
                    self.tokens.token(P.pe0[0] - 1.0 + 0.55 * u, P.pe0[1] + l * P.pe_pas - 0.18, t['west'][l], cw,
                                      size=0.48)
                if t['north'][l]:
                    self.tokens.token(P.pe0[0] + l * P.pe_pas, P.pe0[1] - 1.05 + 0.55 * u, t['north'][l], cn,
                                      size=0.48)

        # paquets sur les liens, et blocs qui travaillent
        if live:
            if t['bus_we'] or t['bus_re']:
                self.paquet('bus', s, 'W' if t['bus_we'] else 'R', C_CFG)
            if wb_hw is not None:
                self.paquet('hi_wb', s, 'W', C_W)
            if ab_hw is not None:
                self.paquet('hi_in', s, 'X', C_X)
            if t['bias_hw'] is not None:
                self.paquet('hi_bias', s, 'b', C_B)
            if t['bus_we'] and cible(t['addr']) == 'REG':
                self.paquet('hi_ctrl', s, 'GO' if (t['addr'] & 0xF) == 1 and t['wdata'] & 1 else 'cfg', C_P)
            if ab_hr is not None:
                self.paquet('out_hi', s, 'Y', C_Y)
            if t['w_rd']:
                self.paquet('wb_df', s, 'W', C_W)
            if t['x_rd']:
                self.paquet('in_df', s, 'X', C_X)
            if t['drain']:
                self.paquet('pe_od', s, 'Σ', C_P)
                self.paquet('od_pp', s, 'Σ', C_P)
            if t['post']:
                self.paquet('pp_out', s, 'Y', C_Y)
            for nom, on in (('data_feeder', t['feeder']), ('output_drain', t['drain']),
                            ('post_processing', t['post']),
                            ('host_interface', t['bus_we'] or t['bus_re'] or t['go'])):
                if on:
                    x0, y0, x1, y1 = P.blocs[nom]
                    v, cds = box_path(x1 - x0, 0.25, y1 - y0)
                    self.glow.add(v + ((x0 + x1) / 2, (y0 + y1) / 2), cds, 'none', ec_alpha=0.9 - 0.5 * s)
            if t['drain']:
                self.txt.text(29.9, 6.25, '  '.join(str(v) for v in t['res']), TEXT, bold=False, pt=7)
            if t['post']:
                self.txt.text(29.9, 10.55, '  '.join(str(v) for v in t['y']), TEXT, bold=False, pt=7)

        # controller : pastilles d'états
        etats = ['IDLE', 'CLEAR', 'STREAM', 'FLUSH', 'DRAIN', 'TAIL', 'DONE'] if self.variant == 'OS' else \
            ['IDLE', 'PRELOAD', 'STREAM', 'FLUSH', 'TAIL', 'DONE']
        ev, ec = box_path(2.05, 0.2, 0.75)
        for n_, e in enumerate(etats):
            x = 13.1 + n_ * 2.25
            on = e == t['etat']
            self.cells.add(ev + (x, 0.8), ec, self.color if on else '#ffffff')
            self.vals.text(x, 0.8, e, 'white' if on else TEXT_2, pt=7)
        self.txt.text(11.15, 0.8, 'controller', TEXT_2, pt=8)
        if t['busy']:
            boucle = ('i_t = {}   j_t = {}' if self.variant == 'OS' else 'i_t = {}   k_t = {}').format(t['t_out'],
                                                                                                    t['t_mid'])
            self.txt.text(29.9, 0.8, boucle, self.color, pt=8)

        # hôte : journal des transactions du bus (la plus récente en bas) et voyants
        self.txt.text(3.5, 3.15, 'bus de 32 bits', TEXT_2, bold=False, pt=7.5)
        for n_, (op, texte, kk) in enumerate(self.journal[k]):
            age = k - kk
            couleur = TEXT if age == 0 else TEXT_2
            self.txt.text(3.5, 3.85 + n_ * 0.62, f'{op}  {texte}', couleur, bold=age == 0, alpha=max(0.35, 1 - age / 30),
                          pt=7)
        lv, lc = box_path(0.42, 0.2)
        for n_, (lab, on, col) in enumerate((('GO', t['go'], self.color), ('BUSY', t['occupe'], C_P),
                                             ('irq', t['irq'], C_Y))):
            x = 1.5 + n_ * 2.0
            self.cells.add(lv + (x, 12.9), lc, col if on and live else CELL)
            self.vals.text(x, 13.55, lab, TEXT_2, bold=False, pt=6.5)

        # frise : progression et compteurs
        M, K, Nd = DIMS
        n = len(tr)
        larg = min(c + s, n)
        for a, b, col in self.segments:
            fin = min(b, larg)
            if fin > a:
                self.prog.add(np.array([[a - 0.5, 0], [fin - 0.5, 0], [fin - 0.5, 0.55], [a - 0.5, 0.55], [a - 0.5, 0]]),
                              [1, 2, 2, 2, 79], col, 0.9)
        self.curseur.set_xdata([larg - 0.5])
        self.ftxt.text(n / 2, 1.3, f'cycle {k}   ·   MAC {self.macs[k]} / {M * K * Nd}   ·   mots de Y écrits '
                       f'{self.nb_y[k]} / {P.dim_out[0] * P.dim_out[1]}   ·   registre CYCLES = {t["cycles"]}',
                       TEXT, bold=False, pt=9)
        return [self.cells.commit(), self.glow.commit(), self.vals.commit()] + self.tokens.commit() + \
            [self.txt.commit(), self.prog.commit(), self.curseur, self.ftxt.commit()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=('OS', 'WS'), default='OS', help='variante du cœur de calcul')
    ap.add_argument('--save', help='enregistre l\'animation (gif) au lieu de l\'afficher')
    ap.add_argument('--fps', type=int, default=30, help='images par seconde')
    args = ap.parse_args()

    trace = trace_puce(args.variant)
    plt.rcParams['figure.facecolor'] = SURFACE
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes((0.01, 0.13, 0.98, 0.77))
    axt = fig.add_axes((0.06, 0.015, 0.88, 0.1))
    puce = Puce(ax, axt, args.variant, trace)
    color = C_OS if args.variant == 'OS' else C_WS
    nom = 'output-stationary' if args.variant == 'OS' else 'weight-stationary'
    title = title_axes(fig, (0, 0.94, 1, 0.06)).text(0.5, 0.75, '', ha='center', va='top', fontsize=15,
                                                      fontweight='bold', color=TEXT, animated=True)
    M, K, Nd = DIMS
    fig.text(0.5, 0.922, f'{nom}  ·  couche Y = requant(ReLU(W·X + b)), GEMM {M}×{K}×{Nd}, pilotée par le bus de '
             '32 bits  ·  chargement et attente accélérés', ha='center', fontsize=10, color=color)
    tl = WA.Timeline(trace, VITESSE)

    def label(t):
        c, _ = tl.locate(t)
        return f'npu_top — la puce complète  ·  {trace[min(c, len(trace) - 1)]["phase"]}'

    return WA.run_animation(fig, [puce], [tl], title, label, args.fps, args.save)


if __name__ == '__main__':
    main()
