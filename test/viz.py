"""
Outils graphiques communs aux benchmarks et aux animations.

  - palette et style des figures
  - rendu rapide pour les animations (≥ 30 images/s) : les jetons, cases et chiffres sont des tracés
    regroupés en collections (une seule opération de dessin chacune), à utiliser avec le blitting
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba
from matplotlib.font_manager import FontProperties
from matplotlib.patches import BoxStyle
from matplotlib.path import Path
from matplotlib.textpath import TextPath

from common import setup_matplotlib_fonts

setup_matplotlib_fonts()

# --- Palette -------------------------------------------------------------------------------------

SURFACE = '#fcfcfb'
TEXT = '#0b0b0b'
TEXT_2 = '#52514e'
GRID = '#e4e3df'
REF = '#8a8984'
CELL = '#eeede9'
C_OS = '#2a78d6'         # variante output-stationary
C_WS = '#eb6834'         # variante weight-stationary
TINT_OS = '#d6e5f8'
TINT_WS = '#fbe0d3'
C_W = '#6250d6'          # jetons de poids
C_X = '#138a60'          # jetons d'activation
C_P = '#eda100'          # jetons de somme partielle
ERR = '#e34948'

BENCH_RC = {
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': TEXT_2, 'axes.titlecolor': TEXT,
    'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.labelsize': 9,
    'xtick.color': TEXT_2, 'ytick.color': TEXT_2, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 2, 'lines.markersize': 7,
    'legend.fontsize': 8, 'legend.frameon': False, 'text.color': TEXT,
}


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


# --- Rendu rapide : boîtes arrondies et chiffres sous forme de tracés ------------------------------

_BOXES = {}
_GLYPHS = {}
FONT_BOLD = FontProperties(family='DejaVu Sans', weight='bold')
FONT_REG = FontProperties(family='DejaVu Sans')


def box_path(size, rounding, height=None):
    """Contour d'une boîte arrondie (size × height), centrée sur l'origine (unités de données)."""
    height = size if height is None else height
    key = (size, height, rounding)
    if key not in _BOXES:
        p = BoxStyle.Round(pad=0, rounding_size=rounding)(-size / 2, -height / 2, size, height, 1)
        _BOXES[key] = (np.asarray(p.vertices, float), p.codes)
    return _BOXES[key]


RECT = (np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5]]),
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])


def glyph_path(s, bold, pt):
    """Tracé d'une chaîne, centré sur l'origine, en points typographiques (axe y vers le haut)."""
    key = (s, bold, pt)
    if key not in _GLYPHS:
        tp = TextPath((0, 0), s, size=pt, prop=FONT_BOLD if bold else FONT_REG)
        v = np.asarray(tp.vertices, float)
        if len(v):
            v = v - (v.min(axis=0) + v.max(axis=0)) / 2
        _GLYPHS[key] = (v, tp.codes)
    return _GLYPHS[key]


class Layer:
    """Une collection de tracés redessinée à chaque image (une seule opération de dessin)."""

    def __init__(self, ax, zorder, lw=0.0, edge=None):
        self.ax = ax
        self.col = PathCollection([], transform=ax.transData, zorder=zorder, linewidths=lw,
                                  animated=True)
        self.edge = edge
        ax.add_collection(self.col, autolim=False)
        self.reset()

    def reset(self):
        self.paths, self.fc, self.ec = [], [], []

    def add(self, verts, codes, fc, alpha=1.0, ec_alpha=None, ec=None):
        self.paths.append(Path(verts, codes))
        self.fc.append(to_rgba(fc, alpha) if fc != 'none' else (0, 0, 0, 0))
        if self.edge is not None:
            self.ec.append(to_rgba(ec or self.edge, alpha if ec_alpha is None else ec_alpha))

    def commit(self):
        self.col.set_paths(self.paths)
        self.col.set_facecolors(self.fc if self.fc else 'none')
        if self.edge is not None:
            self.col.set_edgecolors(self.ec if self.ec else 'none')
        return self.col


def data_per_point(ax):
    """Facteurs (x, y) pour convertir des points typographiques en unités de données (y peut être < 0)."""
    t = ax.transData.transform([(0, 0), (1, 1)])
    px = ax.figure.dpi / 72
    return px / (t[1][0] - t[0][0]), px / (t[1][1] - t[0][1])


class TextLayer(Layer):
    """Chiffres et petits textes dessinés comme des tracés (bien plus rapide que des objets Text)."""

    def text(self, x, y, s, color, alpha=1.0, bold=True, pt=8):
        if not s or alpha <= 0:
            return
        v, codes = glyph_path(s, bold, pt)
        if not len(v):
            return
        sx, sy = self.scale
        self.add(np.column_stack((x + v[:, 0] * sx, y + v[:, 1] * sy)), codes, color, alpha)

    def reset(self):
        super().reset()
        self.scale = data_per_point(self.ax)


class TokenLayers:
    """Jetons arrondis avec une valeur : boîtes + chiffres, en deux collections."""

    def __init__(self, ax, zorder=6):
        self.boxes = Layer(ax, zorder, lw=1.2, edge='white')
        self.labels = TextLayer(ax, zorder + 1)

    def reset(self):
        self.boxes.reset()
        self.labels.reset()

    def token(self, x, y, value, color, text_color='white', alpha=1.0, size=0.5, ec=None):
        if alpha <= 0:
            return
        v, codes = box_path(size, 0.1)
        self.boxes.add(v + (x, y), codes, color, alpha, ec=ec)
        self.labels.text(x, y, str(value), text_color, alpha)

    def commit(self):
        return [self.boxes.commit(), self.labels.commit()]


def title_axes(fig, rect=(0, 0.955, 1, 0.045)):
    """Axe invisible pour un titre animé (le blitting ne redessine que des artistes rattachés à un axe)."""
    ax = fig.add_axes(rect)
    ax.axis('off')
    return ax
