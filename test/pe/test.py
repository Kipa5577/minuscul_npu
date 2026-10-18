"""
Test fonctionnel du pe (spécification : src/pe/interface.md).

Chaque variante est comparée, cycle par cycle, à un modèle de référence Python
écrit directement depuis la spécification.

Lancement : python test/pe/test.py
"""

import sys
import pathlib
import random

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import Bench, wrap, run_tests

import numpy as np
from src.pe.pe import PE_OS, PE_WS


# --- Modèles de référence -------------------------------------------------------

class RefPE_OS:
    def __init__(self, DW=8, AW=32):
        self.AW = AW
        self.w_r = self.x_r = self.acc = 0

    def step(self, rst_n, w_in, x_in, clear):
        if not rst_n:
            self.w_r = self.x_r = self.acc = 0
            return
        self.acc = 0 if clear else wrap(self.acc + w_in * x_in, self.AW)
        self.w_r, self.x_r = w_in, x_in

    def outputs(self):
        return {'w_out': self.w_r, 'x_out': self.x_r, 'acc': self.acc}


class RefPE_WS:
    def __init__(self, DW=8, AW=32):
        self.AW = AW
        self.w_r = self.x_r = self.psum = 0

    def step(self, rst_n, x_in, psum_in, w_load_in, load):
        if not rst_n:
            self.w_r = self.x_r = self.psum = 0
            return
        self.psum = wrap(psum_in + self.w_r * x_in, self.AW)    # ancien poids
        if load:
            self.w_r = w_load_in
        self.x_r = x_in

    def outputs(self):
        return {'x_out': self.x_r, 'psum_out': self.psum, 'w_load_out': self.w_r}


# --- Bancs -----------------------------------------------------------------------

def bench_os(DW=8, AW=32):
    b = Bench(
        inputs={'rst_n': 1, 'w_in': DW, 'x_in': DW, 'clear': 1},
        outputs={'w_out': DW, 'x_out': DW, 'acc': AW},
        build=lambda s, w: PE_OS(s, 'pe', w['rst_n'], w['w_in'], w['x_in'], w['clear'],
                                 w['w_out'], w['x_out'], w['acc']),
        signed=('w_in', 'x_in', 'w_out', 'x_out', 'acc'))
    b.step(rst_n=0)
    b.set(rst_n=1)
    return b


def bench_ws(DW=8, AW=32):
    b = Bench(
        inputs={'rst_n': 1, 'x_in': DW, 'psum_in': AW, 'w_load_in': DW, 'load': 1},
        outputs={'x_out': DW, 'psum_out': AW, 'w_load_out': DW},
        build=lambda s, w: PE_WS(s, 'pe', w['rst_n'], w['x_in'], w['psum_in'], w['w_load_in'],
                                 w['load'], w['x_out'], w['psum_out'], w['w_load_out']),
        signed=('x_in', 'psum_in', 'w_load_in', 'x_out', 'psum_out', 'w_load_out'))
    b.step(rst_n=0)
    b.set(rst_n=1)
    return b


def check(b, ref, ctx=''):
    got = {n: b.get(n) for n in ref.outputs()}
    assert got == ref.outputs(), f'cycle {b.cycle} {ctx}: matériel {got} ≠ référence {ref.outputs()}'


# --- Tests OS ----------------------------------------------------------------------

def test_os_reset():
    b = bench_os()
    for _ in range(5):
        b.step(w_in=7, x_in=-3, clear=0)
    assert b.get('acc') != 0
    b.step(rst_n=0, w_in=7, x_in=-3)
    assert (b.get('w_out'), b.get('x_out'), b.get('acc')) == (0, 0, 0)


def test_os_latence_passage():
    b = bench_os()
    b.step(w_in=42, x_in=-17, clear=0)
    assert (b.get('w_out'), b.get('x_out')) == (42, -17)          # L = 1
    b.step(w_in=0, x_in=0)
    assert (b.get('w_out'), b.get('x_out')) == (0, 0)


def test_os_produit_scalaire():
    rng = np.random.default_rng(1)
    w = rng.integers(-128, 128, 64)
    x = rng.integers(-128, 128, 64)
    b = bench_os()
    b.step(clear=1)
    for wi, xi in zip(w, x):
        b.step(w_in=int(wi), x_in=int(xi), clear=0)
    assert b.get('acc') == int(np.dot(w, x))


def test_os_clear_prioritaire():
    b = bench_os()
    for _ in range(3):
        b.step(w_in=10, x_in=10, clear=0)
    b.step(w_in=10, x_in=10, clear=1)            # le produit de ce cycle est perdu
    assert b.get('acc') == 0
    b.step(w_in=2, x_in=3, clear=0)
    assert b.get('acc') == 6


def test_os_zero_neutre():
    b = bench_os()
    b.step(w_in=5, x_in=5, clear=0)
    for w, x in [(0, 99), (-77, 0), (0, 0)]:
        b.step(w_in=w, x_in=x)
        assert b.get('acc') == 25


def test_os_extremes_int8():
    for w, x in [(-128, -128), (-128, 127), (127, 127), (127, -128)]:
        b = bench_os()
        for _ in range(10):
            b.step(w_in=w, x_in=x, clear=0)
        assert b.get('acc') == 10 * w * x, (w, x)


def test_os_aleatoire():
    random.seed(2)
    b, ref = bench_os(), RefPE_OS()
    for _ in range(2000):
        v = dict(rst_n=int(random.random() > 0.01),
                 w_in=random.randint(-128, 127), x_in=random.randint(-128, 127),
                 clear=int(random.random() < 0.05))
        b.step(**v)
        ref.step(**v)
        check(b, ref, v)


def test_os_parametres():
    for DW, AW in [(4, 8), (4, 16), (16, 40)]:
        random.seed(DW * 100 + AW)
        b, ref = bench_os(DW, AW), RefPE_OS(DW, AW)
        lo, hi = -(1 << (DW - 1)), (1 << (DW - 1)) - 1
        for _ in range(500):
            v = dict(rst_n=1, w_in=random.randint(lo, hi), x_in=random.randint(lo, hi),
                     clear=int(random.random() < 0.02))
            b.step(**v)
            ref.step(**v)
            check(b, ref, f'DW={DW} AW={AW}')


def test_os_debordement_boucle():
    # avec DW=4, AW=8 : (-8)·(-8) = 64, le 2e cycle donne 128 → -128 (modulo 2^8)
    b = bench_os(4, 8)
    b.step(w_in=-8, x_in=-8, clear=0)
    b.step(w_in=-8, x_in=-8)
    assert b.get('acc') == -128


# --- Tests WS ----------------------------------------------------------------------

def test_ws_reset():
    b = bench_ws()
    b.step(w_load_in=5, load=1)
    b.step(x_in=3, psum_in=100, load=0)
    b.step(rst_n=0)
    assert (b.get('x_out'), b.get('psum_out'), b.get('w_load_out')) == (0, 0, 0)


def test_ws_latence_passage():
    b = bench_ws()
    b.step(x_in=-99, load=0)
    assert b.get('x_out') == -99


def test_ws_chargement_et_maintien():
    b = bench_ws()
    b.step(w_load_in=-37, load=1)
    assert b.get('w_load_out') == -37
    for v in (12, -5, 127):                     # load = 0 : le poids est conservé
        b.step(w_load_in=v, load=0)
        assert b.get('w_load_out') == -37


def test_ws_somme_partielle():
    b = bench_ws()
    b.step(w_load_in=-6, load=1)
    for x, p in [(7, 1000), (-128, -5), (0, 42), (127, 0)]:
        b.step(x_in=x, psum_in=p, load=0)
        assert b.get('psum_out') == p + (-6) * x
        assert b.get('x_out') == x


def test_ws_load_utilise_ancien_poids():
    b = bench_ws()
    b.step(w_load_in=2, load=1)
    b.step(x_in=10, psum_in=0, w_load_in=9, load=1)   # chargement de 9 pendant ce cycle
    assert b.get('psum_out') == 20                    # produit calculé avec l'ancien poids 2
    b.step(x_in=10, psum_in=0, load=0)
    assert b.get('psum_out') == 90


def test_ws_debordement_boucle():
    b = bench_ws()
    b.step(w_load_in=1, load=1)
    b.step(x_in=1, psum_in=(1 << 31) - 1, load=0)
    assert b.get('psum_out') == -(1 << 31)


def test_ws_colonne_de_4():
    # 4 PE empilés à la main : la colonne calcule Σ_k W[k]·X[k] avec un décalage d'un cycle par ligne
    rng = np.random.default_rng(3)
    W = rng.integers(-128, 128, 4)
    X = rng.integers(-128, 128, (4, 10))     # 10 colonnes j
    pes = [bench_ws() for _ in range(4)]
    for k in range(4):
        pes[k].step(w_load_in=int(W[k]), load=1)
    psums = [0] * 4
    out = []
    for t in range(10 + 4):
        new = []
        for k in range(4):
            j = t - k                              # la ligne k reçoit X[k][j] au cycle j + k
            x = int(X[k][j]) if 0 <= j < 10 else 0
            p_in = psums[k - 1] if k > 0 else 0
            pes[k].step(x_in=x, psum_in=p_in, load=0)
            new.append(pes[k].get('psum_out'))
        psums = new
        if t >= 3:
            out.append(psums[3])
    assert out[:10] == [int(np.dot(W, X[:, j])) for j in range(10)]


def test_ws_aleatoire():
    random.seed(4)
    b, ref = bench_ws(), RefPE_WS()
    for _ in range(2000):
        v = dict(rst_n=int(random.random() > 0.01),
                 x_in=random.randint(-128, 127), psum_in=random.randint(-(1 << 31), (1 << 31) - 1),
                 w_load_in=random.randint(-128, 127), load=int(random.random() < 0.1))
        b.step(**v)
        ref.step(**v)
        check(b, ref, v)


def test_ws_parametres():
    for DW, AW in [(4, 8), (16, 40)]:
        random.seed(DW * 10 + AW)
        b, ref = bench_ws(DW, AW), RefPE_WS(DW, AW)
        lo, hi = -(1 << (DW - 1)), (1 << (DW - 1)) - 1
        for _ in range(500):
            v = dict(rst_n=1, x_in=random.randint(lo, hi),
                     psum_in=random.randint(-(1 << (AW - 1)), (1 << (AW - 1)) - 1),
                     w_load_in=random.randint(lo, hi), load=int(random.random() < 0.1))
            b.step(**v)
            ref.step(**v)
            check(b, ref, f'DW={DW} AW={AW}')


if __name__ == '__main__':
    sys.exit(run_tests(globals()))
