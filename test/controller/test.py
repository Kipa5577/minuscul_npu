"""
Test fonctionnel du controller (spécification : src/controller/interface.md).

  - cycle par cycle, le controller émet exactement ce qu'émet le planificateur de référence (plan_os / plan_ws de
    test/output_drain/test.py), lui-même validé de bout en bout : adresses, lectures, masques, ctl, vidage ou
    sideband, tags ; puis un cycle DONE
  - busy, done, cycles ; dimensions invalides ; go ignoré pendant un GEMM ; reset en cours de route ; GEMM successifs
  - de bout en bout : controller + chemin de données complet avec mémoires (tout le NPU sauf host_interface),
    piloté par go / done comme le fera l'hôte, y compris deux couches enchaînées

Lancement : python test/controller/test.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import Bench, load_test_module, run_tests
import layout

import numpy as np
from src.controller.controller import Controller_OS, Controller_WS

OD = load_test_module('output_drain')
PP = load_test_module('post_processing')
AB = load_test_module('activation_buffer')
IW = 6


# --- Banc ----------------------------------------------------------------------------------------------------

def bench(variant='OS', N=4):
    TW = (64 // N).bit_length() - 1
    BAW = TW + IW
    TAG = BAW + N + IW
    lw = max(1, (N - 1).bit_length())
    outs = dict(busy=1, done=1, cycles=32, w_addr=BAW, w_re=1, x_addr=BAW, x_re=1, w_mask=N, x_mask=N, ctl=1, tag=TAG)
    outs |= dict(drain_en=1, drain_row=lw) if variant == 'OS' else dict(s_valid=1, s_j=IW, s_first=1, s_last=1)
    order = ['busy', 'done', 'cycles', 'w_addr', 'w_re', 'x_addr', 'x_re', 'w_mask', 'x_mask', 'ctl', 'tag']

    def build(s, w):
        args = [w['rst_n'], w['go'], w['dim_m'], w['dim_k'], w['dim_n']] + [w[n] for n in order]
        if variant == 'OS':
            return Controller_OS(s, 'ctrl', *args, w['drain_en'], w['drain_row'])
        return Controller_WS(s, 'ctrl', *args, w['s_valid'], w['s_j'], w['s_first'], w['s_last'])

    b = Bench(inputs=dict(rst_n=1, go=1, dim_m=IW + 1, dim_k=IW + 1, dim_n=IW + 1), outputs=outs, build=build)
    b.N, b.variant = N, variant
    b.step(rst_n=0)
    b.set(rst_n=1)
    return b


def emission(b):
    """Ce que le controller émet pendant le cycle courant, au format du planificateur."""
    g = b.get
    e = dict(w_addr=g('w_addr') if g('w_re') else None, x_addr=g('x_addr') if g('x_re') else None,
             w_mask=g('w_mask'), x_mask=g('x_mask'), ctl=g('ctl'), tag=g('tag'))
    if b.variant == 'OS':
        e |= dict(drain_en=g('drain_en'), drain_row=g('drain_row') if g('drain_en') else 0)
    else:
        e |= dict(s_valid=g('s_valid'), s_j=g('s_j'), s_first=g('s_first'), s_last=g('s_last'))
    return e


def attendu(p, variant):
    keys = ['w_addr', 'x_addr', 'w_mask', 'x_mask', 'ctl', 'tag']
    keys += ['drain_en', 'drain_row'] if variant == 'OS' else ['s_valid', 's_j', 's_first', 's_last']
    return {k: p[k] for k in keys}


def lancer(b, M, K, Nd):
    """go au cycle courant ; après le front, les sorties sont celles de la première émission."""
    b.step(go=1, dim_m=M, dim_k=K, dim_n=Nd)
    b.set(go=0)


def plan(variant, M, K, Nd, N=4):
    W, X = np.zeros((M, K), int), np.zeros((K, Nd), int)
    return (OD.plan_os if variant == 'OS' else OD.plan_ws)(W, X, N, np.random.default_rng(0))


DIMS = [(4, 4, 4), (1, 1, 1), (10, 7, 9), (5, 13, 6), (16, 3, 16), (64, 2, 5), (3, 64, 2)]


# --- Tests -----------------------------------------------------------------------------------------------------

def test_identique_au_planificateur():
    for variant in ('OS', 'WS'):
        for N in (2, 4):
            for M, K, Nd in DIMS:
                b = bench(variant, N)
                lancer(b, M, K, Nd)
                for c, p in enumerate(plan(variant, M, K, Nd, N)):
                    assert b.get('busy') == 1, (variant, N, M, K, Nd, c)
                    got, exp = emission(b), attendu(p, variant)
                    assert got == exp, f'{variant} N={N} {M}×{K}×{Nd} cycle {c} ({p["phase"]}) : {got} ≠ {exp}'
                    b.step()
                assert (b.get('done'), b.get('busy')) == (1, 0), 'un cycle DONE après le plan'
                b.step()
                assert (b.get('done'), b.get('busy')) == (0, 0), 'retour à IDLE'


def test_compteur_de_cycles():
    for variant in ('OS', 'WS'):
        b = bench(variant)
        n = len(plan(variant, 8, 8, 8))
        lancer(b, 8, 8, 8)
        b.step()
        assert b.get('cycles') == 1, 'compte depuis la première émission'
        while not b.get('done'):
            b.step()
        b.step()
        assert b.get('cycles') == n + 1, 'plan + cycle DONE'
        b.step()
        assert b.get('cycles') == n + 1, 'conservé jusqu\'au prochain go'
        lancer(b, 4, 4, 4)
        assert b.get('cycles') == 0, 'remis à 0 par go'


def test_dimensions_invalides():
    for variant in ('OS', 'WS'):
        for dims in [(0, 4, 4), (4, 0, 4), (4, 4, 0), (65, 4, 4), (4, 127, 4)]:
            b = bench(variant)
            lancer(b, *dims)
            assert (b.get('done'), b.get('busy')) == (1, 0), (variant, dims)
            assert emission(b) == attendu(OD.IDLE, variant), 'rien n\'est émis'
            b.step()
            assert b.get('done') == 0


def test_go_ignore_pendant_un_gemm():
    for variant in ('OS', 'WS'):
        b = bench(variant)
        lancer(b, 6, 5, 7)
        for c, p in enumerate(plan(variant, 6, 5, 7)):
            assert emission(b) == attendu(p, variant), (variant, c)
            b.step(go=c % 3 == 0, dim_m=1, dim_k=1, dim_n=1)   # go parasites, autres dimensions
        assert b.get('done') == 1


def test_reset_en_cours_de_gemm():
    for variant in ('OS', 'WS'):
        b = bench(variant)
        lancer(b, 8, 8, 8)
        for _ in range(20):
            b.step()
        b.step(rst_n=0)
        b.set(rst_n=1)
        assert (b.get('busy'), b.get('done'), b.get('cycles')) == (0, 0, 0)
        assert emission(b) == attendu(OD.IDLE, variant)
        lancer(b, 3, 3, 3)                                    # repart normalement
        for c, p in enumerate(plan(variant, 3, 3, 3)):
            assert emission(b) == attendu(p, variant)
            b.step()


def test_gemm_successifs():
    for variant in ('OS', 'WS'):
        b = bench(variant)
        for dims in [(5, 6, 7), (12, 2, 3), (1, 9, 1)]:
            lancer(b, *dims)
            for p in plan(variant, *dims):
                assert emission(b) == attendu(p, variant), (variant, dims)
                b.step()
            assert b.get('done') == 1
            b.step()


# --- De bout en bout : tout le NPU sauf host_interface ------------------------------------------------------------

def test_bout_en_bout_pilote_par_go_done():
    rng = np.random.default_rng(180)
    for variant in ('OS', 'WS'):
        b = OD.core(variant, ctrl=True)
        AB.salir(b, rng)
        for (M, K, Nd), relu in [((4, 4, 4), 1), ((9, 11, 6), 0), ((16, 16, 16), 1), ((3, 20, 2), 1)]:
            W = rng.integers(-128, 128, (M, K))
            X = rng.integers(-128, 128, (K, Nd))
            bias = list(rng.integers(-20000, 20000, M))
            Mq, s = int(rng.integers(1, 1 << 16)), int(rng.integers(14, 24))
            Y, st = AB.run_layer(b, variant, W, X, bias, Mq, s, relu)
            assert (Y == PP.reference(W, X, bias + [0] * 64, Mq, s, relu)).all(), f'{variant} {M}×{K}×{Nd}'
            assert st['calcul'] == len(plan(variant, M, K, Nd)) + 2, 'go + plan + DONE'
            assert b.get('cycles') == len(plan(variant, M, K, Nd)) + 1


def test_bout_en_bout_deux_couches():
    rng = np.random.default_rng(181)
    X = rng.integers(-128, 128, (12, 6))
    W1, b1 = rng.integers(-128, 128, (10, 12)), list(rng.integers(-3000, 3000, 10))
    W2, b2 = rng.integers(-128, 128, (5, 10)), list(rng.integers(-3000, 3000, 5))
    Y1_ref = PP.reference(W1, X, b1 + [0] * 64, 300, 13, 1)
    Y2_ref = PP.reference(W2, Y1_ref, b2 + [0] * 64, 200, 12, 0)
    for variant in ('OS', 'WS'):
        b = OD.core(variant, ctrl=True)
        Y1, _ = AB.run_layer(b, variant, W1, X, b1, 300, 13, 1)
        AB.copy_out_to_in(b, layout.y_addresses(10, 6, variant))
        Y2, _ = AB.run_layer(b, variant, W2, Y1, b2, 200, 12, 0, load_x=False)
        assert (Y1 == Y1_ref).all() and (Y2 == Y2_ref).all(), variant


if __name__ == '__main__':
    sys.exit(run_tests(globals()))
