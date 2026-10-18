"""
Test fonctionnel du post_processing (spécification : src/post_processing/interface.md).

  - latence exacte de 3 cycles et débit d'une entrée par cycle
  - choix du biais : le même pour les N voies (OS), un par voie (WS)
  - ReLU, arrondi au plus proche (demi vers +∞), saturation, débordement de l'addition du biais
  - port hôte de la mémoire des biais (latence 1, inactif quand busy = 1)
  - comparaison cycle par cycle avec un modèle de référence
  - de bout en bout : data_feeder → pe_array → output_drain → post_processing sur des GEMM complets,
    Y int8 reconstruit à partir des seules écritures (y_addr, y_wdata, y_wmask)

Lancement : python test/post_processing/test.py
"""

import sys
import pathlib
import random

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import Bench, pack, unpack, load_test_module, run_tests

import numpy as np
from src.post_processing.post_processing import PostProcessing_OS, PostProcessing_WS

OD = load_test_module('output_drain')            # cœur de calcul, planificateur, tags
DW, AW, IW, TAG_W = 8, 32, OD.IW, OD.TAG_W
N = 4
BAW = TAG_W - N - IW
VARIANTS = {'OS': PostProcessing_OS, 'WS': PostProcessing_WS}


# --- Banc unitaire ---------------------------------------------------------------------------------

def bench(variant='OS'):
    cls = VARIANTS[variant]
    b = Bench(
        inputs={'rst_n': 1, 'busy': 1, 'res': N * AW, 'res_valid': 1, 'tag_in': TAG_W, 'relu_en': 1,
                'req_mult': 16, 'req_shift': 5, 'bias_host_addr': IW, 'bias_host_wdata': AW,
                'bias_host_we': 1, 'bias_host_re': 1},
        outputs={'bias_host_rdata': AW, 'y_addr': BAW, 'y_wdata': N * DW, 'y_we': 1, 'y_wmask': N},
        build=lambda s, w: cls(s, 'post', w['rst_n'], w['busy'], w['res'], w['res_valid'], w['tag_in'],
                               w['relu_en'], w['req_mult'], w['req_shift'], w['bias_host_addr'],
                               w['bias_host_wdata'], w['bias_host_we'], w['bias_host_re'], w['bias_host_rdata'],
                               w['y_addr'], w['y_wdata'], w['y_we'], w['y_wmask']),
        signed=('bias_host_rdata',))
    b.step(rst_n=0)
    b.set(rst_n=1, req_mult=1, req_shift=0)
    return b


def load_bias(b, biases):
    """Écrit les biais par le port hôte (busy = 0), puis repasse busy à 1."""
    for i, v in enumerate(biases):
        b.step(busy=0, bias_host_addr=i, bias_host_wdata=int(v) & 0xFFFFFFFF, bias_host_we=1)
    b.step(busy=1, bias_host_we=0)


def y_lanes(b):
    return unpack(b.w['y_wdata'].get(), N, DW)


def tag(y_addr=0, mask=0b1111, row=0):
    return OD.tag_pack(y_addr, mask, row, N)


def run_one(b, res, row=0, cfg=None):
    """Envoie un seul vecteur, attend 3 cycles et renvoie les N octets écrits."""
    b.step(res=pack(res, AW), res_valid=1, tag_in=tag(row=row), **(cfg or {}))
    b.step(res_valid=0)
    b.step()
    assert b.get('y_we') == 1
    return y_lanes(b)


# --- Modèle de référence ---------------------------------------------------------------------------------

def requant(x, M, s, relu):
    """Formule de la spécification, écrite indépendamment (division entière par défaut = arrondi vers −∞)."""
    if relu:
        x = max(0, x)
    q = (x * M + (2 ** (s - 1) if s else 0)) // (2 ** s)
    return min(127, max(-128, q))


def wrap32(v):
    v &= (1 << 32) - 1
    return v - (1 << 32) if v >> 31 else v


class RefPost:
    def __init__(self, variant, bias):
        self.per_lane, self.bias = variant == 'WS', bias
        self.pipe = [None, None, None]

    def step(self, res, valid, tag_in, M, s, relu):
        row = tag_in & ((1 << IW) - 1)
        b = [self.bias[(row + l) % 64 if self.per_lane else row] for l in range(N)]
        entree = (valid, [wrap32(r + bb) for r, bb in zip(res, b)], tag_in)
        # après le front de ce cycle, le registre de sortie contient l'entrée d'il y a 2 cycles
        self.pipe = [entree] + self.pipe[:-1]
        sortie = self.pipe[-1]
        if sortie is None:
            return dict(we=0, mask=0)
        v, a, t = sortie
        return dict(we=v, mask=((t >> IW) & 0xF) if v else 0, addr=t >> (IW + N),
                    y=[requant(x, M, s, relu) for x in a])


# --- Tests unitaires --------------------------------------------------------------------------------------

def test_reset():
    b = bench()
    b.step(res=pack([5] * 4, AW), res_valid=1, tag_in=tag())
    b.step(rst_n=0)
    b.step(rst_n=1, res_valid=0)
    b.step()
    assert b.get('y_we') == 0 and b.get('y_wmask') == 0


def test_latence_3_et_debit_1_par_cycle():
    for v in VARIANTS:
        b = bench(v)
        we = []
        for c in range(10):
            # 4 vecteurs consécutifs aux cycles 0..3 ; res = c pour identifier le vecteur
            b.step(res=pack([c] * 4, AW), res_valid=int(c < 4), tag_in=tag(y_addr=c))
            we.append((b.get('y_we'), y_lanes(b)[0], b.get('y_addr')))
        # entrée au cycle t → y_we pendant le cycle t + 3, c'est-à-dire après le pas d'indice t + 2
        assert [w[0] for w in we] == [0, 0, 1, 1, 1, 1, 0, 0, 0, 0], v
        assert [w[1] for w in we[2:6]] == [0, 1, 2, 3] and [w[2] for w in we[2:6]] == [0, 1, 2, 3], v


def test_biais_os_diffuse_ws_par_voie():
    biases = [10 * i - 300 for i in range(64)]
    b = bench('OS')
    load_bias(b, biases)
    assert run_one(b, [0, 1, 2, 3], row=17) == [requant(biases[17] + l, 1, 0, False) for l in range(4)]
    b = bench('WS')
    load_bias(b, biases)
    assert run_one(b, [0, 1, 2, 3], row=16) == [requant(biases[16 + l] + l, 1, 0, False) for l in range(4)]


def test_relu():
    b = bench()
    assert run_one(b, [-50, -1, 0, 50], cfg=dict(relu_en=1)) == [0, 0, 0, 50]
    assert run_one(b, [-50, -1, 0, 50], cfg=dict(relu_en=0)) == [-50, -1, 0, 50]


def test_arrondi_demi_vers_plus_infini():
    b = bench()
    cas = [(5, 1, 1, 3), (-5, 1, 1, -2), (3, 1, 1, 2), (-3, 1, 1, -1), (7, 1, 2, 2), (6, 1, 2, 2),
           (-6, 1, 2, -1), (-7, 1, 2, -2), (100, 1, 0, 100), (1000, 3, 5, 94), (-1000, 3, 5, -94)]
    for x, M, s, attendu in cas:
        assert requant(x, M, s, False) == attendu, (x, M, s)
        got = run_one(b, [x] * 4, cfg=dict(req_mult=M, req_shift=s, relu_en=0))
        assert got == [attendu] * 4, (x, M, s, got)


def test_saturation_et_valeurs_extremes():
    b = bench()
    cfg = dict(relu_en=0)
    for x, M, s, attendu in [(200, 1, 0, 127), (-200, 1, 0, -128), (2**31 - 1, 65535, 0, 127),
                             (-2**31, 65535, 0, -128), (2**31 - 1, 65535, 31, 127), (-2**31, 65535, 31, -128),
                             (2**31 - 1, 65535, 24, 127), (12345, 65535, 31, 0)]:
        got = run_one(b, [x] * 4, cfg=cfg | dict(req_mult=M, req_shift=s))
        assert got == [requant(x, M, s, False)] * 4 == [attendu] * 4, (x, M, s, got)


def test_biais_deborde_modulo_2_32():
    b = bench()
    load_bias(b, [1] * 64)
    # (2^31 − 1) + 1 boucle sur −2^31 : la ReLU le met à 0
    assert run_one(b, [2**31 - 1] * 4, cfg=dict(relu_en=1)) == [0] * 4
    assert run_one(b, [2**31 - 1] * 4, cfg=dict(relu_en=0)) == [-128] * 4


def test_adresse_et_masque():
    b = bench('WS')
    b.step(res=pack([1, 2, 3, 4], AW), res_valid=1, tag_in=tag(y_addr=0x2A5, mask=0b0101, row=8))
    b.step(res_valid=0)
    assert b.get('y_wmask') == 0
    b.step()
    assert (b.get('y_we'), b.get('y_addr'), b.get('y_wmask')) == (1, 0x2A5, 0b0101)
    b.step()
    assert (b.get('y_we'), b.get('y_wmask')) == (0, 0), 'pas d\'écriture sans donnée valide'


def test_port_hote_des_biais():
    b = bench()
    b.step(busy=0, bias_host_addr=9, bias_host_wdata=(-12345) & 0xFFFFFFFF, bias_host_we=1)
    b.step(busy=0, bias_host_addr=9, bias_host_we=0, bias_host_re=1)
    assert b.get('bias_host_rdata') == -12345                    # latence 1
    # busy = 1 : le port hôte est inactif (ni lecture ni écriture)
    b.step(busy=1, bias_host_addr=9, bias_host_wdata=777, bias_host_we=1, bias_host_re=0)
    b.step(busy=1, bias_host_addr=3, bias_host_we=0, bias_host_re=1)
    assert b.get('bias_host_rdata') == -12345, 'lecture ignorée pendant busy'
    b.step(busy=0, bias_host_addr=9, bias_host_re=1)
    assert b.get('bias_host_rdata') == -12345, 'écriture ignorée pendant busy'


def test_aleatoire_cycle_par_cycle():
    for v in VARIANTS:
        for run in range(4):
            rng = random.Random(100 * run + len(v))
            biases = [rng.randint(-5000, 5000) for _ in range(64)]
            M, s, relu = rng.randrange(1 << 16), rng.randrange(32), rng.randint(0, 1)
            b, ref = bench(v), RefPost(v, biases)
            load_bias(b, biases)
            b.set(req_mult=M, req_shift=s, relu_en=relu)
            for _ in range(300):
                res = [rng.randint(-2**31, 2**31 - 1) if rng.random() < 0.2 else rng.randint(-20000, 20000)
                       for _ in range(N)]
                valid = int(rng.random() < 0.7)
                t = rng.randrange(1 << TAG_W)
                b.step(res=pack(res, AW), res_valid=valid, tag_in=t)
                exp = ref.step(res, valid, t, M, s, relu)
                assert b.get('y_we') == exp['we'] and b.get('y_wmask') == exp['mask'], (v, b.cycle)
                if exp['we']:
                    assert b.get('y_addr') == exp['addr'] and y_lanes(b) == exp['y'], (v, b.cycle)


# --- De bout en bout : cœur de calcul + post_processing ---------------------------------------------------

def gemm_complet(variant, W, X, bias, M, s, relu, seed=0):
    """
    Exécute Y = requant(ReLU(W·X + b)) sur data_feeder → pe_array → output_drain → post_processing.
    Renvoie (Y int8 reconstruit des écritures, nombre d'écritures par élément, liste des écritures).
    """
    Mr, Nd = W.shape[0], X.shape[1]
    b = OD.core(variant, N, post=True)
    for i, v in enumerate(bias):
        b.step(busy=0, bias_host_addr=i, bias_host_wdata=int(v) & 0xFFFFFFFF, bias_host_we=1)
    b.set(busy=1, bias_host_we=0, req_mult=M, req_shift=s, relu_en=relu)
    plan = (OD.plan_os if variant == 'OS' else OD.plan_ws)(W, X, N, np.random.default_rng(seed))
    writes = []

    def hook(c, bb):
        if bb.get('y_we'):
            writes.append((c, bb.get('y_addr'), unpack(bb.w['y_wdata'].get(), N, DW), bb.get('y_wmask')))

    OD.run_core(b, plan, on_cycle=hook)
    Y = np.zeros((Mr, Nd), dtype=int)
    count = np.zeros((Mr, Nd), dtype=int)
    for _, addr, y, mask in writes:
        t, idx = addr >> IW, addr & ((1 << IW) - 1)
        for l in range(N):
            if (mask >> l) & 1:
                i, j = (idx, N * t + l) if variant == 'OS' else (N * t + l, idx)
                Y[i][j] = y[l]
                count[i][j] += 1
    return Y, count, writes


def reference(W, X, bias, M, s, relu):
    acc = W @ X + np.array(bias[:W.shape[0]])[:, None]
    return np.vectorize(lambda x: requant(int(x), M, s, relu))(acc)


def test_bout_en_bout():
    rng = np.random.default_rng(110)
    for variant in ('OS', 'WS'):
        for (Mr, K, Nd), relu in [((4, 4, 4), 1), ((5, 9, 7), 0), ((8, 16, 8), 1), ((10, 6, 3), 0)]:
            W = rng.integers(-128, 128, (Mr, K))
            X = rng.integers(-128, 128, (K, Nd))
            bias = list(rng.integers(-20000, 20000, 64))
            M, s = int(rng.integers(1, 1 << 16)), int(rng.integers(14, 24))
            Y, count, _ = gemm_complet(variant, W, X, bias, M, s, relu)
            assert (count == 1).all(), f'{variant} {Mr}×{K}×{Nd} : écritures {count.tolist()}'
            assert (Y == reference(W, X, bias, M, s, relu)).all(), f'{variant} {Mr}×{K}×{Nd} : Y faux'


def test_bout_en_bout_os_ws_identiques():
    """Les deux variantes doivent écrire exactement le même Y."""
    rng = np.random.default_rng(111)
    W = rng.integers(-128, 128, (7, 12))
    X = rng.integers(-128, 128, (12, 9))
    bias = list(rng.integers(-5000, 5000, 64))
    Y_os, _, _ = gemm_complet('OS', W, X, bias, 300, 12, 1)
    Y_ws, _, _ = gemm_complet('WS', W, X, bias, 300, 12, 1)
    assert (Y_os == Y_ws).all()


if __name__ == '__main__':
    sys.exit(run_tests(globals()))
