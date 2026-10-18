"""
Test fonctionnel de l'activation_buffer (spécification : src/activation_buffer/interface.md).

  - accès hôte aux deux bancs (bit BAW de l'adresse), latence 1, banc lu mémorisé pour host_rdata
  - lecture cœur du banc IN et écriture cœur du banc OUT avec masque par octet, dans le même cycle
  - un seul côté actif à la fois, choisi par busy ; comparaison avec un modèle de référence
  - disposition D6 : X rangé par l'hôte, relu aux adresses émises par le planificateur du controller
  - chemin de données complet avec ses mémoires (weight_buffer, activation_buffer) : l'hôte charge W, X et b,
    le cœur calcule, l'hôte relit Y dans le banc OUT ; puis deux couches enchaînées en recopiant OUT → IN

Ce fichier fournit aussi les fonctions « pilote hôte » (load_*, read_*, run_layer) réutilisées par le benchmark
et l'animation.

Lancement : python test/activation_buffer/test.py
"""

import sys
import pathlib
import random

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import Bench, pack, unpack, load_test_module, run_tests
import layout

import numpy as np
from src.activation_buffer.activation_buffer import ActivationBuffer

OD = load_test_module('output_drain')
PP = load_test_module('post_processing')
N, DW, BAW, IW, AW = 4, 8, 10, 6, 32
IN, OUT = 0, 1


def bench():
    b = Bench(
        inputs={'busy': 1, 'host_addr': BAW + 1, 'host_wdata': N * DW, 'host_we': 1, 'host_re': 1,
                'core_rd_addr': BAW, 'core_rd_re': 1, 'core_wr_addr': BAW, 'core_wr_wdata': N * DW,
                'core_wr_we': 1, 'core_wr_wmask': N},
        outputs={'host_rdata': N * DW, 'core_rd_rdata': N * DW},
        build=lambda s, w: ActivationBuffer(s, 'abuf', w['busy'], w['host_addr'], w['host_wdata'], w['host_we'],
                                            w['host_re'], w['host_rdata'], w['core_rd_addr'], w['core_rd_re'],
                                            w['core_rd_rdata'], w['core_wr_addr'], w['core_wr_wdata'],
                                            w['core_wr_we'], w['core_wr_wmask']))
    b.set(busy=0)
    return b


def hw(b, bank, a, word):
    b.step(busy=0, host_addr=(bank << BAW) | a, host_wdata=word, host_we=1, host_re=0)


def hr(b, bank, a):
    b.step(busy=0, host_addr=(bank << BAW) | a, host_we=0, host_re=1)
    return b.get('host_rdata')


# --- Tests unitaires ---------------------------------------------------------------------------------------

def test_hote_deux_bancs():
    b = bench()
    hw(b, IN, 12, 0x01020304)
    hw(b, OUT, 12, 0x0A0B0C0D)
    assert hr(b, IN, 12) == 0x01020304
    assert hr(b, OUT, 12) == 0x0A0B0C0D
    b.step(busy=0, host_re=0)
    assert b.get('host_rdata') == 0x0A0B0C0D, 'banc lu mémorisé, donnée maintenue'


def test_coeur_lit_in_et_ecrit_out_dans_le_meme_cycle():
    b = bench()
    hw(b, IN, 3, 0x11111111)
    hw(b, OUT, 3, 0x22222222)
    b.step(busy=1, core_rd_addr=3, core_rd_re=1, core_wr_addr=3, core_wr_wdata=0x99999999, core_wr_we=1,
           core_wr_wmask=0b1111)
    assert b.get('core_rd_rdata') == 0x11111111, 'le banc IN n\'est pas touché par l\'écriture'
    b.step(busy=1, core_rd_re=0, core_wr_we=0)
    assert hr(b, OUT, 3) == 0x99999999 and hr(b, IN, 3) == 0x11111111


def test_masque_par_octet():
    b = bench()
    hw(b, OUT, 8, 0xAABBCCDD)
    for mask, attendu in [(0b0001, 0xAABBCC11), (0b0110, 0xAA2222DD), (0b1000, 0x33BBCCDD), (0, 0xAABBCCDD)]:
        hw(b, OUT, 8, 0xAABBCCDD)
        val = {0b0001: 0x11111111, 0b0110: 0x22222222, 0b1000: 0x33333333, 0: 0x44444444}[mask]
        b.step(busy=1, core_wr_addr=8, core_wr_wdata=val, core_wr_we=1, core_wr_wmask=mask)
        b.step(busy=1, core_wr_we=0)
        assert hr(b, OUT, 8) == attendu, f'masque {mask:04b}'


def test_cote_actif_choisi_par_busy():
    b = bench()
    hw(b, IN, 1, 0x55)
    b.step(busy=1, host_addr=1, host_wdata=0x66, host_we=1, host_re=1)
    assert hr(b, IN, 1) == 0x55, 'écriture hôte ignorée pendant busy'
    b.step(busy=0, core_wr_addr=1, core_wr_wdata=0x77, core_wr_we=1, core_wr_wmask=0b1111, core_rd_re=1,
           core_rd_addr=1, host_we=0, host_re=0)
    assert hr(b, OUT, 1) == 0, 'écriture cœur ignorée hors busy'


def test_aleatoire():
    rng = random.Random(140)
    b = bench()
    mem = {IN: {}, OUT: {}}
    in_r = out_r = sel = 0
    for _ in range(3000):
        busy = rng.randint(0, 1)
        v = dict(busy=busy, host_addr=rng.randrange(1 << (BAW + 1)), host_wdata=rng.randrange(1 << 32),
                 host_we=rng.randint(0, 1), host_re=rng.randint(0, 1), core_rd_addr=rng.randrange(1 << BAW),
                 core_rd_re=rng.randint(0, 1), core_wr_addr=rng.randrange(1 << BAW),
                 core_wr_wdata=rng.randrange(1 << 32), core_wr_we=rng.randint(0, 1), core_wr_wmask=rng.randrange(16))
        b.step(**v)
        if busy:
            if v['core_rd_re']:
                in_r = mem[IN].get(v['core_rd_addr'], 0)
            if v['core_wr_we']:
                old = mem[OUT].get(v['core_wr_addr'], 0)
                m = sum(0xFF << (8 * l) for l in range(4) if (v['core_wr_wmask'] >> l) & 1)
                mem[OUT][v['core_wr_addr']] = (old & ~m) | (v['core_wr_wdata'] & m)
        else:
            bank, a = v['host_addr'] >> BAW, v['host_addr'] & ((1 << BAW) - 1)
            if v['host_re']:
                if bank:
                    out_r = mem[OUT].get(a, 0)
                else:
                    in_r = mem[IN].get(a, 0)
                sel = bank
            if v['host_we']:
                mem[bank][a] = v['host_wdata']
        assert b.get('core_rd_rdata') == in_r
        assert b.get('host_rdata') == (out_r if sel else in_r)


def test_disposition_x_relue_aux_adresses_du_planificateur():
    rng = np.random.default_rng(141)
    for variant in ('OS', 'WS'):
        for M, K, Nd in [(4, 4, 4), (6, 9, 10), (5, 3, 7)]:
            W = rng.integers(-128, 128, (M, K))
            X = rng.integers(-128, 128, (K, Nd))
            b = bench()
            for a in range(1 << BAW):
                hw(b, IN, a, int(rng.integers(0, 1 << 32)))
            for a, lanes in layout.x_image(X, variant).items():
                hw(b, IN, a, pack(lanes, DW))
            for e in (OD.plan_os if variant == 'OS' else OD.plan_ws)(W, X, N, rng):
                if e['x_addr'] is not None:
                    b.step(busy=1, core_rd_addr=e['x_addr'], core_rd_re=1)
                    got = unpack(b.get('core_rd_rdata'), N, DW)
                    for l in range(N):
                        if (e['x_mask'] >> l) & 1:
                            assert got[l] == e['x'][l], (variant, M, K, Nd, e['x_addr'], l)


# --- Pilote hôte du chemin de données avec mémoires ------------------------------------------------------------

def datapath(variant):
    return OD.core(variant, N, mem=True)


def hote(b, **v):
    """Un cycle d'accès hôte ; busy n'est imposé que s'il est une entrée du banc (sans controller)."""
    if 'busy' in b.inputs:
        v['busy'] = 0
    b.step(**v)


def sig(b, name):
    """Valeur d'un signal du banc, entrée, sortie ou fil interne."""
    return (b.w[name] if name in b.w else b.dut['w'][name]).get()


def load_wb(b, image):
    for a, lanes in image.items():
        hote(b, wb_host_addr=a, wb_host_wdata=pack(lanes, DW), wb_host_we=1)
    hote(b, wb_host_we=0)


def load_ab(b, bank, image):
    for a, lanes in image.items():
        hote(b, ab_host_addr=(bank << BAW) | a, ab_host_wdata=pack(lanes, DW), ab_host_we=1)
    hote(b, ab_host_we=0)


def read_ab(b, bank, addrs):
    words = {}
    for a in addrs:
        hote(b, ab_host_addr=(bank << BAW) | a, ab_host_we=0, ab_host_re=1)
        words[a] = unpack(b.w['ab_host_rdata'].get(), N, DW)
    hote(b, ab_host_re=0)
    return words


def load_bias(b, bias):
    for i, v in enumerate(bias):
        hote(b, bias_host_addr=i, bias_host_wdata=int(v) & 0xFFFFFFFF, bias_host_we=1)
    hote(b, bias_host_we=0)


def salir(b, rng):
    """Remplit toute la mémoire de valeurs non nulles : les données hors limites ne doivent jamais compter."""
    load_wb(b, {a: list(rng.integers(-128, 128, N)) for a in range(1 << BAW)})
    load_ab(b, IN, {a: list(rng.integers(-128, 128, N)) for a in range(1 << BAW)})


def run_layer(b, variant, W, X, bias, M, s, relu, load_x=True, on_cycle=None, log=None):
    """
    Une couche Y = requant(ReLU(W·X + b)), pilotée comme le fera l'hôte :
    chargement de W (et de X si load_x), des biais et de la configuration, GEMM (busy = 1), relecture de Y.
    Renvoie (Y, nombre de cycles par phase).

    Si `log` est une liste, on y ajoute un enregistrement par cycle décrivant les accès mémoire :
      phase, wb_w (adresse écrite par l'hôte dans le weight_buffer), ab_w ((banc, adresse) écrit par l'hôte),
      ab_r ((banc, adresse) lu par l'hôte), w_rd / x_rd (adresses lues par le cœur), y_wr ((adresse, masque) écrit
      par le post_processing).
    """
    Mr, Nd = W.shape[0], X.shape[1]
    phase = ['chargement']
    if log is not None:
        orig_step = b.step

        def step(**v):
            orig_step(**v)
            g = lambda n: b.w[n].get()
            log.append(dict(phase=phase[0], wb_w=g('wb_host_addr') if g('wb_host_we') else None,
                            ab_w=(g('ab_host_addr') >> BAW, g('ab_host_addr') & ((1 << BAW) - 1))
                            if g('ab_host_we') else None,
                            ab_r=(g('ab_host_addr') >> BAW, g('ab_host_addr') & ((1 << BAW) - 1))
                            if g('ab_host_re') else None,
                            w_rd=None, x_rd=None, y_wr=None))
        b.step = step

        def hook(c, bb):
            g = lambda n: sig(bb, n)
            log.append(dict(phase='calcul', wb_w=None, ab_w=None, ab_r=None,
                            w_rd=g('w_addr') if g('w_re') else None, x_rd=g('x_addr') if g('x_re') else None,
                            y_wr=(g('y_addr'), g('y_wmask')) if g('y_we') else None))
            if on_cycle:
                on_cycle(c, bb)
    else:
        hook = on_cycle
    try:
        c0 = b.cycle
        load_wb(b, layout.w_image(W))
        if load_x:
            load_ab(b, IN, layout.x_image(X, variant))
        load_bias(b, list(bias)[:Mr])                       # seuls les biais des lignes utiles
        c1 = b.cycle
        if b.ctrl:
            b.set(req_mult=M, req_shift=s, relu_en=relu)
            run_go(b, Mr, W.shape[1], Nd, on_cycle=hook)
        else:
            b.set(busy=1, req_mult=M, req_shift=s, relu_en=relu)
            plan = (OD.plan_os if variant == 'OS' else OD.plan_ws)(W, X, N, np.random.default_rng(0))
            OD.run_core(b, plan, on_cycle=hook)
        c2 = b.cycle
        phase[0] = 'relecture'
        words = read_ab(b, OUT, layout.y_addresses(Mr, Nd, variant))
    finally:
        if log is not None:
            b.step = orig_step
    Y = np.array(layout.y_from_words(words, Mr, Nd, variant))
    return Y, dict(chargement=c1 - c0, calcul=c2 - c1, relecture=b.cycle - c2)


def run_go(b, M, K, Nd, on_cycle=None, limite=100000):
    """
    Avec le controller : impulsion go avec les dimensions, puis cycles jusqu'au cycle où done vaut 1 (compris).
    Renvoie le nombre de cycles écoulés, cycle du go compris.
    """
    b.set(go=1, dim_m=M, dim_k=K, dim_n=Nd)
    for c in range(limite):
        b.sim.propagateAll()
        if on_cycle:
            on_cycle(c, b)
        fin = b.get('done')                                # done pendant ce cycle (sortie de Moore)
        b.sim.clk(1)
        b.cycle += 1
        b.set(go=0)
        if fin:
            return c + 1
    raise AssertionError('done jamais levé')


def copy_out_to_in(b, addrs):
    """Recopie par l'hôte des mots de Y (banc OUT) vers le banc IN, pour la couche suivante."""
    words = read_ab(b, OUT, addrs)
    load_ab(b, IN, words)
    return 3 * len(addrs) + 2


# --- Tests de bout en bout avec mémoires ---------------------------------------------------------------------

def test_chemin_de_donnees_complet_avec_memoires():
    rng = np.random.default_rng(142)
    for variant in ('OS', 'WS'):
        b = datapath(variant)
        salir(b, rng)
        for (Mr, K, Nd), relu in [((4, 4, 4), 1), ((9, 11, 6), 0), ((16, 16, 16), 1)]:
            W = rng.integers(-128, 128, (Mr, K))
            X = rng.integers(-128, 128, (K, Nd))
            bias = list(rng.integers(-20000, 20000, Mr))
            M, s = int(rng.integers(1, 1 << 16)), int(rng.integers(14, 24))
            Y, _ = run_layer(b, variant, W, X, bias, M, s, relu)
            assert (Y == PP.reference(W, X, bias + [0] * 64, M, s, relu)).all(), f'{variant} {Mr}×{K}×{Nd}'


def test_deux_couches_enchainees_par_la_memoire():
    """Y1 est recopié tel quel du banc OUT vers le banc IN : il a déjà le format X de la variante (D6)."""
    rng = np.random.default_rng(143)
    X = rng.integers(-128, 128, (12, 6))
    W1, b1 = rng.integers(-128, 128, (10, 12)), list(rng.integers(-3000, 3000, 10))
    W2, b2 = rng.integers(-128, 128, (5, 10)), list(rng.integers(-3000, 3000, 5))
    Y1_ref = PP.reference(W1, X, b1 + [0] * 64, 300, 13, 1)
    Y2_ref = PP.reference(W2, Y1_ref, b2 + [0] * 64, 200, 12, 0)
    for variant in ('OS', 'WS'):
        b = datapath(variant)
        salir(b, rng)
        Y1, _ = run_layer(b, variant, W1, X, b1, 300, 13, 1)
        assert (Y1 == Y1_ref).all(), variant
        copy_out_to_in(b, layout.y_addresses(10, 6, variant))
        Y2, _ = run_layer(b, variant, W2, Y1, b2, 200, 12, 0, load_x=False)
        assert (Y2 == Y2_ref).all(), variant


if __name__ == '__main__':
    sys.exit(run_tests(globals()))
