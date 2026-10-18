"""
Test fonctionnel du weight_buffer (spécification : src/weight_buffer/interface.md).

  - écriture / lecture par l'hôte, latence de lecture 1, donnée lue maintenue tant que re = 0
  - un seul port actif à la fois, choisi par busy ; contenu conservé quand busy change
  - comparaison avec un modèle de référence sur des accès aléatoires
  - disposition D6 : W rangé par l'hôte (test/layout.py) puis relu aux adresses émises par le planificateur
    du controller → chaque lecture rend exactement le mot attendu par le cœur de calcul

Lancement : python test/weight_buffer/test.py
"""

import sys
import pathlib
import random

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import Bench, pack, unpack, load_test_module, run_tests
import layout

import numpy as np
from src.weight_buffer.weight_buffer import WeightBuffer

OD = load_test_module('output_drain')
N, DW, BAW = 4, 8, 10


def bench():
    b = Bench(
        inputs={'busy': 1, 'host_addr': BAW, 'host_wdata': N * DW, 'host_we': 1, 'host_re': 1,
                'core_addr': BAW, 'core_re': 1},
        outputs={'host_rdata': N * DW, 'core_rdata': N * DW},
        build=lambda s, w: WeightBuffer(s, 'wbuf', w['busy'], w['host_addr'], w['host_wdata'], w['host_we'],
                                        w['host_re'], w['host_rdata'], w['core_addr'], w['core_re'], w['core_rdata']))
    b.set(busy=0)
    return b


def write(b, a, word):
    b.step(busy=0, host_addr=a, host_wdata=word, host_we=1, host_re=0)


def host_read(b, a):
    b.step(busy=0, host_addr=a, host_we=0, host_re=1)
    return b.get('host_rdata')


def core_read(b, a):
    b.step(busy=1, core_addr=a, core_re=1)
    return b.get('core_rdata')


# --- Tests ---------------------------------------------------------------------------------------------

def test_ecriture_lecture_hote():
    b = bench()
    write(b, 5, 0x11223344)
    write(b, 1023, 0xDEADBEEF)
    assert host_read(b, 5) == 0x11223344                       # visible au cycle suivant la demande
    assert host_read(b, 1023) == 0xDEADBEEF
    assert host_read(b, 0) == 0


def test_latence_1_et_maintien():
    b = bench()
    write(b, 7, 0xCAFE)
    b.step(busy=0, host_addr=7, host_we=0, host_re=1)
    assert b.get('host_rdata') == 0xCAFE
    for a in (1, 2, 3):                                        # re = 0 : la donnée lue est maintenue
        b.step(busy=0, host_addr=a, host_re=0)
        assert b.get('host_rdata') == 0xCAFE


def test_lecture_et_ecriture_simultanees():
    b = bench()
    write(b, 9, 111)
    b.step(busy=0, host_addr=9, host_wdata=222, host_we=1, host_re=1)
    assert b.get('host_rdata') == 111, 'la lecture rend l\'ancienne valeur'
    assert host_read(b, 9) == 222


def test_port_actif_choisi_par_busy():
    b = bench()
    write(b, 3, 0xAAAA)
    # busy = 1 : l'hôte est ignoré (écriture et lecture), le cœur lit
    b.step(busy=1, host_addr=3, host_wdata=0xBBBB, host_we=1, host_re=1, core_addr=3, core_re=0)
    assert b.get('host_rdata') == 0, 'lecture hôte ignorée pendant busy'
    assert core_read(b, 3) == 0xAAAA, 'écriture hôte ignorée pendant busy'
    # busy = 0 : le cœur est ignoré
    b.step(busy=0, host_we=0, host_re=0, core_addr=4, core_re=1)
    assert b.get('core_rdata') == 0xAAAA
    # le contenu survit aux changements de busy
    assert host_read(b, 3) == 0xAAAA


def test_aleatoire():
    rng = random.Random(130)
    b, ref, rdata = bench(), {}, 0
    for _ in range(3000):
        busy, a = rng.randint(0, 1), rng.randrange(1 << BAW)
        we, re, word = rng.randint(0, 1), rng.randint(0, 1), rng.randrange(1 << 32)
        b.step(busy=busy, host_addr=a, host_wdata=word, host_we=we, host_re=re, core_addr=a, core_re=re)
        if busy and re:
            rdata = ref.get(a, 0)
        elif not busy:
            if re:
                rdata = ref.get(a, 0)
            if we:
                ref[a] = word
        assert b.get('host_rdata') == b.get('core_rdata') == rdata


def test_disposition_w_relue_aux_adresses_du_planificateur():
    """Le controller (ici son planificateur de référence) lit exactement les mots que le cœur attend."""
    rng = np.random.default_rng(131)
    for variant in ('OS', 'WS'):
        for M, K, Nd in [(4, 4, 4), (10, 7, 3), (3, 13, 9)]:
            W = rng.integers(-128, 128, (M, K))
            X = rng.integers(-128, 128, (K, Nd))
            b = bench()
            for a in range(1 << BAW):                          # mémoire « sale »
                write(b, a, int(rng.integers(0, 1 << 32)))
            for a, lanes in layout.w_image(W).items():
                write(b, a, pack(lanes, DW))
            plan = (OD.plan_os if variant == 'OS' else OD.plan_ws)(W, X, N, rng)
            lus = 0
            for e in plan:
                if e['w_addr'] is not None:
                    got = unpack(core_read(b, e['w_addr']), N, DW)
                    for l in range(N):
                        if (e['w_mask'] >> l) & 1:
                            assert got[l] == e['w'][l], (variant, M, K, Nd, e['w_addr'], l)
                    lus += 1
            assert lus > 0


if __name__ == '__main__':
    sys.exit(run_tests(globals()))
