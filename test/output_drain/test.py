"""
Test fonctionnel de l'output_drain (spécification : src/output_drain/interface.md).

  - OS : une ligne d'accumulateurs par cycle, latence 1, tag transmis
  - WS : redressement de l'escalier, retard de 2N cycles du sideband, accumulation sur les tuiles de K
  - comparaison cycle par cycle avec des modèles de référence
  - de bout en bout : data_feeder → pe_array → output_drain sur des GEMM multi-tuiles, pilotés par un
    planificateur Python qui émet cycle par cycle ce que la FSM du controller doit émettre
    (src/controller/interface.md). Y est reconstruit uniquement à partir des (tag, res) sortis.

Lancement : python test/output_drain/test.py
"""

import sys
import pathlib
import random

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import Bench, pack, unpack, run_tests

import numpy as np
from src.data_feeder.data_feeder import DataFeeder_OS, DataFeeder_WS
from src.pe_array.pe_array import PE_Array_OS, PE_Array_WS
from src.output_drain.output_drain import OutputDrain_OS, OutputDrain_WS

DW, AW = 8, 32
DIM_MAX = 64
IW = 6                       # log2(DIM_MAX)
TAG_W = 20                   # {y_addr[19:10], lane_mask[9:6], row[5:0]} pour N = 4


# --- Tag (architecture.md §5.4) -------------------------------------------------------------------

def tag_pack(y_addr, lane_mask, row, N=4):
    return (y_addr << (N + IW)) | (lane_mask << IW) | row


def tag_unpack(tag, N=4):
    return tag >> (N + IW), (tag >> IW) & ((1 << N) - 1), tag & ((1 << IW) - 1)


# --- Bancs unitaires ---------------------------------------------------------------------------------

def bench_os(N=4):
    lw = max(1, (N - 1).bit_length())
    b = Bench(
        inputs={'rst_n': 1, 'acc': N * N * AW, 'drain_en': 1, 'drain_row': lw, 'tag_in': TAG_W},
        outputs={'res': N * AW, 'res_valid': 1, 'tag_out': TAG_W},
        build=lambda s, w: OutputDrain_OS(s, 'drain', w['rst_n'], w['acc'], w['drain_en'], w['drain_row'],
                                          w['tag_in'], w['res'], w['res_valid'], w['tag_out'], N=N))
    b.N = N
    b.step(rst_n=0)
    b.set(rst_n=1)
    return b


def bench_ws(N=4):
    b = Bench(
        inputs={'rst_n': 1, 'psum_south': N * AW, 's_valid': 1, 's_j': IW, 's_first': 1, 's_last': 1,
                'tag_in': TAG_W},
        outputs={'res': N * AW, 'res_valid': 1, 'tag_out': TAG_W},
        build=lambda s, w: OutputDrain_WS(s, 'drain', w['rst_n'], w['psum_south'], w['s_valid'], w['s_j'],
                                          w['s_first'], w['s_last'], w['tag_in'], w['res'], w['res_valid'],
                                          w['tag_out'], N=N))
    b.N = N
    b.step(rst_n=0)
    b.set(rst_n=1)
    return b


def res_lanes(b):
    return unpack(b.w['res'].get(), b.N, AW)


# --- Modèles de référence ----------------------------------------------------------------------------

class RefDrainOS:
    def __init__(self, N):
        self.N, self.res, self.valid, self.tag = N, [0] * N, 0, 0

    def step(self, acc, drain_en, drain_row, tag_in):
        self.valid = drain_en
        if drain_en:
            self.res, self.tag = list(acc[drain_row]), tag_in


class RefDrainWS:
    def __init__(self, N):
        self.N = N
        self.deskew = [[0] * (N - 1 - i) for i in range(N)]
        self.sb = [None] * (2 * N)
        self.mem = {}
        self.res, self.valid, self.tag = [0] * N, 0, 0

    def step(self, south, s_valid, s_j, s_first, s_last, tag_in):
        N = self.N
        aligned = [self.deskew[i][-1] if N - 1 - i else south[i] for i in range(N)]
        sb, self.valid = self.sb[-1], 0
        if sb and sb[0]:
            _, j, first, last, tag = sb
            s = aligned if first else [wrap32(a + p) for a, p in zip(self.mem.get(j, [0] * N), aligned)]
            if last:
                self.res, self.valid, self.tag = s, 1, tag
            else:
                self.mem[j] = s
        for i in range(N - 1):
            self.deskew[i] = [south[i]] + self.deskew[i][:-1]
        self.sb = [(s_valid, s_j, s_first, s_last, tag_in)] + self.sb[:-1]


def wrap32(v):
    v &= (1 << 32) - 1
    return v - (1 << 32) if v >> 31 else v


# --- Tests OS --------------------------------------------------------------------------------------------

def test_os_reset():
    b = bench_os()
    b.step(acc=pack(range(1, 17), AW), drain_en=1, drain_row=2, tag_in=123)
    assert b.get('res_valid') == 1
    b.step(rst_n=0)
    assert (b.get('res'), b.get('res_valid'), b.get('tag_out')) == (0, 0, 0)


def test_os_vidage_ligne_par_ligne():
    N = 4
    A = np.arange(-8, 8).reshape(N, N) * 1000
    b = bench_os(N)
    b.set(acc=pack(A.flatten(), AW))
    for r in range(N):
        b.step(drain_en=1, drain_row=r, tag_in=100 + r)              # latence 1 : visible après le front
        assert b.get('res_valid') == 1 and res_lanes(b) == list(A[r]) and b.get('tag_out') == 100 + r
    b.step(drain_en=0)
    assert b.get('res_valid') == 0


def test_os_aleatoire_cycle_par_cycle():
    random.seed(50)
    for N in (2, 3, 4, 8):
        b, ref = bench_os(N), RefDrainOS(N)
        for _ in range(400):
            acc = [[random.randint(-2**31, 2**31 - 1) for _ in range(N)] for _ in range(N)]
            v = dict(drain_en=random.randint(0, 1), drain_row=random.randrange(N), tag_in=random.randrange(1 << TAG_W))
            b.step(acc=pack(sum(acc, []), AW), **v)
            ref.step(acc, **v)
            assert (res_lanes(b), b.get('res_valid'), b.get('tag_out')) == (ref.res, ref.valid, ref.tag), N


# --- Tests WS --------------------------------------------------------------------------------------------

def test_ws_redressement_et_latence():
    """Escalier en entrée (voie i en retard de i cycles) → voies alignées, res_valid à e + 2N + 1."""
    for N in (1, 2, 4, 8):
        b = bench_ws(N)
        vals = [(i + 1) * 111 for i in range(N)]
        e = 0                                       # cycle d'émission de la colonne
        hist = []
        for c in range(4 * N + 4):
            # la voie i porte la somme partielle pendant le cycle e + 1 + i + N (contrat du pe_array)
            south = [vals[i] if c == e + 1 + i + N else 0 for i in range(N)]
            sb = dict(s_valid=1, s_j=5, s_first=1, s_last=1, tag_in=777) if c == e else \
                dict(s_valid=0, s_j=0, s_first=0, s_last=0, tag_in=0)
            b.step(psum_south=pack(south, AW), **sb)
            hist.append((b.get('res_valid'), res_lanes(b), b.get('tag_out')))
        valid = [c for c, h in enumerate(hist) if h[0]]
        assert valid == [e + 2 * N], f'N={N} : res_valid après les pas {valid}'   # visible pendant e + 2N + 1
        assert hist[e + 2 * N][1] == vals and hist[e + 2 * N][2] == 777, N


def test_ws_accumulation_sur_tuiles_de_k():
    N = 4
    b = bench_ws(N)
    # trois passages sur la même colonne j = 7 : first, milieu, last
    parts = [[1, 2, 3, 4], [10, 20, 30, 40], [-100, 200, -300, 400]]
    out = []
    for t, part in enumerate(parts):
        e = t * 20
        for c in range(e, e + 20):
            south = [part[i] if c == e + 1 + i + N else 0 for i in range(N)]
            sb = dict(s_valid=1, s_j=7, s_first=int(t == 0), s_last=int(t == 2), tag_in=42) if c == e else \
                dict(s_valid=0, s_j=0, s_first=0, s_last=0, tag_in=0)
            b.step(psum_south=pack(south, AW), **sb)
            if b.get('res_valid'):
                out.append(res_lanes(b))
    assert out == [list(np.sum(parts, axis=0))]


def test_ws_first_et_last_ensemble():
    """K ≤ N : une seule tuile de K, le résultat sort directement sans passer par la mémoire."""
    N = 4
    b = bench_ws(N)
    b.dut.accmem[3] = [999] * N                     # contenu ancien : doit être ignoré
    for c in range(3 * N):
        south = [5 * (i + 1) if c == 1 + i + N else 0 for i in range(N)]
        sb = dict(s_valid=1, s_j=3, s_first=1, s_last=1, tag_in=1) if c == 0 else {}
        b.step(psum_south=pack(south, AW), **({'s_valid': 0} | sb))
        if b.get('res_valid'):
            assert res_lanes(b) == [5, 10, 15, 20]
            return
    raise AssertionError('aucun résultat')


def test_ws_aleatoire_cycle_par_cycle():
    random.seed(51)
    for N in (2, 4):
        b, ref = bench_ws(N), RefDrainWS(N)
        for _ in range(800):
            south = [random.randint(-2**20, 2**20) for _ in range(N)]
            v = dict(s_valid=int(random.random() < 0.7), s_j=random.randrange(8), s_first=int(random.random() < 0.3),
                     s_last=int(random.random() < 0.3), tag_in=random.randrange(1 << TAG_W))
            b.step(psum_south=pack(south, AW), **v)
            ref.step(south, **v)
            assert (res_lanes(b), b.get('res_valid'), b.get('tag_out')) == (ref.res, ref.valid, ref.tag), N


# --- Planificateur : ce que la FSM du controller émet, cycle par cycle -----------------------------------

def tiles(D, N):
    return -(-D // N)


def lane_mask(t, D, N):
    return sum(1 << l for l in range(N) if N * t + l < D)


IDLE = dict(w=None, x=None, w_addr=None, x_addr=None, w_mask=0, x_mask=0, ctl=0, drain_en=0, drain_row=0,
            s_valid=0, s_j=0, s_first=0, s_last=0, tag=0, phase='repos')


def dirty(rng, shape, valid_block):
    """Matrice mémoire : le bloc valide, entouré de valeurs non nulles (mémoire « sale »)."""
    m = rng.integers(-128, 128, shape)
    m[:valid_block.shape[0], :valid_block.shape[1]] = valid_block
    return m


def plan_os(W, X, N, rng):
    """Boucles i_t → j_t → k : CLEAR, STREAM (K), FLUSH (2N − 1), DRAIN (N), puis TAIL (4)."""
    M, K = W.shape
    Nd = X.shape[1]
    Wd = dirty(rng, (N * tiles(M, N), K), W)
    Xd = dirty(rng, (K, N * tiles(Nd, N)), X)
    em = []
    for it in range(tiles(M, N)):
        for jt in range(tiles(Nd, N)):
            em.append(IDLE | dict(ctl=1, phase='clear'))
            for k in range(K):
                em.append(IDLE | dict(w=list(Wd[N * it:N * it + N, k]), x=list(Xd[k, N * jt:N * jt + N]),
                                      w_addr=(it << IW) | k, x_addr=(jt << IW) | k,
                                      w_mask=lane_mask(it, M, N), x_mask=lane_mask(jt, Nd, N), phase='flux'))
            em += [IDLE | dict(phase='vidange')] * (2 * N - 1)
            for r in range(N):
                i = N * it + r
                em.append(IDLE | dict(drain_en=1, drain_row=r, phase='lecture',
                                      tag=tag_pack((jt << IW) | i, lane_mask(jt, Nd, N) if i < M else 0, i, N)))
    em += [IDLE | dict(phase='fin')] * 4
    return em


def plan_ws(W, X, N, rng):
    """Boucles i_t → k_t → j : PRELOAD (N), STREAM (N_dim), FLUSH (2N − 3), puis TAIL (2N + 4)."""
    M, K = W.shape
    Nd = X.shape[1]
    Wd = dirty(rng, (N * tiles(M, N), N * tiles(K, N)), W)
    Xd = dirty(rng, (N * tiles(K, N), Nd), X)
    em = []
    for it in range(tiles(M, N)):
        for kt in range(tiles(K, N)):
            for n in range(N):
                k = N * kt + N - 1 - n
                em.append(IDLE | dict(w=list(Wd[N * it:N * it + N, k]), w_addr=(it << IW) | k, ctl=1, phase='préchargement',
                                      w_mask=lane_mask(it, M, N) if k < K else 0))
            for j in range(Nd):
                em.append(IDLE | dict(x=list(Xd[N * kt:N * kt + N, j]), x_addr=(kt << IW) | j,
                                      x_mask=lane_mask(kt, K, N), phase='flux',
                                      s_valid=1, s_j=j, s_first=int(kt == 0), s_last=int(kt == tiles(K, N) - 1),
                                      tag=tag_pack((it << IW) | j, lane_mask(it, M, N), N * it, N)))
            em += [IDLE | dict(phase='vidange')] * (2 * N - 3)
    em += [IDLE | dict(phase='fin')] * (2 * N + 4)
    return em


# --- Cœur de calcul : data_feeder → pe_array → output_drain ------------------------------------------------

def core(variant, N=4, post=False, mem=False, ctrl=False):
    """
    Assemblage incrémental du NPU autour du cœur de calcul (data_feeder → pe_array → output_drain) :
      - post=True : le post_processing est branché à la suite ; sa configuration et le port hôte des biais
        deviennent des entrées du banc, ses sorties y_* des sorties ;
      - mem=True (implique post) : le weight_buffer et l'activation_buffer sont branchés ; le banc émet des adresses
        (w_addr/w_re, x_addr/x_re) au lieu de fournir les mots, et les ports hôte des buffers sont exposés ;
      - ctrl=True (implique mem) : le controller pilote tout ; le banc ne fournit plus que go et les dimensions,
        et busy, done, cycles deviennent des sorties. Il ne manque alors que host_interface.
    Les signaux qui ne sont ni des entrées ni des sorties du banc sont des fils internes.
    """
    mem = mem or ctrl
    post = post or mem
    lw = max(1, (N - 1).bit_length())
    BAW = TAG_W - N - IW
    width = dict(rst_n=1, w_mask=N, x_mask=N, ctl_in=1, tag_in=TAG_W, w_addr=BAW, w_re=1, x_addr=BAW, x_re=1,
                 w_rdata=N * DW, x_rdata=N * DW, drain_en=1, drain_row=lw, s_valid=1, s_j=IW, s_first=1, s_last=1,
                 res=N * AW, res_valid=1, tag_out=TAG_W, busy=1, relu_en=1, req_mult=16, req_shift=5,
                 bias_host_addr=IW, bias_host_wdata=AW, bias_host_we=1, bias_host_re=1, bias_host_rdata=AW,
                 y_addr=BAW, y_wdata=N * DW, y_we=1, y_wmask=N,
                 wb_host_addr=BAW, wb_host_wdata=N * DW, wb_host_we=1, wb_host_re=1, wb_host_rdata=N * DW,
                 ab_host_addr=BAW + 1, ab_host_wdata=N * DW, ab_host_we=1, ab_host_re=1, ab_host_rdata=N * DW,
                 go=1, dim_m=IW + 1, dim_k=IW + 1, dim_n=IW + 1, done=1, cycles=32)
    commandes = ['w_mask', 'x_mask', 'ctl_in', 'tag_in'] + \
        (['drain_en', 'drain_row'] if variant == 'OS' else ['s_valid', 's_j', 's_first', 's_last'])
    ins = ['rst_n'] + ([] if ctrl else commandes)
    ins += (['w_addr', 'w_re', 'x_addr', 'x_re'] if mem and not ctrl else []) + ([] if mem else ['w_rdata', 'x_rdata'])
    outs = ['res', 'res_valid', 'tag_out']
    if post:
        ins += ([] if ctrl else ['busy']) + ['relu_en', 'req_mult', 'req_shift', 'bias_host_addr', 'bias_host_wdata',
                                             'bias_host_we', 'bias_host_re']
        outs += ['bias_host_rdata', 'y_addr', 'y_wdata', 'y_we', 'y_wmask']
    if mem:
        ins += ['wb_host_addr', 'wb_host_wdata', 'wb_host_we', 'wb_host_re',
                'ab_host_addr', 'ab_host_wdata', 'ab_host_we', 'ab_host_re']
        outs += ['wb_host_rdata', 'ab_host_rdata']
    if ctrl:
        ins += ['go', 'dim_m', 'dim_k', 'dim_n']
        outs += ['busy', 'done', 'cycles']

    def build(s, w):
        w = dict(w)
        for name, wd in width.items():                     # fils internes
            if name not in w:
                w[name] = s.wire(name, wd)
        wa, xa, ca = s.wire('w_arr', N * DW), s.wire('x_arr', N * DW), s.wire('ctl_arr')
        if variant == 'OS':
            acc = s.wire('acc', N * N * AW)
            f = DataFeeder_OS(s, 'feeder', w['rst_n'], w['w_rdata'], w['x_rdata'], w['w_mask'], w['x_mask'],
                              w['ctl_in'], wa, xa, ca)
            a = PE_Array_OS(s, 'arr', w['rst_n'], wa, xa, ca, acc, N=N)
            d = OutputDrain_OS(s, 'drain', w['rst_n'], acc, w['drain_en'], w['drain_row'], w['tag_in'],
                               w['res'], w['res_valid'], w['tag_out'], N=N)
        else:
            south = s.wire('psum_south', N * AW)
            f = DataFeeder_WS(s, 'feeder', w['rst_n'], w['w_rdata'], w['x_rdata'], w['w_mask'], w['x_mask'],
                              w['ctl_in'], wa, xa, ca)
            a = PE_Array_WS(s, 'arr', w['rst_n'], xa, wa, ca, south, N=N)
            d = OutputDrain_WS(s, 'drain', w['rst_n'], south, w['s_valid'], w['s_j'], w['s_first'], w['s_last'],
                               w['tag_in'], w['res'], w['res_valid'], w['tag_out'], N=N)
        parts = dict(feeder=f, array=a, drain=d)
        if post:
            from src.post_processing.post_processing import PostProcessing_OS, PostProcessing_WS
            cls = PostProcessing_OS if variant == 'OS' else PostProcessing_WS
            parts['post'] = cls(s, 'post', w['rst_n'], w['busy'], w['res'], w['res_valid'], w['tag_out'],
                                w['relu_en'], w['req_mult'], w['req_shift'], w['bias_host_addr'],
                                w['bias_host_wdata'], w['bias_host_we'], w['bias_host_re'], w['bias_host_rdata'],
                                w['y_addr'], w['y_wdata'], w['y_we'], w['y_wmask'])
        if mem:
            from src.weight_buffer.weight_buffer import WeightBuffer
            from src.activation_buffer.activation_buffer import ActivationBuffer
            parts['wbuf'] = WeightBuffer(s, 'wbuf', w['busy'], w['wb_host_addr'], w['wb_host_wdata'], w['wb_host_we'],
                                         w['wb_host_re'], w['wb_host_rdata'], w['w_addr'], w['w_re'], w['w_rdata'])
            parts['abuf'] = ActivationBuffer(s, 'abuf', w['busy'], w['ab_host_addr'], w['ab_host_wdata'],
                                             w['ab_host_we'], w['ab_host_re'], w['ab_host_rdata'],
                                             w['x_addr'], w['x_re'], w['x_rdata'],
                                             w['y_addr'], w['y_wdata'], w['y_we'], w['y_wmask'])
        if ctrl:
            from src.controller.controller import Controller_OS, Controller_WS
            common = [w[n] for n in ('rst_n', 'go', 'dim_m', 'dim_k', 'dim_n', 'busy', 'done', 'cycles', 'w_addr',
                                     'w_re', 'x_addr', 'x_re', 'w_mask', 'x_mask', 'ctl_in', 'tag_in')]
            if variant == 'OS':
                parts['ctrl'] = Controller_OS(s, 'ctrl', *common, w['drain_en'], w['drain_row'])
            else:
                parts['ctrl'] = Controller_WS(s, 'ctrl', *common, w['s_valid'], w['s_j'], w['s_first'], w['s_last'])
        parts['w'] = w                                     # tous les fils, y compris internes
        return parts

    b = Bench(inputs={n: width[n] for n in ins}, outputs={n: width[n] for n in outs}, build=build)
    b.N, b.variant, b.mem, b.ctrl = N, variant, mem, ctrl
    b.step(rst_n=0)
    b.set(rst_n=1)
    return b


def run_core(b, em, on_cycle=None):
    """
    Joue le plan : au cycle c, le controller émet em[c] et la SRAM rend les mots de em[c − 1].
    Sans mémoires, le banc fournit lui-même ces mots ; avec mem=True, il émet les adresses et les buffers répondent.
    Renvoie la liste des (cycle, tag, res) sortis par l'output_drain.
    """
    N = b.N
    zero = [0] * N
    out, prev = [], IDLE
    for c, cur in enumerate(em):
        v = dict(w_mask=cur['w_mask'], x_mask=cur['x_mask'], ctl_in=cur['ctl'], tag_in=cur['tag'])
        if b.mem:
            v |= dict(w_addr=cur['w_addr'] or 0, w_re=int(cur['w_addr'] is not None),
                      x_addr=cur['x_addr'] or 0, x_re=int(cur['x_addr'] is not None))
        else:
            v |= dict(w_rdata=pack(prev['w'] or zero, DW), x_rdata=pack(prev['x'] or zero, DW))
        if b.variant == 'OS':
            v |= dict(drain_en=cur['drain_en'], drain_row=cur['drain_row'])
        else:
            v |= dict(s_valid=cur['s_valid'], s_j=cur['s_j'], s_first=cur['s_first'], s_last=cur['s_last'])
        b.set(**v)
        if on_cycle:
            b.sim.propagateAll()
            on_cycle(c, b)
        b.sim.clk(1)
        b.cycle += 1
        if b.get('res_valid'):
            out.append((c + 1, b.get('tag_out'), res_lanes(b)))
        prev = cur
    return out


def rebuild_y(variant, out, M, Nd, N=4):
    """Reconstruit Y à partir des seuls (tag, res) : chaque élément doit être écrit exactement une fois."""
    Y = np.zeros((M, Nd), dtype=np.int64)
    count = np.zeros((M, Nd), dtype=int)
    for _, tag, res in out:
        y_addr, mask, row = tag_unpack(tag, N)
        t, idx = y_addr >> IW, y_addr & ((1 << IW) - 1)
        for l in range(N):
            if (mask >> l) & 1:
                i, j = (idx, N * t + l) if variant == 'OS' else (N * t + l, idx)
                Y[i][j] = res[l]
                count[i][j] += 1
    return Y, count


def gemm_bout_en_bout(variant, M, K, Nd, seed):
    rng = np.random.default_rng(seed)
    W = rng.integers(-128, 128, (M, K))
    X = rng.integers(-128, 128, (K, Nd))
    em = (plan_os if variant == 'OS' else plan_ws)(W, X, 4, rng)
    out = run_core(core(variant), em)
    Y, count = rebuild_y(variant, out, M, Nd)
    return W @ X, Y, count, out, em


DIMS = [(4, 4, 4), (3, 5, 2), (1, 1, 1), (8, 8, 8), (10, 7, 9), (5, 13, 6)]


def test_bout_en_bout_os():
    for s, (M, K, Nd) in enumerate(DIMS):
        ref, Y, count, _, _ = gemm_bout_en_bout('OS', M, K, Nd, 60 + s)
        assert (count == 1).all(), f'OS {M}×{K}×{Nd} : éléments écrits {count.tolist()}'
        assert (Y == ref).all(), f'OS {M}×{K}×{Nd} : Y faux'


def test_bout_en_bout_ws():
    for s, (M, K, Nd) in enumerate(DIMS):
        ref, Y, count, _, _ = gemm_bout_en_bout('WS', M, K, Nd, 70 + s)
        assert (count == 1).all(), f'WS {M}×{K}×{Nd} : éléments écrits {count.tolist()}'
        assert (Y == ref).all(), f'WS {M}×{K}×{Nd} : Y faux'


def test_bout_en_bout_valeurs_extremes():
    for variant in ('OS', 'WS'):
        W = np.full((4, 16), -128)
        X = np.full((16, 4), -128)
        em = (plan_os if variant == 'OS' else plan_ws)(W, X, 4, np.random.default_rng(0))
        Y, count = rebuild_y(variant, run_core(core(variant), em), 4, 4)
        assert (count == 1).all() and (Y == 16 * 16384).all(), variant


if __name__ == '__main__':
    sys.exit(run_tests(globals()))
