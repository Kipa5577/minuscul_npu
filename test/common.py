"""
Outils communs aux bancs de test (test.py, benchmark.py, animation.py).

Utilisation depuis test/<bloc>/<fichier>.py :
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from common import ...
"""

import sys
import pathlib
import inspect
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))          # permet `from src.<bloc>.<bloc> import ...`

import py4hw


# --- Complément à deux -------------------------------------------------------

def to_c2(v: int, w: int) -> int:
    """Entier signé → motif de w bits."""
    return v & ((1 << w) - 1)


def from_c2(v: int, w: int) -> int:
    """Motif de w bits → entier signé."""
    v &= (1 << w) - 1
    return v - (1 << w) if v >> (w - 1) else v


def wrap(v: int, w: int) -> int:
    """Réduit un entier signé modulo 2^w (comportement d'un additionneur de w bits)."""
    return from_c2(to_c2(v, w), w)


def pack(values, w: int) -> int:
    """Liste de valeurs signées → bus empaqueté (voie 0 = bits de poids faible)."""
    bus = 0
    for l, v in enumerate(values):
        bus |= to_c2(int(v), w) << (w * l)
    return bus


def unpack(bus: int, n: int, w: int) -> list:
    """Bus empaqueté → liste de n valeurs signées."""
    return [from_c2(bus >> (w * l), w) for l in range(n)]


def load_test_module(bloc: str, fichier: str = 'test.py'):
    """
    Charge test/<bloc>/<fichier> comme module (pour réutiliser ses bancs et modèles de référence).
    Un `import test` classique tomberait sur le module `test` de la bibliothèque standard.
    """
    import importlib.util
    path = ROOT / 'test' / bloc / fichier
    spec = importlib.util.spec_from_file_location(f'{bloc}_{path.stem}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def setup_matplotlib_fonts():
    """
    Matplotlib relit le fichier de police à chaque texte dessiné. Selon le système de fichiers où il
    est installé, cet accès peut devenir le facteur limitant des animations. Quand une copie identique
    d'une police fournie par matplotlib existe parmi les polices du système, on utilise celle-ci.
    Sans effet si aucune copie système n'est trouvée.
    """
    import os
    import dataclasses
    from matplotlib import font_manager
    bundled = os.path.join(os.path.dirname(font_manager.__file__), 'mpl-data')
    system = {}
    for f in font_manager.findSystemFonts():
        system.setdefault(os.path.basename(f), f)
    fm = font_manager.fontManager
    fm.ttflist = [dataclasses.replace(e, fname=system[os.path.basename(e.fname)])
                  if e.fname.startswith(bundled) and os.path.basename(e.fname) in system else e
                  for e in fm.ttflist]
    fm._findfont_cached.cache_clear()
    font_manager._get_font.cache_clear()


# --- Banc de test générique ----------------------------------------------------

class Bench:
    """
    Système py4hw minimal autour d'un bloc.

    - `inputs`  : {nom: largeur} des fils pilotés par le banc
    - `outputs` : {nom: largeur} des fils lus par le banc
    - `build(sys, wires)` instancie le bloc sous test

    Un appel à `step(**valeurs)` présente les entrées pendant un cycle puis
    avance d'un front d'horloge ; les sorties lues ensuite sont celles d'après le front.
    """

    def __init__(self, inputs: dict, outputs: dict, build, signed=()):
        self.sys = py4hw.HWSystem()
        self.widths = {**inputs, **outputs}
        self.signed = set(signed)
        self.w = {n: self.sys.wire(n, wd) for n, wd in self.widths.items()}
        self.inputs = set(inputs)
        self.dut = build(self.sys, self.w)
        self.sim = self.sys.getSimulator()
        self.cycle = 0

    def set(self, **values):
        for n, v in values.items():
            assert n in self.inputs, f'{n} n\'est pas une entrée'
            self.w[n].put(to_c2(v, self.widths[n]))

    def get(self, name: str) -> int:
        v = self.w[name].get()
        return from_c2(v, self.widths[name]) if name in self.signed else v

    def step(self, **values):
        self.set(**values)
        self.sim.clk(1)
        self.cycle += 1

    def observe(self, **values) -> dict:
        """
        Présente les entrées, propage la logique combinatoire, lit toutes les sorties PENDANT le cycle,
        puis avance d'un front. Utile pour les blocs dont certaines sorties sont combinatoires.
        """
        self.set(**values)
        self.sim.propagateAll()
        snap = self.snapshot()
        self.sim.clk(1)
        self.cycle += 1
        return snap

    def snapshot(self) -> dict:
        return {n: self.get(n) for n in self.widths}


# --- Mini exécuteur de tests ---------------------------------------------------

def run_tests(module_globals: dict) -> int:
    """Exécute toutes les fonctions test_* du module, affiche le bilan, renvoie le code de sortie."""
    tests = [(n, f) for n, f in module_globals.items()
             if n.startswith('test_') and inspect.isfunction(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f'  OK    {name}')
        except Exception:
            failed += 1
            print(f'  ÉCHEC {name}')
            traceback.print_exc(limit=3)
    print(f'\n{len(tests) - failed}/{len(tests)} tests réussis')
    return 1 if failed else 0
