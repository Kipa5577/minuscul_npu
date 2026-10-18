# minuscul_npu

Un petit NPU (*Neural Processing Unit*, accélérateur de réseaux de neurones) conçu pour **apprendre** : comprendre
comment fonctionne une vraie puce d'inférence, de la multiplication-accumulation jusqu'au bus de l'hôte, et
**mesurer ses compromis d'architecture** au lieu de les deviner.

Le NPU est prototypé et simulé en Python avec [py4hw](https://pypi.org/project/py4hw/), au cycle près. Chaque bloc a
sa spécification, ses tests, ses mesures et une animation qui le montre fonctionner cycle par cycle. Une fois
l'architecture arrêtée, le design est transcrit à la main en Verilog, d'abord pour FPGA, puis pour
[Tiny Tapeout](https://tinytapeout.com/).

> **État : v1 complète.** La puce, pilotée uniquement par son bus de 32 bits, calcule des couches de réseau de
> neurones justes dans ses deux variantes, enchaîne les couches et exécute un petit réseau quantifié qui reproduit
> le calcul en virgule flottante (corrélation 0,9999). 96 tests, 10 benchmarks, 10 animations.

---

## Sommaire

1. [Ce que calcule le NPU](#ce-que-calcule-le-npu)
2. [Comment ça marche](#comment-ça-marche)
3. [Les deux variantes : output-stationary et weight-stationary](#les-deux-variantes--output-stationary-et-weight-stationary)
4. [Architecture de la puce](#architecture-de-la-puce)
5. [Programmer la puce](#programmer-la-puce)
6. [Résultats mesurés](#résultats-mesurés)
7. [Organisation du dépôt](#organisation-du-dépôt)
8. [Installation](#installation)
9. [Tests, benchmarks et animations](#tests-benchmarks-et-animations)
10. [Méthode de travail](#méthode-de-travail)
11. [Feuille de route](#feuille-de-route)
12. [Contribuer](#contribuer)
13. [Licence](#licence)

---

## Ce que calcule le NPU

Plus de 90 % du calcul d'un réseau de neurones est un **produit de matrices**. Le NPU calcule une couche complète,
quantifiée en entiers :

```
Y = requant( ReLU( W · X + b ) )
```

| Élément | Format | Taille |
|---|---|---|
| W : poids | int8 signé | M × K |
| X : activations d'entrée | int8 signé | K × N |
| b : biais | int32 signé | M |
| accumulation | int32 signé | — |
| Y : activations de sortie | int8 signé | M × N |

- M, K et N vont de 1 à 64, sans contrainte de multiple : les bords sont gérés par masquage.
- La **requantification** ramène l'accumulateur int32 en int8 : `y = sat8((x · M_q + 2^(s−1)) >> s)`, avec un
  multiplicateur `M_q` sur 16 bits, un décalage `s` sur 5 bits et un arrondi au plus proche.
- La quantification est symétrique (point zéro = 0). Une convolution se ramène à ce calcul par im2col, côté hôte.
- Y sort dans le même format que X : la sortie d'une couche sert directement d'entrée à la suivante.

## Comment ça marche

Un processeur classique passe l'essentiel de son énergie à **déplacer** les données, pas à calculer. Un NPU est
une machine à **réutiliser** chaque donnée lue le plus de fois possible avant de la jeter.

Le cœur du NPU est une **grille systolique** de 4×4 *processing elements* (PE). Chaque PE fait une
multiplication-accumulation int8 × int8 → int32 par cycle et ne parle qu'à ses voisins immédiats. Les données
« pulsent » de PE en PE : une valeur lue une fois en mémoire sert à 4 MAC en traversant la grille. Pour que les bonnes
valeurs se rencontrent au bon PE, les données entrent **en escalier** : la voie l est retardée de l cycles.

```
            x0       x1       x2       x3          ← activations, décalées dans le temps
            ↓        ↓        ↓        ↓
   w0 →  [ PE ] → [ PE ] → [ PE ] → [ PE ]
            ↓        ↓        ↓        ↓
   w1 →  [ PE ] → [ PE ] → [ PE ] → [ PE ]        16 MAC par cycle au maximum
            ↓        ↓        ↓        ↓
   ...
```

Une matrice plus grande que la grille est découpée en **tuiles** de 4×4, que le séquenceur (`controller`)
parcourt une à une.

## Les deux variantes : output-stationary et weight-stationary

La question centrale d'un NPU est : *quelle donnée reste immobile dans le PE ?* Le projet implémente les deux
réponses classiques derrière la même interface, pour les comparer chiffres à l'appui.

| | Output-stationary (OS) | Weight-stationary (WS) |
|---|---|---|
| Reste dans le PE | l'accumulateur int32 | le poids int8 (préchargé) |
| Circule entre les PE | W vers l'est, X vers le sud (8 bits) | X vers l'est, sommes partielles vers le sud (32 bits) |
| Fin d'une tuile | on lit les accumulateurs ligne par ligne | les résultats sortent en escalier au sud |
| Réutilisation | W et X relus pour chaque tuile | chaque poids lu une seule fois par GEMM |
| Avantage mesuré | K grand, N petit, petits lots | N grand, gros lots |

La variante est choisie à la construction de la puce (`dataflow='OS'` ou `'WS'`). Les deux produisent exactement
le même Y, bit pour bit.

## Architecture de la puce

```
 Hôte ◄──► host_interface ──► controller ──────────────── (contrôle vers tous les blocs)
                                   │
 Mém. ext. ◄──► dma (v2) ──► weight_buffer ──────┐
                         └──► activation_buffer ─┼──► data_feeder ──► pe_array (N×N pe)
                                      ▲          │                        │
                                      │          │                  output_drain
                                      │          │                        │
                                      └──────────┴──────────────── post_processing
```

| Bloc | Rôle |
|---|---|
| `pe` | Processing element : multiplication-accumulation int8 → int32, registres de passage vers les voisins |
| `pe_array` | Grille N×N de PE (4×4 par défaut, paramétrable) |
| `data_feeder` | Aligne les commandes sur la mémoire, masque les voies hors limites, crée l'escalier systolique |
| `output_drain` | Récupère les résultats de la grille ; en WS, additionne les sommes partielles des tuiles de K |
| `post_processing` | Biais → ReLU → requantification int8, sur 4 voies en parallèle ; mémoire des biais |
| `weight_buffer` | SRAM des poids (4 Kio) |
| `activation_buffer` | SRAM des activations : banc IN (X) et banc OUT (Y), 4 Kio chacun |
| `controller` | FSM qui parcourt les tuiles et émet adresses, masques et commandes, au cycle près |
| `host_interface` | Bus de 32 bits, registres de configuration et de statut, fenêtres sur les mémoires |
| `npu_top` | Assemblage : la puce complète, vue de l'extérieur par son bus et sa ligne d'interruption |

Choix principaux (détaillés et justifiés dans [`src/architecture.md`](src/architecture.md)) :
- un seul domaine d'horloge, avec un reset synchrone actif bas ;
- un contrôle centralisé et déterministe : chaque latence est fixe et connue du `controller`, sans handshake ;
- des mémoires simple port à lecture synchrone : l'hôte y accède quand la puce est au repos, le cœur pendant le calcul ;
- un tag de 20 bits (adresse de Y, masque d'écriture, ligne) qui voyage avec chaque groupe de résultats, pour que
  l'écriture de Y n'ait pas besoin du `controller`.

## Programmer la puce

### Carte mémoire (adresses de mots de 32 bits)

| Adresse | Nom | Accès | Contenu |
|---|---|---|---|
| `0x0000` | `ID` | lecture | taille de grille, variante (0 = OS, 1 = WS), `DIM_MAX`, version |
| `0x0001` | `CTRL` | lecture / écriture | bit 0 `GO` (lance le calcul), bit 1 `RELU_EN` |
| `0x0002` | `STATUS` | lecture | bit 0 `BUSY`, bit 1 `DONE` (aussi sur la ligne `irq`) |
| `0x0003`–`0x0005` | `DIM_M`, `DIM_K`, `DIM_N` | lecture / écriture | dimensions, de 1 à 64 |
| `0x0006` | `REQ_MULT` | lecture / écriture | multiplicateur de requantification (16 bits) |
| `0x0007` | `REQ_SHIFT` | lecture / écriture | décalage de requantification (5 bits) |
| `0x0008` | `CYCLES` | lecture | durée du dernier calcul, en cycles |
| `0x0400`–`0x07FF` | `WEIGHT` | lecture / écriture | mémoire des poids |
| `0x0800`–`0x0BFF` | `ACT_IN` | lecture / écriture | activations d'entrée X |
| `0x0C00`–`0x0FFF` | `ACT_OUT` | lecture / écriture | activations de sortie Y |
| `0x1000`–`0x103F` | `BIAS` | lecture / écriture | biais int32 |

Un mot de mémoire contient 4 valeurs int8. La façon de ranger W, X et Y en mots dépend de la variante : elle est
décrite dans `src/architecture.md` (D6) et implémentée côté hôte dans [`test/layout.py`](test/layout.py). Pendant un
calcul, les accès aux mémoires et les écritures de configuration sont ignorés ; les registres restent lisibles.

### Séquence pour une couche

1. Écrire W dans `WEIGHT`, X dans `ACT_IN` et b dans `BIAS`.
2. Écrire `DIM_M`, `DIM_K`, `DIM_N`, `REQ_MULT` et `REQ_SHIFT`.
3. Écrire `CTRL = GO | RELU_EN`.
4. Attendre `irq`, ou lire `STATUS.DONE` en boucle.
5. Lire Y dans `ACT_OUT`, et `CYCLES` pour la mesure.

Pour la couche suivante, il suffit de recopier les mots de Y de `ACT_OUT` vers `ACT_IN`.

### En Python, sur la puce simulée

La classe `Npu` de [`test/npu_top/test.py`](test/npu_top/test.py) est le logiciel hôte de référence. Elle ne connaît
que le bus et la carte mémoire. Depuis la racine du dépôt :

```python
import sys
sys.path.insert(0, 'test')
from common import load_test_module
import numpy as np

npu_test = load_test_module('npu_top')
npu = npu_test.Npu('OS')                      # ou 'WS'

rng = np.random.default_rng(0)
W = rng.integers(-128, 128, (16, 32))         # poids int8, M×K
X = rng.integers(-128, 128, (32, 8))          # activations int8, K×N
b = list(rng.integers(-1000, 1000, 16))       # biais int32, un par ligne de Y

Y, cycles, phases = npu.gemm(W, X, b, mult=300, shift=16, relu=True)
print(Y.shape, cycles, phases)
# (16, 8) 357 {'chargement': 213, 'calcul': 360, 'relecture': 33}
```

## Résultats mesurés

Toutes les valeurs viennent de la simulation (grille 4×4). La liste complète, avec le fichier qui produit chaque
mesure, est dans `src/architecture.md` (section 3b).

| Mesure | OS | WS |
|---|---|---|
| GEMM 64×64×64, calcul seul (go → done) | 19 461 cycles | 18 701 cycles |
| Débit d'une couche 64³, calcul seul / transferts de l'hôte compris | 13,5 / 11,6 MAC/cycle | 14,0 / 12,0 MAC/cycle |
| Utilisation des PE quand K = 4 (M = N = 32) | 25 % | 75 % |
| Utilisation des PE quand N = 4 (M = K = 32) | 72 % | 30 % |
| MAC utiles par octet lu en mémoire (tuile, flux de 64) | 2,0 | 3,8 |
| Lectures de W pour un GEMM 32³ (minimum : 256) | 2 048 | 256 |
| Couche 64 → 64 : cycles par échantillon, lot de 1 / lot de 64 | 2 446 / 353 | 3 702 / 341 |
| MLP 64 → 32 → 16 quantifié, comparé au calcul flottant | corrélation 0,9999 | identique |

Ce que ces chiffres enseignent :
- **Aucune variante ne gagne partout.** Selon la forme du GEMM, l'une est jusqu'à 3,4 fois plus rapide que l'autre
  (carte de décision dans `test/controller/benchmark.py`).
- **WS réutilise mieux les données** : chaque poids n'est lu qu'une fois, et il faut 2 fois moins de débit mémoire.
- **Sans DMA, l'hôte coûte cher** : sur une petite couche, le chargement et la relecture prennent plus de la moitié
  du temps total.
- **L'accumulateur int32 est surdimensionné** : pour K ≤ 64, 22 bits suffisent ; un multiplicateur de
  requantification sur 14 bits donne déjà le résultat exact.

## Organisation du dépôt

```
README.md                    ce fichier
CLAUDE.md                    guide de reprise pour les agents IA
LICENSE                      licence Apache 2.0
src/
  architecture.md            décisions d'architecture, conventions d'interface, état, résultats, journal
  <bloc>/interface.md        spécification d'un bloc : ports, comportement, latences, implémentation
  <bloc>/<bloc>.py           implémentation py4hw (comportementale)
test/
  README.md                  organisation et conventions des tests
  common.py                  banc de test py4hw et utilitaires
  viz.py                     style des figures, rendu rapide des animations
  layout.py                  disposition des données en mémoire, côté hôte
  <bloc>/test.py             tests fonctionnels
  <bloc>/benchmark.py        mesures des limites et des compromis
  <bloc>/animation.py        visualisation du bloc cycle par cycle
```

Documents à lire, dans l'ordre :
1. [`src/architecture.md`](src/architecture.md) : le *pourquoi* (décisions D1 à D12 et leur justification) ;
2. `src/<bloc>/interface.md` : le *quoi* de chaque bloc ;
3. [`test/README.md`](test/README.md) : comment tout est vérifié.

## Installation

Testé avec Python 3.13.

```
python -m venv .venv
source .venv/bin/activate
pip install py4hw numpy matplotlib
```

Les benchmarks et les animations ouvrent une fenêtre matplotlib : il faut un backend graphique (par exemple Tk).
Avec l'option `--save`, ils écrivent un fichier et n'en ont pas besoin.

## Tests, benchmarks et animations

Chaque bloc a trois fichiers dans `test/<bloc>/`, à lancer depuis la racine du dépôt :

| Fichier | Rôle | Commande |
|---|---|---|
| `test.py` | vérifie que le bloc fait exactement ce que dit sa spécification | `python test/<bloc>/test.py` |
| `benchmark.py` | mesure ses limites et ses compromis (figure à 4 panneaux) | `python test/<bloc>/benchmark.py [--save fig.png]` |
| `animation.py` | le montre fonctionner, cycle par cycle (≥ 30 images/s) | `python test/<bloc>/animation.py [--save anim.gif]` |

Tous les tests :

```
for t in pe pe_array data_feeder output_drain post_processing weight_buffer activation_buffer controller host_interface npu_top; do
    python test/$t/test.py | tail -1
done
```

Quelques points d'entrée :

```
python test/npu_top/animation.py              # la puce complète pendant une couche
python test/npu_top/animation.py --variant WS # la même chose en weight-stationary
python test/pe_array/animation.py             # les données qui traversent la grille systolique
python test/controller/benchmark.py           # quelle variante choisir selon la forme du GEMM
python test/npu_top/benchmark.py              # la puce sur de vraies charges
```

## Méthode de travail

- **La spécification d'abord.** Chaque décision est prise, justifiée et datée dans `src/architecture.md` avant
  d'être implémentée ; chaque bloc a sa spécification avant son code.
- **Tout est vérifié contre une référence indépendante** : NumPy pour les résultats, des modèles de référence cycle
  par cycle pour le comportement, et les latences annoncées au cycle près. Pour s'assurer que les tests détectent
  vraiment les erreurs, on introduit volontairement des défauts dans les blocs et on vérifie qu'ils échouent.
- **On assemble progressivement.** Le cœur de calcul est d'abord piloté par un modèle Python du séquenceur, puis on
  ajoute le post-traitement, les mémoires, le vrai `controller`, et enfin le bus : à chaque étape, tout le chemin
  est revérifié de bout en bout.
- **On mesure les compromis au lieu de les supposer**, et on les montre : chaque bloc a son benchmark et son
  animation.

## Feuille de route

| Version | Cible | Priorité | État |
|---|---|---|---|
| v1 | Simulation (py4hw) | Comprendre, mesurer les compromis | ✅ complète |
| v2 | FPGA | Faire tenir et fonctionner sur du vrai matériel | à venir |
| v3 | Tiny Tapeout | Surface et broches minimales | à venir |

Pistes identifiées par les mesures de la v1 :
- un **DMA** et un double tampon mémoire, pour que le calcul n'attende plus les transferts de l'hôte ;
- en OS, **recouvrir le calcul et la lecture des accumulateurs** (39 % des cycles d'un GEMM 16³) ;
- en WS, un **double registre de poids** pour recouvrir le préchargement (16 % des cycles) ;
- un **accumulateur réduit** (22 bits) et un multiplicateur de requantification plus étroit, pour la v3 ;
- un bit d'échange des bancs IN / OUT, pour enchaîner les couches sans recopie.

## Contribuer

L'architecture en py4hw est développée par un humain avec l'aide d'agents IA ; la transcription en Verilog est
faite entièrement à la main par un humain. Les agents IA (et les personnes) qui reprennent le travail trouveront
dans [`CLAUDE.md`](CLAUDE.md) le périmètre, les conventions du code et des tests, et les pièges connus de py4hw.

## Licence

Apache 2.0 — voir [`LICENSE`](LICENSE).
