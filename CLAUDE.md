# Guide de reprise pour les agents IA

Ce fichier est destiné aux agents IA qui travaillent sur ce dépôt. Lis-le en entier avant de modifier quoi que ce
soit, puis lis [`src/architecture.md`](src/architecture.md) et [`test/README.md`](test/README.md).

## 1. Le projet et ton périmètre

`minuscul_npu` est un petit NPU (accélérateur de réseaux de neurones) conçu pour **apprendre** : comprendre une
architecture fonctionnelle et **explorer ses compromis**. Il calcule `Y = requant(ReLU(W·X + b))` en int8 sur une
grille systolique 4×4, en deux variantes comparées : output-stationary (OS) et weight-stationary (WS).

| Partie | Qui |
|---|---|
| Architecture en py4hw (blocs, bancs de test, benchmarks, animations, documentation) | **les agents IA** |
| Décisions d'architecture | **l'utilisateur** (l'IA analyse, propose, puis consigne la décision) |
| Transcription en Verilog, FPGA, Tiny Tapeout | **un humain, entièrement à la main** |

**Ne travaille pas sur le Verilog** : ne l'écris pas, ne le prépare pas, ne le propose pas comme étape suivante,
n'ajoute pas de commentaires « équivalent Verilog » dans le code. L'utilisateur peut demander de l'aide sur d'autres
parties : fais-le seulement s'il le demande.

## 2. État actuel (v1 complète)

Les 10 blocs de la v1 sont implémentés, testés (96 tests), mesurés et animés ; la puce complète (`npu_top`),
pilotée uniquement par son bus de 32 bits, calcule des couches justes et exécute un petit réseau de neurones quantifié.
L'état bloc par bloc, les résultats mesurés et les **pistes non encore explorées** sont dans la section 3b de
`src/architecture.md` : c'est le point de départ naturel pour la suite (v2 : DMA, double tampon, recouvrement
calcul / vidage, accumulateur réduit, etc.). Le dossier `dma/` est réservé à la v2. Le dossier `algorithms/`, à la
racine, est vide : demande à l'utilisateur à quoi il doit servir avant d'y mettre quoi que ce soit.

## 3. Carte du dépôt

```
README.md                    présentation publique du projet
CLAUDE.md                    ce guide
src/architecture.md          décisions (D1 à D12), conventions d'interface (§5), état et résultats (§3b), journal (§4)
src/<bloc>/interface.md      spécification d'un bloc : ports, comportement, latences, puis « Implémentation »
src/<bloc>/<bloc>.py         implémentation py4hw
test/README.md               organisation des tests, outils partagés, pilotes
test/common.py               Bench (banc py4hw), complément à deux, pack/unpack, load_test_module, run_tests
test/viz.py                  palette, style des figures, moteur de rendu rapide des animations
test/layout.py               disposition des données en mémoire (D6), côté logiciel hôte
test/<bloc>/test.py          tests fonctionnels
test/<bloc>/benchmark.py     limites et compromis (figure matplotlib)
test/<bloc>/animation.py     visualisation cycle par cycle (≥ 30 images/s)
```

Fonctions de référence réutilisées d'un bloc à l'autre (à charger avec `load_test_module`) :

| Où | Quoi |
|---|---|
| `test/output_drain/test.py` | `plan_os` / `plan_ws` (ce que la FSM du controller émet, cycle par cycle), `core(variant, post, mem, ctrl)` (assemblage incrémental du NPU), `run_core`, `rebuild_y`, `tag_pack` |
| `test/post_processing/test.py` | `reference` (Y attendu, requantification comprise), `gemm_complet` |
| `test/activation_buffer/test.py` | pilote hôte du chemin de données : `datapath`, `load_*`, `read_ab`, `run_layer` (avec journal des accès), `run_go` |
| `test/npu_top/test.py` | classe `Npu` : le logiciel hôte de référence, qui pilote la puce uniquement par son bus |
| `test/npu_top/animation.py` | `trace_puce` : activité de chaque bloc de la puce, cycle par cycle |
| `test/weight_buffer/animation.py` | `MemGrid`, `Timeline`, `etats_cumules`, `run_animation` (boucle d'animation commune) |

## 4. Lancer les tests

Python 3.13 avec `py4hw`, `numpy` et `matplotlib` (voir le README). Depuis la racine :

```
for t in pe pe_array data_feeder output_drain post_processing weight_buffer activation_buffer controller host_interface npu_top; do
    python test/$t/test.py | tail -1
done
```

Tout doit passer avant et après chaque modification. Les fichiers s'appellent `test.py` : pytest ne les découvre pas
tout seul, on les lance avec `python`. `test/compute_core/` ne contient que des fichiers d'attente : le cœur de calcul
est testé dans `test/output_drain/`.

## 5. Conventions de code (demandées par l'utilisateur)

- **py4hw comportemental** : une classe `Logic` par bloc et par variante (`…_OS`, `…_WS`), avec `clock()` pour les
  registres (sorties publiées par `wire.prepare()`) et `propagate()` pour la logique combinatoire (`wire.put()`).
  Les blocs purement structurels (`pe_array`, `npu_top`) se contentent d'instancier et de câbler.
- **La spécification d'abord** : un bloc suit `src/<bloc>/interface.md` et les conventions de
  `src/architecture.md` §5 (paramètres, nommage, voie l dans les bits `[W·l + W−1 : W·l]`, latences, cycle
  d'émission, tag). Si l'implémentation oblige à changer la spécification, mets-la à jour et consigne-le au journal.
- **Pas d'estimation du coût matériel** dans les benchmarks (bascules, multiplieurs…) : on mesure le comportement
  (cycles, débit, utilisation, accès mémoire, précision).
- **Langue** : tout est en français (code commenté, documentation, messages, figures).
- **Dépôt public** : n'écris jamais d'information sur la machine de l'utilisateur (chemins locaux, système, nom
  d'utilisateur, contournements présentés comme spécifiques à sa machine). Avant de finir, vérifie :
  `grep -rniE "/mnt/|wsl|windows|/home/|/tmp/" --include=*.py --include=*.md src test README.md` (ce guide contient le motif lui-même, il est donc exclu).

## 6. Bancs de test, benchmarks, animations

- `test.py` : chaque résultat est comparé à une référence indépendante (NumPy, modèle de référence cycle par cycle,
  planificateur). Les latences annoncées sont vérifiées **au cycle près** (ni avant, ni après).
  Quand des tests passent du premier coup, vérifie qu'ils mordent : introduis temporairement un défaut dans le bloc
  (en le remplaçant à l'exécution) et contrôle que les tests concernés échouent.
- `benchmark.py` : une figure matplotlib à 4 panneaux, option `--save fichier.png`, résumé chiffré dans le terminal,
  style `viz.BENCH_RC`. **Couleurs fixes : OS en bleu (`viz.C_OS`), WS en orange (`viz.C_WS`)** ; n'utilise jamais
  ces deux couleurs pour autre chose. Regarde la figure produite avant de conclure (étiquettes qui se chevauchent,
  échelles log avec des zéros, etc.).
- `animation.py` : **au moins 30 images par seconde, à vitesse constante**. L'utilisateur tient à un style visuel
  soigné : jetons qui circulent, cases qui s'allument (voir `test/pe_array/animation.py` et `test/npu_top/animation.py`).
  Recette, toujours la même :
  1. simuler entièrement avant d'animer (trace par cycle) ;
  2. dessiner jetons, cases et chiffres avec `viz.Layer`, `viz.TextLayer`, `viz.TokenLayers` (une collection de
     tracés par couche, chiffres convertis en tracés) — jamais des milliers d'objets `Text` ou `Patch` ;
  3. blitting : tout artiste animé doit appartenir à un axe (titre animé : `viz.title_axes`) et ne doit pas toucher
     le bord de l'axe (sinon il laisse une trace) ; le texte statique est recouvert par les couches animées ;
  4. image choisie d'après l'horloge réelle (`run_animation` dans `test/weight_buffer/animation.py`) ;
  5. vérifier sans écran en passant par le vrai mécanisme de `FuncAnimation` (`anim._init_draw()`, puis
     `anim._draw_next_frame(f, True)`), mesurer le temps par image (cible < 15 ms en moyenne) et regarder des images.
  Option `--save fichier.gif` dans chaque animation.

## 7. Pièges de py4hw rencontrés

- **Fausse boucle combinatoire** : le simulateur trie les blocs ayant un `propagate()` *bloc par bloc*. Deux blocs
  qui se lisent mutuellement (même à travers des registres) provoquent « Excessive loop count in topological
  count ». Solution utilisée : un bloc dont les sorties ne dépendent que de son état (machine de Moore) publie ses
  sorties dans `clock()` et n'a pas de `propagate()` (c'est le cas du `controller`).
- **Lire une sortie combinatoire** : après `Bench.step()`, on lit l'état *après* le front. Pour lire ce qu'un bloc
  sort *pendant* un cycle, utiliser `Bench.observe()` (entrées, propagation, lecture, puis front).
- **`import test`** tombe sur le module `test` de la bibliothèque standard : charger les bancs avec
  `load_test_module('<bloc>')`.
- **Polices** : matplotlib relit le fichier de police à chaque texte ; selon le système de fichiers c'est très lent.
  `common.setup_matplotlib_fonts()` (appelé par `viz.py`) utilise la copie système quand elle existe.
- **Vitesse de simulation** (ordre de grandeur) : un PE ≈ 400 000 cycles/s, la grille 4×4 ≈ 30 000, la puce
  complète ≈ 11 000 à 13 000 cycles/s. Dimensionne les benchmarks en conséquence.
- **Valeurs nulles** : dans les animations et les mesures d'activité, utilise des données non nulles, sinon un MAC
  avec un opérande nul n'est pas vu comme actif. En WS, un MAC peut avoir lieu pendant le premier cycle de
  rechargement des poids : ne pas exclure les cycles où `load = 1` quand on compte les MAC.

## 8. Méthode de travail avec l'utilisateur

- L'utilisateur écrit en français, apprend en construisant, et veut **comprendre les compromis**.
- **Les décisions d'architecture lui appartiennent** : explique les options et leurs conséquences, recommande, puis
  pose la question ; consigne ensuite la décision dans le journal de `src/architecture.md` (§4). Les décisions de
  détail que tu prends toi-même pendant l'implémentation doivent aussi y être consignées et lui être signalées.
- Après chaque bloc ou changement : mettre à jour la section « Implémentation » de `src/<bloc>/interface.md`,
  l'état et les résultats (§3b) de `src/architecture.md`, `test/README.md`, le README, et ce guide si besoin ;
  vérifier que les liens des fichiers `.md` pointent vers des fichiers existants.
- Le schéma qui a été validé pour un bloc : implémentation + `test.py` + `benchmark.py` + `animation.py` +
  documentation, avec un compte rendu qui signale les erreurs trouvées et corrigées en route.
