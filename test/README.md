# Organisation des tests

Chaque sous-dossier de `test/` est le banc de test d'**un bloc** ou d'**un assemblage de blocs**.
Le nom d'un bloc est le même que celui de son dossier dans `src/`.

## Lancer les tests

Depuis la racine du dépôt, avec l'environnement Python du projet activé :

```
python test/<bloc>/test.py                      # tests fonctionnels : bilan « n/n tests réussis »
python test/<bloc>/benchmark.py                 # figure matplotlib dans une fenêtre
python test/<bloc>/benchmark.py --save b.png    # … ou enregistrée dans un fichier
python test/<bloc>/animation.py                 # animation dans une fenêtre (≥ 30 images/s)
python test/<bloc>/animation.py --save a.gif    # … ou enregistrée en GIF (environ une minute)
```

Les fichiers s'appellent `test.py` et non `test_*.py` : pytest ne les découvre pas automatiquement. On les lance
directement avec `python`, et `test.py` renvoie un code de sortie non nul en cas d'échec.

## Les trois fichiers d'un banc de test

| Fichier | Rôle | Question à laquelle il répond | Résultat |
|---|---|---|---|
| `test.py` | Test **fonctionnel** | « Le bloc fait-il exactement ce que la spécification demande ? » | Succès / échec |
| `benchmark.py` | **Limites et compromis** | « Combien de cycles, quel débit, où ça casse ? » | Figure matplotlib à 4 panneaux + résumé chiffré dans le terminal |
| `animation.py` | **Visualisation** au fil des cycles | « Que se passe-t-il à chaque front d'horloge ? » | Animation des valeurs qui circulent dans le bloc |

### Règles
- **Modèle de référence** : tout résultat matériel est comparé à un calcul indépendant — NumPy pour les résultats,
  et un modèle de référence Python écrit depuis la spécification pour les comparaisons cycle par cycle.
- **Variantes de dataflow** : pour les blocs qui existent en deux versions (`pe`, `pe_array`, `data_feeder`,
  `output_drain`), les trois fichiers traitent **les deux variantes, OS et WS**, afin que
  `benchmark.py` puisse les comparer directement.
- **Paramètres** : les tests balaient les paramètres de l'architecture (taille de grille N, dimensions M/K/N,
  largeurs de bits) plutôt que de se limiter aux valeurs par défaut.
- **Cas limites** : ils sont vérifiés dans `test.py` : valeurs extrêmes int8 (-128, 127), débordement,
  dimensions non multiples de N, K = 1, reset en cours d'opération.
- **Timing exact** : quand la spécification annonce une latence, le test vérifie que le résultat arrive
  **à ce cycle-là**, ni avant ni après.
- `test.py` doit passer avant qu'on fasse tourner `benchmark.py` ou `animation.py`.
- **Benchmarks** : on mesure le comportement (cycles, débit, utilisation, limites numériques), pas le coût matériel.
  Couleurs fixes : OS en bleu, WS en orange.
- **Animations** : au moins 30 images par seconde, à vitesse constante. Toutes les valeurs affichées viennent de la
  simulation py4hw, calculée entièrement avant l'animation.

## Outils partagés

| Fichier | Contenu |
|---|---|
| [`common.py`](common.py) | `Bench` : système py4hw minimal autour d'un bloc (`step()` avance d'un front ; `observe()` lit les sorties **pendant** le cycle, utile pour les sorties combinatoires) ; `to_c2` / `from_c2` / `wrap` (complément à deux) ; `pack` / `unpack` (bus de voies) ; `load_test_module(bloc)` pour réutiliser les bancs et modèles d'un autre bloc ; `run_tests()` ; `setup_matplotlib_fonts()`. |
| [`layout.py`](layout.py) | Disposition des données en mémoire (D6), côté logiciel hôte : `w_image`, `x_image`, `y_addresses`, `y_from_words`. |
| [`viz.py`](viz.py) | Palette et style des figures ; moteur de rendu rapide des animations : les jetons, cases et chiffres sont des tracés regroupés en collections (`Layer`, `TextLayer`, `TokenLayers`), redessinés avec le blitting. |

Un fichier `test.py` d'un autre bloc se charge avec `load_test_module('<bloc>')` : un `import test` classique
tomberait sur le module `test` de la bibliothèque standard de Python.

## Bancs de test

### Blocs unitaires
| Dossier | Bloc testé | État |
|---|---|---|
| `pe/` | Un Processing Element (MAC, registres de passage, clear de l'accumulateur, chargement du poids) | ✅ |
| `pe_array/` | Grille N×N de PE, sans le data_feeder (les flux sont injectés à la main selon le contrat de la spécification) | ✅ |
| `data_feeder/` | Alignement, masquage et décalage en escalier des flux W et X ; chaîne data_feeder → pe_array | ✅ |
| `output_drain/` | Vidage (OS), redressement et accumulation sur les tuiles de K (WS) ; cœur de calcul de bout en bout | ✅ |
| `post_processing/` | Biais → ReLU → requantification → saturation ; chaîne complète jusqu'à l'écriture de Y, MLP à 2 couches | ✅ |
| `weight_buffer/` | SRAM des poids ; disposition de W relue aux adresses du planificateur | ✅ |
| `activation_buffer/` | SRAM des activations (bancs IN / OUT) ; chemin de données complet avec ses mémoires, piloté comme par l'hôte ; deux couches enchaînées | ✅ |
| `controller/` | FSM identique cycle par cycle au planificateur de référence ; NPU complet sans `host_interface`, piloté par go / done ; carte de décision OS / WS | ✅ |
| `host_interface/` | Décodage du bus, registres, impulsion go, DONE / irq, latence de lecture ; coût du bus sur la puce complète | ✅ |

### Assemblages
| Dossier | Contenu | État |
|---|---|---|
| `compute_core/` | Cœur de calcul : `data_feeder` + `pe_array` + `output_drain` | fichiers d'attente : le cœur est testé et mesuré dans `output_drain/` |
| `npu_top/` | Puce complète pilotée uniquement par son bus : l'hôte écrit W, X et b, lance GO, attend DONE et relit Y | ✅ |

`dma/` n'a pas de banc de test : ce bloc est prévu pour la v2.

## Planificateur de référence du controller

[`output_drain/test.py`](output_drain/test.py) contient `plan_os` et `plan_ws` : pour un GEMM donné, ils produisent
cycle par cycle ce que la FSM du `controller` doit émettre (mots lus en SRAM, masques, `clear` / `load`, commandes de
vidage ou sideband, tags, phase), selon [`src/controller/interface.md`](../src/controller/interface.md).
`core(variant, post=False, mem=False, ctrl=False)` et `run_core(bench, plan)` exécutent le cœur de calcul complet avec
ce plan (avec `post=True`, le `post_processing` est branché à la suite ; avec `mem=True`, les deux buffers aussi, et le
plan émet des adresses au lieu de fournir les mots ; avec `ctrl=True`, le vrai `controller` remplace le plan), et `rebuild_y` reconstruit
Y à partir des seuls résultats sortis. Ces fonctions serviront de référence pour tester le vrai `controller`.

## Pilote hôte

[`activation_buffer/test.py`](activation_buffer/test.py) contient les fonctions qui pilotent le chemin de données
complet comme le fera l'hôte : `datapath(variant)`, `load_wb`, `load_ab`, `read_ab`, `load_bias`, `copy_out_to_in` et
`run_layer` (chargement, GEMM, relecture de Y ; avec `log=[]`, un journal de tous les accès mémoire, cycle par cycle).
Avec le controller (`ctrl=True`), le GEMM est lancé par `run_go` : impulsion `go` avec les dimensions, puis attente
de `done`.

## Logiciel hôte de référence

[`npu_top/test.py`](npu_top/test.py) contient la classe `Npu` : elle pilote la puce complète uniquement par son bus de
32 bits (`write`, `read`), comme le fera le vrai processeur hôte, avec la carte mémoire de `architecture.md` (D7) et la
disposition des données de `layout.py` (D6) : `load_w`, `load_x`, `load_bias`, `configure`, `start`, `wait_done`,
`read_y`, `copy_out_to_in`, et `gemm` qui enchaîne le tout pour une couche. [`npu_top/animation.py`](npu_top/animation.py)
fournit `trace_puce`, qui relève l'activité de chaque bloc de la puce cycle par cycle.

