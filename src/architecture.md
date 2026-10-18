# Architecture du minuscul_npu

Document de référence des décisions d'architecture.
Prototypage en py4hw (travail partagé avec des agents IA), puis transcription manuelle en Verilog (faite
entièrement par un humain, hors du périmètre des agents). Guide de reprise pour les agents : [`CLAUDE.md`](../CLAUDE.md).

Statuts : `OUVERT` (à décider) · `PROPOSÉ` (recommandation, à confirmer) · `DÉCIDÉ`

L'état d'avancement de l'implémentation et les résultats mesurés sont en [section 3b](#3b-implémentation-et-vérification).

---

## 0. Objectifs et feuille de route

**Objectif principal** : apprendre et comprendre une architecture de NPU *fonctionnelle*,
et explorer ses compromis et ses points d'amélioration.

**Principe directeur** : architecture **paramétrable** (taille de grille, largeurs de bits,
tailles de buffers) afin de pouvoir mesurer l'impact de chaque choix.

| Version | Cible | Priorité | Architecture |
|---|---|---|---|
| v1 | Simulation (py4hw) | Clarté, compréhension | Référence |
| v2 | FPGA | Faire tenir et fonctionner sur du vrai matériel | Dérivée de v1, peut diverger |
| v3 | Tiny Tapeout | Surface et broches minimales | Peut diverger fortement |

---

## 1. Vue d'ensemble

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

## 2. Blocs fonctionnels

L'interface détaillée de chaque bloc (ports, largeurs, latences) est dans `src/<bloc>/interface.md`,
selon les conventions de la [section 5](#5-conventions-dinterface).

| Bloc | Fonction |
|---|---|
| `npu_top` | Assemblage de tous les blocs, broches de la puce |
| `host_interface` | Protocole de bus hôte + banc de registres de configuration/statut |
| `controller` | Séquenceur : boucles de tuiles, génération des adresses et des signaux de contrôle |
| `dma` | Transferts mémoire externe ↔ buffers internes |
| `weight_buffer` | SRAM locale des poids W |
| `activation_buffer` | SRAM locale des activations X (entrées et sorties de couche) |
| `data_feeder` | Lecture des buffers et injection décalée (en escalier) dans la grille |
| `pe` | Processing Element : registres de passage w/x + MAC + accumulateur |
| `pe_array` | Grille N×N de `pe` connectés aux voisins |
| `output_drain` | Récupération et sérialisation des accumulateurs |
| `post_processing` | Biais → activation → requantification → saturation |

---

## 3. Décisions d'architecture

### D1 — Cible d'implémentation · `DÉCIDÉ` : par étapes (simulation → FPGA → Tiny Tapeout)
- v1 en simulation, optimisée pour l'apprentissage : pas de contrainte physique.
- v2 sur FPGA (modèle à préciser plus tard).
- v3 sur Tiny Tapeout (~8 entrées / 8 sorties / 8 bidirectionnelles, surface minuscule).
- Chaque version peut avoir sa propre architecture. Les écarts par rapport à v1 seront documentés.

### D2 — Charge de travail de référence · `DÉCIDÉ` : GEMM générique
- Le NPU calcule `Y = W · X (+ b)` pour des dimensions M, K, N paramétrables.
- Pas de convolution native : une convolution passe par im2col, fait côté hôte.
- Conséquence : le pooling sort du périmètre de `post_processing` (D10).

### D3 — Dataflow · `DÉCIDÉ` : output-stationary ET weight-stationary, comparables
- Les deux variantes sont implémentées derrière **la même interface externe**, pour pouvoir les comparer.
- Row-stationary : écarté (trop complexe, et pensé pour les convolutions).

| | Output-stationary (OS) | Weight-stationary (WS) |
|---|---|---|
| Stationnaire dans le PE | accumulateur int32 | poids int8 |
| Circule entre PE | W (8 b) →, X (8 b) ↓ | X (8 b) →, psum (32 b) ↓ |
| Phase de préchargement | non | oui (poids) |
| Sortie des résultats | vidage des accumulateurs | flux en bas de la grille |
| Grille pour une tuile | M × N | K × M |

**Frontière de variante — le « cœur de calcul »** : ce n'est pas seulement le PE qui change.
`data_feeder`, `pe`, `pe_array` et `output_drain` diffèrent entre OS et WS.
Le cœur de calcul est donc l'ensemble de ces quatre blocs. Son interface commune :
- en entrée : lecture de `weight_buffer` et `activation_buffer` + commandes du `controller` ;
- en sortie : flux de résultats int32 vers `post_processing`.

Les buffers, le `controller` (à quelques phases près), `post_processing` et `host_interface` sont partagés.

**Métriques de comparaison** : nombre de cycles par GEMM, nombre de lectures SRAM,
nombre de registres et de bits de fils entre PE, taux d'utilisation des PE.

### D4 — Taille de la grille de PE · `DÉCIDÉ` : 4×4 par défaut, paramétrable
- 16 PE : assez pour observer le pipeline, le décalage et le tiling.
- Le paramètre N reste libre (2×2 pour le débogage, 8×8 pour l'exploration).
- Surface ∝ N², débit crête = N² MAC/cycle.

### D5 — Format numérique · `DÉCIDÉ` : int8 signé × int8 signé → accumulateur int32 signé
- Complément à deux partout.
- Produit 8×8 signé : au plus (−128)² = 16 384 = 2^14. Un accumulateur int32 ne peut donc pas déborder tant que
  K ≤ 131 071 (valeur mesurée sur le PE dans `test/pe/benchmark.py`), très au-delà de `DIM_MAX` = 64.
  Pour K ≤ 64, 22 bits suffiraient : piste de réduction pour la v3.
- Les largeurs restent des paramètres (exploration int4, accumulateur réduit, etc.).

### D6 — Mémoire interne · `DÉCIDÉ`
- `weight_buffer` et `activation_buffer` **séparés**, lus en parallèle (pas de conflit d'accès).
- **Simple buffer** : pas de recouvrement entre chargement et calcul en v1.
  Le double buffering est un compromis à explorer plus tard (utilisation des PE contre mémoire ×2).
- Largeur de lecture : une ligne de N éléments int8 par cycle (N×8 bits), pour alimenter la grille à plein débit.
- Les résultats int8 sont écrits dans `activation_buffer` (zone de sortie), où l'hôte vient les relire.
- **Tailles** : matrices jusqu'à **64×64** (paramètre `DIM_MAX`). Tout tient en mémoire,
  et c'est le `controller` qui fait tout le tiling.

| Mémoire | Contenu | Taille (DIM_MAX = 64) |
|---|---|---|
| `weight_buffer` | W, int8, M×K | 4 Kio |
| `activation_buffer` | X en entrée (K×N) + Y en sortie (M×N), int8 | 8 Kio (2 zones) |
| biais (dans `post_processing`) | b, int32, M valeurs | 256 o |
| accumulateurs (dans `output_drain`, **WS seulement**) | sommes partielles int32 : une entrée par colonne j (64), N = 4 voies | 1 Kio |

- Registres de dimensions M, K, N : 7 bits chacun, valeurs de 1 à 64.
- Avec une grille 4×4, un GEMM 64×64×64 compte (64/4)² = 256 tuiles de sortie, chacune de 64 pas en K.

#### Disposition des données (layout) · `DÉCIDÉ` : tuilée, propre à chaque variante
- Un mot mémoire fait 32 bits et contient **4 × int8**. Ce sont exactement les 4 éléments que la grille consomme
  en un cycle. L'élément de la voie `l` (l = 0..3) occupe l'octet `l`, l'octet 0 étant celui de poids faible.
- La taille des tuiles est fixe (`DIM_MAX` = 64), donc l'adresse se forme par **concaténation de bits**,
  sans multiplieur : `adresse = {indice_de_tuile[3:0], indice[5:0]}`.

| Matrice | Variante | Mot à l'adresse `{t, u}` | contient |
|---|---|---|---|
| W | OS et WS | `{i_t, k}` | `W[4·i_t + l][k]` |
| X | OS | `{j_t, k}` | `X[k][4·j_t + l]` |
| X | WS | `{k_t, j}` | `X[4·k_t + l][j]` |
| Y | OS | `{j_t, i}` | `Y[i][4·j_t + l]` |
| Y | WS | `{i_t, j}` | `Y[4·i_t + l][j]` |

- Le format de **W est commun** aux deux variantes. Celui de **X est transposé** entre OS et WS.
- Chaque variante écrit Y **dans le format X qu'elle attend en entrée**. On peut donc donner la sortie d'une couche
  directement en entrée de la couche suivante, sans reformatage.
- C'est l'hôte qui formate W et X selon la variante. Compromis écarté pour l'instant : un transposeur matériel,
  ou des bancs entrelacés en diagonale (*skewed storage*), à explorer dans une version ultérieure.

### D7 — Interface hôte · `DÉCIDÉ` : bus parallèle simple de 32 bits
- Signaux : `addr[12:0]` (adresse de mot), `wdata[31:0]`, `we`, `re`, `rdata[31:0]`.
  Un accès par cycle, latence de lecture fixe.
- Un mot du bus correspond à un mot des buffers (4 × int8) ou à un biais int32.
- Adaptateurs prévus : Wishbone (v2 FPGA), SPI (v3 Tiny Tapeout).

#### Carte mémoire · `DÉCIDÉ`
| Adresse (mots) | Nom | Accès | Contenu |
|---|---|---|---|
| `0x0000` | `ID` | RO | type de dataflow (0 = OS, 1 = WS), taille de grille N, `DIM_MAX` |
| `0x0001` | `CTRL` | RW | b0 `GO` (remis à 0 automatiquement), b1 `RELU_EN` |
| `0x0002` | `STATUS` | RO | b0 `BUSY`, b1 `DONE` (remis à 0 par `GO`) |
| `0x0003` | `DIM_M` | RW | 1..64 |
| `0x0004` | `DIM_K` | RW | 1..64 |
| `0x0005` | `DIM_N` | RW | 1..64 |
| `0x0006` | `REQ_MULT` | RW | M, 16 bits non signé |
| `0x0007` | `REQ_SHIFT` | RW | s, 5 bits |
| `0x0008` | `CYCLES` | RO | nombre de cycles du dernier GEMM, pour les benchmarks |
| `0x0009`–`0x03FF` | — | — | réservé |
| `0x0400`–`0x07FF` | `WEIGHT` | RW | `weight_buffer`, 1024 mots |
| `0x0800`–`0x0BFF` | `ACT_IN` | RW | `activation_buffer`, zone d'entrée X |
| `0x0C00`–`0x0FFF` | `ACT_OUT` | RW | `activation_buffer`, zone de sortie Y |
| `0x1000`–`0x103F` | `BIAS` | RW | biais int32, un par ligne de sortie |

- Pendant un GEMM (`BUSY = 1`, ou impulsion `go` en cours), les accès de l'hôte aux mémoires et les écritures de
  configuration sont **ignorés** ; les registres restent lisibles.
- Amélioration possible : un bit qui échange les rôles de `ACT_IN` et `ACT_OUT`, pour enchaîner les couches sans copie.

### D8 — Accès mémoire externe · `DÉCIDÉ` : pas de DMA en v1
- Les buffers sont mappés dans l'espace d'adresse de `host_interface`, et l'hôte les remplit et les relit directement.
- Le dossier `dma/` est réservé à la v2 (FPGA).

### D9 — Contrôle · `DÉCIDÉ` : FSM câblée
- L'hôte écrit la configuration (M, K, N, paramètres de post-traitement ; les zones mémoire sont fixes) dans des registres, puis lance `go`.
- Le `controller` exécute le GEMM complet (toutes les tuiles) et lève `done`.
- Les phases dépendent du dataflow : WS ajoute une phase de préchargement des poids.
- Évolution possible : mini-ISA dans une version ultérieure (compromis à explorer).

#### Ordre des boucles de tuiles · `DÉCIDÉ`
```
OS :  for i_t in tuiles(M):          WS :  for i_t in tuiles(M):
        for j_t in tuiles(N):                for k_t in tuiles(K):
          clear accumulateurs                  précharger W[4·i_t.., 4·k_t..]
          for k in 0..K-1:                     for j in 0..N-1:
            injecter W, X                        injecter X[4·k_t.., j]
          vider → post_processing                acc[.., j] += psum en bas de la grille
                                             acc → post_processing
```
- **OS** : k est forcément la boucle interne, puisque l'accumulation se fait dans le PE. Il n'y a pas de cache
  entre les buffers et la grille, donc l'ordre de (i_t, j_t) ne change pas le nombre de lectures SRAM.
- **WS** : avec i_t à l'extérieur et k_t au milieu, la mémoire d'accumulateurs se limite à 4 × N int32 (1 Kio).
  L'ordre k_t → i_t → j demanderait M × N int32 (16 Kio). C'est un contre-exemple à mesurer (pas encore fait).
- Dans les deux cas, i_t est à l'extérieur, donc le post-traitement reçoit les lignes de sortie dans l'ordre.

#### Dimensions non multiples de 4 · `DÉCIDÉ` : masquage en lecture et en écriture
- Le `data_feeder` injecte **0** dans chaque voie dont l'indice dépasse la limite (i ≥ M, j ≥ N ou k ≥ K).
  Un 0 est neutre pour le MAC, donc le calcul reste juste sans que l'hôte complète ses matrices.
- Les écritures de Y hors limites (i ≥ M ou j ≥ N) sont **bloquées** : chaque octet a son propre masque d'écriture.
- En OS, K n'a pas besoin de ce masquage (on envoie simplement K pas). En WS, ce sont les voies k ≥ K qui sont masquées.
- Coût matériel : un comparateur par voie et par dimension.
- Variante à comparer (pas encore faite) : un signal enable par PE, qui gèle les PE inutiles pour économiser de l'énergie.

### D10 — Opérations de post-traitement · `DÉCIDÉ` : biais → ReLU → requantification int8
Pipeline, dans cet ordre :
1. **Biais** : `acc + b[i]`, avec b en int32, un biais par ligne de sortie.
2. **ReLU** : `max(0, x)`, activable ou non par un bit de configuration.
3. **Requantification** : `y = sat8( (x × M + 2^(s-1)) >> s )`
   - M : 16 bits non signé (registre `REQ_MULT`). s : 5 bits, de 0 à 31 (registre `REQ_SHIFT`). Si s = 0, il n'y a pas d'arrondi.
   - Multiplieur 32 × 16 bits, produit sur 48 bits. Le décalage `>>` est arithmétique (il conserve le signe).
   - Arrondi au plus proche, demi vers +∞ (ajout de 2^(s-1)).
   - Saturation dans [-128, 127].
   - **Granularité par couche** : une seule paire (M, s) pour tout le GEMM.
     Une paire par ligne de sortie est une évolution possible (64 mots en plus, à côté des biais).
- **Quantification symétrique** : le point zéro vaut 0 pour W, X et Y, donc il n'y a pas de correction de zero-point.
- **Débit** : `post_processing` traite **4 voies en parallèle**, au rythme de la grille. Il contient donc 4 multiplieurs 32×16.
  Une version à une seule voie, plus petite mais plus lente au vidage, est un compromis à mesurer.
- Écartés : pooling (hors périmètre GEMM), LUT sigmoïde/tanh, bypass int32.
- Note : sans bypass, on vérifiera les accumulateurs bruts directement en simulation py4hw.

### D11 — Horloge et reset · `DÉCIDÉ` : une seule horloge, reset synchrone actif bas
- Un seul domaine d'horloge `clk`, donc pas de franchissement de domaine (CDC).
- Reset `rst_n` synchrone actif bas, comme la convention Tiny Tapeout.
- Fréquence : sans objet en v1. Elle sera fixée en v2 selon le FPGA.

### D12 — Protocole entre blocs · `DÉCIDÉ` : contrôle centralisé
- Le `controller` connaît la latence de chaque bloc et pilote tout de façon déterministe, au cycle près.
- Pas de valid/ready interne en v1. Ça deviendra nécessaire en v2 avec un DMA (latence mémoire variable).

---

## 3b. Implémentation et vérification

### Style d'implémentation
Chaque bloc est écrit en **py4hw comportemental**, dans `src/<bloc>/<bloc>.py` : une classe `Logic` dont la
méthode `clock()` décrit les registres (mis à jour au front, sorties publiées avec `prepare()`) et, si le bloc a des
sorties combinatoires, une méthode `propagate()` (sorties publiées avec `put()`). Les blocs purement structurels
(`pe_array`) se contentent d'instancier et de câbler d'autres blocs.
Les variantes de dataflow sont des classes distinctes (`…_OS`, `…_WS`) avec la même interface.

### Vérification
Chaque bloc et chaque assemblage a un banc de test dans `test/<bloc>/`, composé de `test.py` (fonctionnel),
`benchmark.py` (limites et compromis, figures matplotlib) et `animation.py` (visualisation cycle par cycle).
La convention complète est décrite dans [`test/README.md`](../test/README.md).

### État d'avancement
| Bloc | Implémentation | Tests | Benchmark | Animation |
|---|---|---|---|---|
| `pe` | ✅ `PE_OS`, `PE_WS` | ✅ 18 | ✅ | ✅ |
| `pe_array` | ✅ `PE_Array_OS`, `PE_Array_WS` | ✅ 10 | ✅ | ✅ |
| `data_feeder` | ✅ `DataFeeder_OS`, `DataFeeder_WS` | ✅ 7 | ✅ | ✅ |
| `output_drain` | ✅ `OutputDrain_OS`, `OutputDrain_WS` | ✅ 10, dont GEMM de bout en bout | ✅ | ✅ |
| `post_processing` | ✅ `PostProcessing_OS`, `PostProcessing_WS` | ✅ 12, dont GEMM de bout en bout | ✅ | ✅ |
| `weight_buffer` | ✅ `WeightBuffer` | ✅ 6 | ✅ | ✅ |
| `activation_buffer` | ✅ `ActivationBuffer` | ✅ 8, dont deux couches enchaînées par la mémoire | ✅ | ✅ |
| `controller` | ✅ `Controller_OS`, `Controller_WS` | ✅ 8, dont identité cycle par cycle avec le planificateur et NPU complet sans `host_interface` | ✅ | ✅ |
| `host_interface` | ✅ `HostInterface` | ✅ 9 | ✅ | ✅ |
| `npu_top` | ✅ `NpuTop` | ✅ 8, puce complète pilotée par son bus | ✅ | ✅ |
| `dma` | v2 | | | |

Le **chemin de données complet avec ses mémoires** (`weight_buffer`, `activation_buffer` → `data_feeder` →
`pe_array` → `output_drain` → `post_processing` → `activation_buffer`) est vérifié de bout en bout, piloté comme le
fera l'hôte : chargement de W, X et des biais selon la disposition D6, GEMM multi-tuiles (dimensions non multiples
de N, mémoire « sale », biais, ReLU, requantification), relecture de Y, puis deux couches enchaînées par simple
recopie OUT → IN. Les deux variantes écrivent exactement le même Y.

**La puce complète (`npu_top`) fonctionne** : pilotée uniquement par son bus de 32 bits et sa carte mémoire, par un
logiciel hôte de référence (`test/npu_top/test.py`), elle calcule des couches justes dans les deux variantes,
enchaîne des couches et exécute un petit réseau de neurones quantifié.

### Résultats mesurés (N = 4)
| Mesure | OS | WS | Source |
|---|---|---|---|
| Latences du PE | 1 cycle | 1 cycle | `test/pe/benchmark.py` |
| Cycles pour une tuile isolée (flux de longueur L) | L + 2N − 2 | L + 3N − 2 (préchargement inclus) | `test/pe_array/benchmark.py` |
| Écart minimal avant rechargement des poids | — | 2N − 2 cycles (FLUSH = 2N − 3) | `test/pe_array/test.py` |
| MAC utiles par octet lu en SRAM (tuile, flux de 64) | 2,0 | 3,8 | `test/data_feeder/benchmark.py` |
| Débit SRAM nécessaire (tuile, flux de 64) | 7,2 octets/cycle | 3,6 octets/cycle | `test/data_feeder/benchmark.py` |
| Voies perdues au masquage (D = 1..64, moyenne) | 8 % | 8 % | `test/data_feeder/benchmark.py` |
| GEMM 16×16×16 de bout en bout (dernier résultat) | 448 cycles | 403 cycles | `test/output_drain/benchmark.py` |
| Utilisation de la grille, GEMM 16³ / 32³ | 57 % / 73 % | 64 % / 78 % | `test/output_drain/benchmark.py` |
| Dernier mot de Y écrit, GEMM 16³ (avec post_processing) | cycle 451 | cycle 406 | `test/post_processing/benchmark.py` |
| Erreur de requantification, M sur 16 bits / 8 bits | 0 / 1 pas int8 | identique | `test/post_processing/benchmark.py` |
| MLP 64 → 32 → 16 sur le NPU simulé contre le flottant | corrélation 0,9998 | identique | `test/post_processing/benchmark.py` |
| Lectures de W pour un GEMM 32³ (minimum 256) | 2048 | 256 | `test/weight_buffer/benchmark.py` |
| Part du calcul dans le temps total (hôte compris), GEMM 4³ / 64³ | 50 % / 86 % | 56 % / 86 % | `test/activation_buffer/benchmark.py` |
| Couche 64 → 48 sur un lot de 16 (WS moins bon quand le lot est petit) | 3652 cycles | 4812 cycles | `test/activation_buffer/benchmark.py` |
| GEMM 64³ piloté par le controller (go → done) | 19 461 cycles | 18 701 cycles | `test/controller/benchmark.py` |
| Utilisation quand K = 4 (M = N = 32) / quand N = 4 (M = K = 32) | 25 % / 72 % | 75 % / 30 % | `test/controller/benchmark.py` |
| Débit de la puce, couche 64³, hôte compris / calcul seul | 11,6 / 13,5 MAC/cycle | 12,0 / 14,0 MAC/cycle | `test/host_interface/benchmark.py` |
| MLP 64 → 32 → 16 flottant quantifié, sur la puce par le bus | corrélation 0,9999 | identique | `test/npu_top/benchmark.py` |
| Couche 64 → 64, cycles par échantillon, lot de 1 / de 64 | 2446 / 353 | 3702 / 341 | `test/npu_top/benchmark.py` |

Les cycles mesurés suivent exactement les formules de `src/controller/interface.md`
(OS : `T(M)·T(N)·(K + 3N)` = 16 · 28 = 448).

### Pistes identifiées, pas encore explorées
- OS : recouvrir calcul et vidage (registre fantôme par PE) — le vidage et la lecture prennent 39 % des cycles d'un GEMM 16³.
- WS : double tampon de poids pour recouvrir le préchargement (16 % des cycles d'un GEMM 16³).
- Accumulateur réduit (22 bits suffisent pour K ≤ 64).
- Ordre de boucles WS k_t → i_t → j, enable par PE, double buffering des SRAM.
- DMA (v2) : le chargement par l'hôte pèse jusqu'à la moitié du temps sur les petits GEMM.
- Bit d'échange des rôles IN / OUT : supprimerait la recopie de Y entre deux couches.

---

## 4. Journal des décisions

| Date | Décision | Choix | Justification |
|---|---|---|---|
| 2026-10-18 | Découpage en blocs | 11 blocs (section 2) | Correspondance 1:1 avec des modules Verilog |
| 2026-10-18 | D1 Cible | Simulation → FPGA → Tiny Tapeout | Apprendre d'abord, puis confronter aux contraintes réelles |
| 2026-10-18 | D2 Workload | GEMM générique | Noyau de tous les réseaux, périmètre maîtrisé |
| 2026-10-18 | D3 Dataflow | OS + WS comparables | Explorer le compromis réutilisation des poids / simplicité |
| 2026-10-18 | D4 Grille | 4×4 par défaut, paramétrable | Lisible en simulation, tiling non trivial |
| 2026-10-18 | D5 Format | int8 signé → acc int32 | Standard de l'inférence quantifiée |
| 2026-10-18 | D9 Contrôle | FSM câblée | Simplicité ; la mini-ISA reste une évolution possible |
| 2026-10-18 | D6 Mémoire | Buffers W/X séparés, simple buffer | Lectures parallèles sans conflit, contrôle simple |
| 2026-10-18 | D8 DMA | Aucun en v1, l'hôte écrit directement | Concentrer l'effort sur le cœur de calcul |
| 2026-10-18 | D10 Post-traitement | Biais → ReLU → requant int8 | Sortie réutilisable comme entrée de la couche suivante |
| 2026-10-18 | D12 Protocole | Contrôle centralisé | Déterministe, typique d'un systolic array |
| 2026-10-18 | D6 Tailles | Matrices jusqu'à 64×64 | Tiling entièrement interne, taille raisonnable en simulation |
| 2026-10-18 | D7 Interface | Bus parallèle simple | Simple en simulation, adaptable vers Wishbone ou SPI |
| 2026-10-18 | D11 Horloge/reset | 1 clk, rst_n synchrone | Convention Tiny Tapeout, pas de CDC |
| 2026-10-18 | D6 Layout | Tuilé, propre à chaque variante | 1 mot = 1 lecture de la grille, adresses par concaténation, Y au format X |
| 2026-10-18 | D10 Requant | M 16 b, s 5 b, arrondi, par couche, symétrique | Bonne précision pour un multiplieur 32×16 |
| 2026-10-18 | D7 Carte mémoire | Bus 32 b, 13 bits d'adresse (section D7) | Registres + fenêtres directes sur les buffers |
| 2026-10-18 | D9 Boucles | OS i→j→k, WS i→k→j | Accumulateurs WS limités à 1 Kio |
| 2026-10-18 | Bords | Masquage en lecture (0) et en écriture | Transparent pour l'hôte, quelques comparateurs |
| 2026-10-18 | pe OS | `clear` diffusé à tous les PE, prioritaire | Remise à zéro simple, sans signal qui suit les données |
| 2026-10-18 | pe WS | Poids préchargés par décalage vertical (`load`) | Pas de bus d'écriture vers chaque PE |
| 2026-10-18 | OS : vidage | Pas de recouvrement calcul / vidage, période K + 3N | Simplicité ; un registre fantôme par PE permettrait de recouvrir (à explorer) |
| 2026-10-18 | WS : flush | 2N − 3 cycles avant le préchargement suivant, période N_dim + 3N − 3 | Limite exacte, **confirmée en simulation** : avec un cycle de moins, la dernière colonne est fausse (`test/pe_array/test.py`) |
| 2026-10-18 | data_feeder | Il aligne les commandes (+1 cycle), masque et décale | Le `controller` émet tout au même cycle d'émission |
| 2026-10-18 | Mémoires | SRAM simple port, lecture synchrone L = 1, choix du port par `busy` | Réaliste pour un ASIC ou une BRAM ; deux bancs IN / OUT pour les activations |
| 2026-10-18 | Tag | 20 bits `{y_addr, lane_mask, row}` qui voyagent avec les résultats | L'écriture de Y n'a pas besoin du `controller` |
| 2026-10-18 | output_drain WS | Redressement + ligne à retard de 2N cycles + mémoire d'accumulateurs à lecture combinatoire | Lecture-modification-écriture en un cycle (simple en simulation, à pipeliner pour un ASIC) |
| 2026-10-18 | post_processing | Pipeline de 3 étages, N voies, biais en N bancs | Débit d'une entrée par cycle ; un biais par voie en WS, un seul diffusé en OS |
| 2026-10-18 | host_interface | Latence de lecture 1 partout, reset REQ_MULT = 1, `irq = DONE` | Comportement uniforme ; requantification neutre par défaut |
| 2026-10-18 | Dimensions invalides | 0 ou > 64 : `done` immédiat, rien n'est calculé | Comportement défini plutôt que non défini |
| 2026-10-18 | Implémentation | py4hw comportemental (`clock()` / `propagate()`), une classe par variante | Code lisible, proche de la spécification ; la transcription en Verilog se fait à la main |
| 2026-10-18 | output_drain OS | N passé en paramètre explicite | N ne se déduit pas de la largeur de `drain_row` quand N n'est pas une puissance de 2 |
| 2026-10-18 | controller | Machine de Moore ; `busy` = 0 pendant DONE ; `cycles` = cycles hors IDLE depuis `go`, DONE compris | Pendant DONE la dernière écriture de Y est terminée : l'hôte peut relire dès `done` |
| 2026-10-18 | controller | Sorties publiées dans `clock()`, pas de `propagate()` | Évite une fausse boucle combinatoire host_interface ⇄ controller dans le simulateur |
| 2026-10-18 | host_interface | « Occupé » = `busy` OU impulsion `go` en cours ; accès mémoire et configuration ignorés quand occupé | Sans cela, un accès juste après GO passait avant le début du calcul (trouvé par les tests de npu_top) |

---

## 5. Conventions d'interface

Ces règles s'appliquent à tous les fichiers `src/<bloc>/interface.md`.

### 5.1 Paramètres communs
| Nom | Défaut | Définition | Rôle |
|---|---|---|---|
| `DATAFLOW` | OS | OS ou WS | variante du cœur de calcul, fixée à la construction (ce n'est pas un registre) |
| `N` | 4 | — | côté de la grille ; nombre de voies |
| `DW` | 8 | — | largeur d'une donnée (int8 signé) |
| `AW` | 32 | — | largeur d'un accumulateur ou d'une somme partielle (int32 signé) |
| `DIM_MAX` | 64 | — | dimension maximale de M, K et N |
| `IW` | 6 | log2(DIM_MAX) | largeur d'un indice d'élément (i, j, k) |
| `TW` | 4 | log2(DIM_MAX / N) | largeur d'un indice de tuile (i_t, j_t, k_t) |
| `BAW` | 10 | TW + IW | largeur d'une adresse de mot dans un buffer |
| `DIMW` | 7 | log2(DIM_MAX) + 1 | largeur des registres de dimensions (de 1 à 64) |
| `TAG_W` | 20 | BAW + N + IW | largeur du tag |

Les blocs du cœur de calcul restent paramétrables en N. La v1 de `host_interface` et de `npu_top`
suppose N = 4, puisqu'un mot de buffer fait alors 32 bits, la largeur du bus.

### 5.2 Nommage et conventions de signaux
- Les signaux sont en `snake_case`. Ils sont actifs à 1, sauf `rst_n`.
- Tous les blocs séquentiels reçoivent `clk` et `rst_n` (reset synchrone actif bas, D11).
  Le contenu des mémoires n'est pas réinitialisé.
- Les données sont en complément à deux : int8 pour les données, int32 pour les accumulateurs.
- Suffixes : `_addr` (adresse), `_we` / `_re` (autorisation d'écriture / de lecture), `_wdata` / `_rdata` (données),
  `_mask` (un bit par voie), `_valid` (la donnée qui accompagne ce bit est utilisable).
- Côté grille, les ports sont nommés par point cardinal : `_west`, `_north`, `_south`.
- **Rangement des voies** : un vecteur de N éléments de W bits est empaqueté, la voie `l` occupant les bits
  `[W·l + W−1 : W·l]`. La voie 0 est donc dans les bits de poids faible. C'est aussi l'ordre des octets dans un mot mémoire (D6).
- Une matrice de N×N éléments (`acc` en OS) est rangée ligne par ligne : l'élément (i, j) est à l'indice `i·N + j`.

### 5.3 Latence et cycle d'émission
- **Latence L = n** : une entrée présente pendant le cycle t produit son effet en sortie pendant le cycle t + n.
  L = 0 désigne un chemin combinatoire.
- Une écriture demandée pendant le cycle t est effective au front de fin de ce cycle : la donnée est lisible dès t + 1.
- **Mémoires** : la lecture est synchrone, avec L = 1. `rdata` est valide au cycle qui suit `re`.
- **Cycle d'émission** : c'est le cycle où le `controller` présente une adresse SRAM et toutes les commandes qui vont
  avec (masques, `ctl`, sideband, tag). Ce cycle sert de référence à tous les chronogrammes.
  Chaque bloc placé après la SRAM retarde ses commandes d'autant que les données : le `data_feeder` ajoute 1 cycle,
  la ligne à retard de `output_drain` (WS) ajoute 2N cycles.
- **Contrôle centralisé** (D12) : il n'y a pas de handshake. Chaque latence est fixe et connue du `controller`.

### 5.4 Tag
Le tag accompagne chaque groupe de N résultats, de `output_drain` jusqu'à l'écriture dans `activation_buffer`.

| Bits | Champ | Largeur | Contenu |
|---|---|---|---|
| [19:10] | `y_addr` | BAW | adresse du mot Y dans le banc OUT (OS : `{j_t, i}` · WS : `{i_t, j}`) |
| [9:6] | `lane_mask` | N | autorisation d'écriture de chaque voie (0 hors limites) |
| [5:0] | `row` | IW | indice de ligne pour le biais (OS : i · WS : 4·i_t) |

### 5.5 Récapitulatif des latences
| Chemin | Latence |
|---|---|
| Émission → voie l à l'entrée de la grille (flux décalé) | 1 + l |
| Émission → `w_north` en WS (préchargement) | 1 |
| `pe` : entrée → sortie | 1 |
| `output_drain` OS : `drain_en` → `res_valid` | 1 |
| `output_drain` WS : émission de X → `res_valid` | 2N + 1 |
| `post_processing` : `res_valid` → `y_we` | 3 |
| Bus hôte : `bus_re` → `bus_rdata` | 1 |
