# controller — Séquenceur du GEMM

FSM câblée qui parcourt les tuiles et émet, **au cycle d'émission**, les adresses SRAM, les masques, les commandes
de la grille et les tags. Elle connaît toutes les latences en aval (contrôle centralisé, D12).

## Paramètres
`N` = 4, `DIM_MAX` = 64, `IW` = 6, `TW` = 4, `BAW` = 10, `TAG_W` = 20, `DATAFLOW`

## Ports
| Signal | Dir | Largeur | Description |
|---|---|---|---|
| `clk`, `rst_n` | in | 1 | |
| `go` | in | 1 | impulsion envoyée par `host_interface` |
| `dim_m`, `dim_k`, `dim_n` | in | 7 chacun | dimensions, de 1 à 64 |
| `busy` | out | 1 | un GEMM est en cours ; sert aussi à choisir le port actif des mémoires |
| `done` | out | 1 | impulsion d'un cycle en fin de GEMM |
| `cycles` | out | 32 | nombre de cycles du GEMM en cours ou du dernier GEMM (remis à 0 par `go`) |
| `w_addr`, `w_re` | out | BAW, 1 | vers `weight_buffer` (port cœur) |
| `x_addr`, `x_re` | out | BAW, 1 | vers `activation_buffer` (lecture du banc IN) |
| `w_mask`, `x_mask` | out | N, N | vers `data_feeder` |
| `ctl` | out | 1 | vers `data_feeder.ctl_in` (OS : clear · WS : load) |
| `tag` | out | TAG_W | OS : vers `output_drain` au moment du vidage · WS : sideband au moment du flux |
| `drain_en`, `drain_row` | out | 1, log2 N | **OS seulement**, vers `output_drain` |
| `s_valid`, `s_j`, `s_first`, `s_last` | out | 1, IW, 1, 1 | **WS seulement**, vers `output_drain` |

## Règles générales
- Un `go` avec une dimension à 0 ou supérieure à 64 lève `done` au cycle suivant, sans rien calculer.
- Nombre de tuiles : `T(D) = ⌈D/N⌉ = (D + N − 1) >> 2`.
- Masque de la voie l pour la tuile t de la dimension D : `4·t + l < D`.

## FSM OS — boucles i_t → j_t → k

| État | Durée | Ce qui est émis |
|---|---|---|
| `IDLE` | — | rien ; on attend `go` |
| `CLEAR` | 1 | `ctl = 1` |
| `STREAM` | K | pour chaque k : `w_addr = {i_t, k}`, `x_addr = {j_t, k}`, `w_re = x_re = 1`, masques de i_t (W) et de j_t (X) |
| `FLUSH` | 2N − 1 | masques à 0 : la grille termine son calcul |
| `DRAIN` | N | pour r = 0..N−1 : `drain_en = 1`, `drain_row = r`, `tag = {y_addr = {j_t, i}, lane_mask = masque(j_t) si i < M sinon 0, row = i}` avec `i = 4·i_t + r` |
| → tuile suivante (`CLEAR`), ou `TAIL` | | |
| `TAIL` | 4 | attente de la dernière écriture de `post_processing` |
| `DONE` | 1 | `done = 1`, puis retour à `IDLE` |

**Chronogramme d'une tuile** (`CLEAR` émis au cycle c0) :
- la grille est remise à zéro pendant c0 + 1, et le terme k est émis à c0 + 1 + k ;
- PE(i,j) calcule le terme k pendant le cycle `c0 + 2 + i + j + k` ;
- le dernier terme est calculé pendant c0 + K + 2N − 1, et les accumulateurs sont lisibles à partir de c0 + K + 2N ;
- le vidage occupe les cycles c0 + K + 2N à c0 + K + 3N − 1.

**Période d'une tuile : `K + 3N` cycles.**

## FSM WS — boucles i_t → k_t → j

| État | Durée | Ce qui est émis |
|---|---|---|
| `IDLE` | — | on attend `go` |
| `PRELOAD` | N | pour n = 0..N−1 : `w_addr = {i_t, 4·k_t + N−1−n}`, `w_re = 1`, `ctl = 1`, `w_mask = masque(i_t)` si `4·k_t + N−1−n < K`, sinon 0 |
| `STREAM` | N_dim | pour chaque j : `x_addr = {k_t, j}`, `x_re = 1`, `x_mask = masque_K(k_t)`, `s_valid = 1`, `s_j = j`, `s_first = (k_t = 0)`, `s_last = (k_t = T(K)−1)`, `tag = {y_addr = {i_t, j}, lane_mask = masque(i_t), row = 4·i_t}` |
| `FLUSH` | 2N − 3 | les X déjà injectés finissent de traverser la grille avant qu'on écrase les poids |
| → (i_t, k_t) suivant (`PRELOAD`), ou `TAIL` | | |
| `TAIL` | 2N + 4 | attente de la dernière écriture |
| `DONE` | 1 | `done = 1` |

**Pourquoi FLUSH dure 2N − 3 cycles** : soit s le cycle où le dernier X est émis. PE(N−1,N−1) l'utilise pendant s + 2N − 1.
Le premier `load` du préchargement suivant atteint la grille au cycle p' + 1, et ne modifie les poids qu'à la fin de ce cycle.
Il suffit donc que p' + 1 ≥ s + 2N − 1, c'est-à-dire p' = s + 2N − 2, ce qui laisse 2N − 3 cycles de FLUSH.
*Cette limite est exacte, sans marge. Elle est **confirmée en simulation** par `test/pe_array/test.py`
(`test_ws_ecart_minimal_avant_rechargement`) : avec un cycle de moins, la dernière colonne de Y est fausse.*

**Période d'un couple (i_t, k_t) : `N_dim + 3N − 3` cycles.**

## Performances attendues (N = 4, GEMM 64×64×64, idéal = 64³/16 = 16 384 cycles)
| | Formule (sans TAIL) | Cycles | Utilisation |
|---|---|---|---|
| OS | `T(M)·T(N_dim)·(K + 3N)` | 16·16·76 = 19 456 | 84 % |
| WS | `T(M)·T(K)·(N_dim + 3N − 3)` | 16·16·73 = 18 688 | 88 % |

Quand K est petit (par exemple 4), l'utilisation en OS tombe à 25 %. Quand N_dim est petit, c'est WS qui s'effondre.
`benchmark.py` doit faire apparaître cette asymétrie.

**Vérifié en simulation** (`test/output_drain/benchmark.py`, cœur de calcul complet) : GEMM 16×16×16 en OS = 448 cycles,
exactement `T(M)·T(N)·(K + 3N)` = 16 · 28 ; en WS, dernier résultat au cycle 403 (400 + latence de sortie).

## Coût
Compteurs i_t, j_t ou k_t (TW bits), k ou j (IW bits), compteur de phase, comparateurs de masque (2N), compteur `cycles` (32 bits).

---

## Implémentation

- Fichier : [`controller.py`](controller.py) — classes `Controller_OS` et `Controller_WS`, machine de Moore :
  `clock()` met à jour l'état, les compteurs de boucles (`t_out` = i_t, `t_mid` = j_t ou k_t) et le compteur de
  position dans l'état (`cnt`), puis publie toutes les sorties décodées à partir du nouvel état. Le bloc n'a pas de
  `propagate()` : cela évite au simulateur une fausse boucle combinatoire avec le `host_interface`.
- Les dimensions sont mémorisées au `go`. `N` se déduit de la largeur de `w_mask`, `DIM_MAX` de celle de `dim_m`.
- Les états de durée nulle sont sautés (FLUSH en WS quand N = 1).

**Chronologie** (vérifiée par les tests) :
- une impulsion `go` présente au cycle g fait passer la FSM dans son premier état au front suivant :
  la première émission a lieu au cycle g + 1 ;
- `busy` vaut 1 dans tous les états sauf IDLE et DONE ; pendant DONE (un cycle, `done` = 1), la dernière écriture
  de Y est déjà faite et l'hôte peut relire la mémoire ;
- `cycles` compte les cycles hors IDLE depuis le dernier `go`, DONE compris (plan + 1), et garde sa valeur jusqu'au
  `go` suivant ;
- une dimension nulle ou supérieure à `DIM_MAX` mène directement à DONE ; un `go` reçu pendant un GEMM est ignoré.

| Constructeur | Ports, dans l'ordre |
|---|---|
| `Controller_OS(parent, name, …)` | `rst_n, go, dim_m, dim_k, dim_n, busy, done, cycles, w_addr, w_re, x_addr, x_re, w_mask, x_mask, ctl, tag, drain_en, drain_row` |
| `Controller_WS(parent, name, …)` | `rst_n, go, dim_m, dim_k, dim_n, busy, done, cycles, w_addr, w_re, x_addr, x_re, w_mask, x_mask, ctl, tag, s_valid, s_j, s_first, s_last` |

**Vérification** ([`test/controller/`](../../test/controller/)) : pour 7 formes de GEMM (de 1×1×1 à 64×2×5) et N = 2
ou 4, le controller émet **cycle par cycle exactement** ce qu'émet le planificateur de référence (`plan_os` / `plan_ws`
de [`test/output_drain/test.py`](../../test/output_drain/test.py)), lui-même validé de bout en bout ; compteur de cycles,
dimensions invalides, `go` parasites, reset en cours de route, GEMM successifs. **De bout en bout** : le controller pilote
tout le chemin de données avec ses mémoires (tout le NPU sauf `host_interface`), lancé par `go` et attendu sur `done`,
y compris deux couches enchaînées.

**Mesuré** : les cycles mesurés suivent exactement les formules ci-dessus (écart nul de 1×1×1 à 64×64×64). Selon la
forme du GEMM, OS ou WS peut être jusqu'à 3,4 fois plus rapide (carte de décision dans `test/controller/benchmark.py`).
