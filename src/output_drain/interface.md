# output_drain — Collecte des résultats de la grille

Le bloc récupère les résultats int32 de la grille et les présente à `post_processing` sous une forme commune
aux deux variantes : **N voies int32 + `res_valid` + tag**.
Le tag (adresse de Y, masque d'écriture, ligne) est défini en [architecture.md §5](../architecture.md#5-conventions-dinterface).

## Paramètres
`N` = 4, `AW` = 32, `IW` = 6, `TAG_W` = 20, `DATAFLOW`

N est un paramètre explicite des deux variantes : il ne se déduit pas de la largeur de `drain_row`
quand N n'est pas une puissance de 2.

## Sortie commune
| Signal | Dir | Largeur | Description |
|---|---|---|---|
| `res` | out | N·AW | N résultats int32 |
| `res_valid` | out | 1 | `res` est valide |
| `tag_out` | out | TAG_W | métadonnées qui accompagnent `res` |

---

## Variante OS : vidage des accumulateurs

### Ports d'entrée
| Signal | Dir | Largeur | Description |
|---|---|---|---|
| `clk`, `rst_n` | in | 1 | |
| `acc` | in | N·N·AW | depuis `pe_array` |
| `drain_en` | in | 1 | depuis le `controller` |
| `drain_row` | in | log2 N | ligne r de la tuile à lire |
| `tag_in` | in | TAG_W | depuis le `controller` |

### Comportement
```
res       ← acc[drain_row][0..N−1]      (voie j = PE(drain_row, j))
res_valid ← drain_en
tag_out   ← tag_in
```
- Latence : L = 1. Vider une tuile prend N cycles (une ligne par cycle).
- Coût : un multiplexeur N vers 1 sur N·AW bits, plus N·AW + 1 + TAG_W = 149 bascules.

---

## Variante WS : redressement et accumulation sur les tuiles de K

### Ports d'entrée
| Signal | Dir | Largeur | Description |
|---|---|---|---|
| `clk`, `rst_n` | in | 1 | |
| `psum_south` | in | N·AW | depuis `pe_array`, la voie i est en retard de i cycles |
| `s_valid` | in | 1 | sideband du `controller`, au cycle d'émission de X |
| `s_j` | in | IW | colonne j, qui sert d'adresse dans la mémoire d'accumulateurs |
| `s_first` | in | 1 | première tuile de K (k_t = 0) : on écrase au lieu d'additionner |
| `s_last` | in | 1 | dernière tuile de K : on émet le résultat |
| `tag_in` | in | TAG_W | |

### Structure interne
1. **Redressement** : la voie i passe par N−1−i registres. Les N voies de la colonne j sont alors
   alignées au cycle `e + 2N`, où e est le cycle d'émission de X pour cette colonne.
2. **Ligne à retard du sideband** : `s_valid`, `s_j`, `s_first`, `s_last` et `tag_in` sont retardés de **2N cycles**
   pour arriver en même temps que les données redressées.
3. **Mémoire d'accumulateurs** : 2^IW = 64 entrées × N·AW = 128 bits (1 Kio). Lecture combinatoire, écriture synchrone.

### Comportement (au cycle aligné, si s_valid retardé)
```
somme = s_first ? psum_aligné : accmem[s_j] + psum_aligné      (addition voie par voie)
si !s_last : accmem[s_j] ← somme
si  s_last : res ← somme ; res_valid ← 1 ; tag_out ← tag retardé
```
- Latence : `res_valid` arrive au cycle `e + 2N + 1`.
- Coût (N = 4) : 6 × 32 bits de redressement, 8 × 29 bits de ligne à retard, 1 Kio de mémoire et N additionneurs 32 bits.
- Optimisation possible : reconstruire le tag localement (il ne dépend que de i_t et de j) au lieu de le retarder.

---

## Implémentation

- Fichier : [`output_drain.py`](output_drain.py) — classes `OutputDrain_OS` et `OutputDrain_WS`.
- Toutes les sorties sont des registres (`clock()` uniquement). En WS, la mémoire d'accumulateurs (`accmem`) n'est pas
  réinitialisée par `rst_n` ; les registres de redressement, la ligne à retard et `res` le sont.

| Constructeur | Ports, dans l'ordre |
|---|---|
| `OutputDrain_OS(parent, name, …, N=4)` | `rst_n, acc, drain_en, drain_row, tag_in, res, res_valid, tag_out` |
| `OutputDrain_WS(parent, name, …, N=4)` | `rst_n, psum_south, s_valid, s_j, s_first, s_last, tag_in, res, res_valid, tag_out` |

**Vérification** ([`test/output_drain/`](../../test/output_drain/)) : vidage ligne par ligne, redressement et latence exacte
de 2N + 1 pour N = 1, 2, 4, 8, accumulation sur trois tuiles de K, cas d'une seule tuile, modèles de référence cycle par
cycle. **De bout en bout** : le cœur de calcul complet (`data_feeder` → `pe_array` → `output_drain`), piloté par un
planificateur Python qui reproduit la FSM du `controller`, calcule des GEMM jusqu'à 10×7×9 ; Y est reconstruit
uniquement à partir des (tag, `res`) sortis, et chaque élément doit être écrit exactement une fois.

**Mesuré** : sur un GEMM 16×16×16, les résultats sortent par rafales de N lignes en OS (448 cycles) et en flux continu
pendant la dernière tuile de K en WS (403 cycles).
