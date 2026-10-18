# post_processing — Biais, ReLU, requantification

Pipeline de 3 étages qui traite **N voies en parallèle** et accepte une nouvelle entrée à chaque cycle.
Le bloc contient la mémoire des biais et écrit directement dans le banc OUT de `activation_buffer`.

## Paramètres
`N` = 4, `AW` = 32, `DW` = 8, `IW` = 6, `BAW` = 10, `TAG_W` = 20, `DATAFLOW`

## Ports
| Signal | Dir | Largeur | Description |
|---|---|---|---|
| `clk`, `rst_n` | in | 1 | |
| `busy` | in | 1 | 0 : le port hôte des biais est actif |
| `res` | in | N·AW | depuis `output_drain` |
| `res_valid` | in | 1 | |
| `tag_in` | in | TAG_W | `{y_addr, lane_mask, row}` |
| `relu_en` | in | 1 | registre `CTRL.RELU_EN` |
| `req_mult` | in | 16 | registre `REQ_MULT`, non signé |
| `req_shift` | in | 5 | registre `REQ_SHIFT` |
| `bias_host_addr` | in | IW | indice du biais (0..63) |
| `bias_host_wdata` | in | AW | |
| `bias_host_we` | in | 1 | |
| `bias_host_re` | in | 1 | |
| `bias_host_rdata` | out | AW | latence 1 |
| `y_addr` | out | BAW | vers `activation_buffer.core_wr_addr` |
| `y_wdata` | out | N·DW | vers `core_wr_wdata` |
| `y_we` | out | 1 | vers `core_wr_we` |
| `y_wmask` | out | N | vers `core_wr_wmask` |

## Mémoire des biais
- 64 × int32, rangés en **N bancs de 16 entrées** : le biais b est dans le banc `b mod N`, à l'adresse `b div N`.
- La lecture est combinatoire. L'indice vient de `row[IW−1:2]`, et on lit les 4 bancs à la fois.
- Choix du biais de la voie l :
  - **OS** : les 4 voies appartiennent à la même ligne de Y, donc elles reçoivent toutes `banc[row mod N]`.
  - **WS** : les 4 voies correspondent à 4 lignes consécutives (`row` est un multiple de 4), donc la voie l reçoit `banc[l]`.

## Pipeline (par voie l)
| Étage | Calcul | Largeur du résultat |
|---|---|---|
| S1 | `a = res_l + bias_l` | 32 bits signé (boucle modulo 2^32) |
| S2 | `r = relu_en ? max(0, a) : a` ; `p = r × req_mult` | 48 bits signé |
| S3 | `q = (p + (s > 0 ? 2^(s−1) : 0)) >>> s` ; `y_l = sat(q, −128, 127)` | 8 bits signé |

Après S3, les sorties sont enregistrées : `y_we = valid`, `y_addr = tag.y_addr`, `y_wmask = tag.lane_mask`.
Le bit `valid` et le tag avancent dans le pipeline en même temps que les données.

## Latences
`res_valid` présent au cycle t donne `y_we` au cycle **t + 3**, et le mot est écrit au front qui suit.

## Coût (N = 4)
4 additionneurs 32 bits, 4 multiplieurs 32×16, 4 additionneurs d'arrondi 48 bits, 4 saturateurs et 2 Kibit de biais.
Compromis à mesurer : une version à 1 voie (4 fois moins de multiplieurs) ralentirait le vidage d'un facteur 4.

---

## Implémentation

- Fichier : [`post_processing.py`](post_processing.py) — classes `PostProcessing_OS` (un biais pour les N voies) et
  `PostProcessing_WS` (un biais par voie), sur une base commune.
- Les 3 étages sont des registres mis à jour dans `clock()` ; le registre de sortie de S3 pilote directement
  `y_we`, `y_addr`, `y_wmask` (forcé à 0 quand `y_we` = 0) et `y_wdata`.
- La mémoire des biais est modélisée par une liste de 64 entrées : lire `biais[row]` (OS) ou `biais[row + l]` (WS)
  revient exactement à lire les N bancs décrits plus haut. Elle n'est pas réinitialisée par `rst_n`.
- `relu_en` et `req_mult` sont lus à l'étage S2, `req_shift` à l'étage S3 : ils doivent rester constants pendant un GEMM.
- La largeur de M se déduit de celle du fil `req_mult` (16 bits en v1), ce qui permet au benchmark de l'explorer.

| Constructeur | Ports, dans l'ordre |
|---|---|
| `PostProcessing_OS` / `PostProcessing_WS(parent, name, …)` | `rst_n, busy, res, res_valid, tag_in, relu_en, req_mult, req_shift, bias_host_addr, bias_host_wdata, bias_host_we, bias_host_re, bias_host_rdata, y_addr, y_wdata, y_we, y_wmask` |

**Vérification** ([`test/post_processing/`](../../test/post_processing/)) : latence exacte de 3 cycles et débit d'une
entrée par cycle, choix du biais selon la variante, ReLU, arrondi (y compris les demis négatifs), saturation et
valeurs extrêmes (M = 65 535, s = 31), débordement modulo 2^32 de l'addition du biais, port hôte (latence 1, inactif
pendant `busy`), modèle de référence sur 1200 cycles. **De bout en bout** : la chaîne `data_feeder` → `pe_array` →
`output_drain` → `post_processing` calcule `requant(ReLU(W·X + b))` ; Y int8 est reconstruit à partir des seules
écritures, et OS et WS écrivent exactement le même Y.

**Mesuré** : avec M sur 16 bits, la requantification est identique à l'idéal `round(x · échelle)` (dès 14 bits) ;
avec 8 bits, l'erreur atteint 1 pas int8. Un MLP 64 → 32 → 16 exécuté sur le NPU simulé reproduit le réseau en
virgule flottante avec une corrélation de 0,9998.

