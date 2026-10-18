# npu_top — Assemblage du NPU

C'est le seul bloc visible de l'extérieur. Il instancie tous les autres et les câble entre eux, sans logique propre.

## Paramètres
| Nom | Défaut | Rôle |
|---|---|---|
| `DATAFLOW` | OS | variante du cœur de calcul (OS ou WS), fixée à la construction |
| `N` | 4 | taille de la grille (la v1 impose N = 4 à cause du bus de 32 bits) |
| `DIM_MAX` | 64 | dimension maximale des matrices |

## Ports externes
| Signal | Dir | Largeur | Description |
|---|---|---|---|
| `clk` | in | 1 | horloge unique |
| `rst_n` | in | 1 | reset synchrone actif bas |
| `bus_addr` | in | 13 | |
| `bus_wdata` | in | 32 | |
| `bus_we` | in | 1 | |
| `bus_re` | in | 1 | |
| `bus_rdata` | out | 32 | latence 1 |
| `irq` | out | 1 | = `STATUS.DONE` |

## Câblage
```
host_interface ──cfg──► controller, post_processing
host_interface ◄─busy, done, cycles── controller
host_interface ──ports hôte──► weight_buffer, activation_buffer, post_processing (biais)
controller.busy ──► weight_buffer.busy, activation_buffer.busy, post_processing.busy

controller.w_addr/w_re ──► weight_buffer.core_*          ──core_rdata──►    data_feeder.w_rdata
controller.x_addr/x_re ──► activation_buffer.core_rd_*   ──core_rd_rdata──► data_feeder.x_rdata
controller.w_mask/x_mask/ctl ──► data_feeder

┌──────────────────── cœur de calcul (variante DATAFLOW) ─────────────────────┐
│ data_feeder.w_arr/x_arr/ctl_arr ──► pe_array                                │
│ pe_array.acc (OS) | psum_south (WS) ──► output_drain                        │
│ controller.drain_* + tag (OS) | s_* + tag (WS) ──► output_drain             │
└─────────────────────────────────────────────────────────────────────────────┘

output_drain.res/res_valid/tag_out ──► post_processing
post_processing.y_* ──► activation_buffer.core_wr_*
```

## Le cœur de calcul
L'ensemble `data_feeder` + `pe_array` + `output_drain` est la frontière de variante (D3).
Ses interfaces externes sont communes aux deux variantes :
- en entrée : `w_rdata`, `x_rdata`, `w_mask`, `x_mask`, `ctl`, `tag`, plus la commande propre à la variante
  (`drain_*` en OS, `s_*` en WS) ;
- en sortie : `res`, `res_valid`, `tag_out`.

Il est testé de bout en bout dans `test/output_drain/` (avec les blocs suivants branchés au fur et à mesure).

## Séquence type côté hôte
1. Écrire W dans `WEIGHT`, X dans `ACT_IN` (au format de la variante, voir `ID.DATAFLOW`) et b dans `BIAS`.
2. Écrire `DIM_M`, `DIM_K`, `DIM_N`, `REQ_MULT`, `REQ_SHIFT`.
3. Écrire `CTRL = GO | RELU_EN`.
4. Attendre `irq`, ou lire `STATUS.DONE` en boucle.
5. Lire Y dans `ACT_OUT`, et `CYCLES` pour la mesure.

---

## Implémentation

- Fichier : [`npu_top.py`](npu_top.py) — classe `NpuTop(parent, name, rst_n, bus_addr, bus_wdata, bus_we, bus_re,
  bus_rdata, irq, dataflow='OS', DIM_MAX=64)`. Bloc structurel : il crée les fils internes et instancie les 9 blocs.
- Les blocs sont accessibles par attribut (`host`, `controller`, `weight_buffer`, `activation_buffer`, `data_feeder`,
  `pe_array`, `output_drain`, `post_processing`), ce qui permet aux bancs de test d'observer la puce.
- Les sorties du `controller` sont publiées dans `clock()` (machine de Moore) : sinon le simulateur verrait une
  fausse boucle combinatoire `host_interface` → `go` → `controller` → `busy` → `host_interface`.

**Vérification** ([`test/npu_top/`](../../test/npu_top/)) : la puce n'est pilotée que par son bus, par un logiciel hôte
de référence (classe `Npu`) qui suit la séquence ci-dessus : registre ID, couches de bout en bout dans les deux
variantes (mémoire « sale », dimensions non multiples de 4), deux couches enchaînées, `CYCLES` et `irq`, dimensions
invalides, accès ignorés pendant un GEMM, requantification par défaut, OS et WS identiques.

**Mesuré** : un MLP 64 → 32 → 16 en virgule flottante, quantifié en int8 et exécuté sur la puce, reproduit le calcul
flottant avec une corrélation de 0,9999 ; OS est plus rapide pour les petits lots, WS le rattrape vers un lot de 64.
