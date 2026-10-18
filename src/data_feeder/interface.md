# data_feeder — Alignement, masquage et décalage des flux

Le bloc se place entre les buffers et la grille. Il a trois rôles :
1. **Aligner** : il retarde d'un cycle les commandes émises par le `controller`, pour qu'elles arrivent en même temps
   que les données de la SRAM (latence de lecture = 1).
2. **Masquer** : il force à 0 les voies hors limites (D-Bords).
3. **Décaler** : il retarde la voie l de l cycles (escalier systolique).

Les ports sont les mêmes pour les deux variantes. Seul le routage interne change.

## Paramètres
`N` = 4, `DW` = 8, `DATAFLOW` ∈ {OS, WS}

## Ports
| Signal | Dir | Largeur | Description |
|---|---|---|---|
| `clk`, `rst_n` | in | 1 | |
| `w_rdata` | in | N·DW | mot lu dans `weight_buffer` (`core_rdata`) |
| `x_rdata` | in | N·DW | mot lu dans `activation_buffer`, zone IN (`core_rd_rdata`) |
| `w_mask` | in | N | voies W valides, au cycle d'émission |
| `x_mask` | in | N | voies X valides, au cycle d'émission |
| `ctl_in` | in | 1 | OS : `clear` · WS : `load`, au cycle d'émission |
| `w_arr` | out | N·DW | OS → `w_west` · WS → `w_north` |
| `x_arr` | out | N·DW | OS → `x_north` · WS → `x_west` |
| `ctl_arr` | out | 1 | OS → `clear` · WS → `load` |

## Comportement
```
étage d'alignement (registres) : w_mask_r ← w_mask ; x_mask_r ← x_mask ; ctl_arr ← ctl_in
masquage (combinatoire)        : w_m[l] = w_mask_r[l] ? w_rdata[l] : 0      (idem pour x)
décalage                       : voie l → chaîne de l registres de DW bits (remis à 0 par rst_n)
```

| Flux | OS | WS |
|---|---|---|
| W | masqué puis **décalé** | masqué, **non décalé** (le préchargement est diffusé) |
| X | masqué puis **décalé** | masqué puis **décalé** |

- Quand rien n'est lu, le `controller` met les masques à 0, et le bloc injecte donc des zéros.

## Latences (depuis le cycle d'émission)
| Sortie | L |
|---|---|
| voie l d'un flux décalé | 1 + l |
| `w_arr` en WS | 1 |
| `ctl_arr` | 1 |

## Coût (N = 4)
- Décalage : N(N−1)/2 = 6 registres de 8 bits par flux décalé, soit 96 bascules en OS et 48 en WS.
- Alignement : 2N + 1 = 9 bascules.

---

## Implémentation

- Fichier : [`data_feeder.py`](data_feeder.py) — classes `DataFeeder_OS` (W et X décalés) et `DataFeeder_WS`
  (W non décalé), sur une base commune.
- `clock()` : registres d'alignement (`w_mask_r`, `x_mask_r`, `ctl_r`) et chaînes de décalage ;
  `propagate()` : masquage et sorties. La voie 0 n'a aucun registre de décalage : elle est **combinatoire**
  depuis `rdata`, d'où la latence 1 + l mesurée depuis l'émission (la SRAM apporte le premier cycle).
- `N` se déduit de la largeur de `w_mask`, `DW` de celle de `w_rdata`.

| Constructeur | Ports, dans l'ordre |
|---|---|
| `DataFeeder_OS` / `DataFeeder_WS(parent, name, …)` | `rst_n, w_rdata, x_rdata, w_mask, x_mask, ctl_in, w_arr, x_arr, ctl_arr` |

**Vérification** ([`test/data_feeder/`](../../test/data_feeder/)) : latence de chaque voie pour N = 1, 2, 4, 8, masquage,
zéros injectés au repos, modèle de référence sur 1000 cycles aléatoires, et chaîne `data_feeder` → `pe_array` sur des GEMM
aux dimensions non multiples de N avec des valeurs non nulles hors limites en mémoire.

**Mesuré** : la réutilisation des données est 2 fois meilleure en WS (3,8 MAC par octet lu contre 2,0 en OS pour N = 4)
et le débit SRAM nécessaire 2 fois plus faible ; le masquage des bords gaspille en moyenne 8 % des voies pour N = 4,
15 % pour N = 8.
