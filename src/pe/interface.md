# pe — Processing Element

Brique de base de la grille. Il existe une variante par dataflow, choisie à la construction par le paramètre `DATAFLOW`.
Conventions communes (paramètres, nommage, latences) : voir [architecture.md §5](../architecture.md#5-conventions-dinterface).

## Paramètres
| Nom | Défaut | Rôle |
|---|---|---|
| `DW` | 8 | largeur d'une donnée (int8 signé) |
| `AW` | 32 | largeur de l'accumulateur et des sommes partielles (int32 signé) |

---

## Variante OS (output-stationary)

```
            x_in
             ↓
 w_in  →  [ acc += w·x ]  →  w_out
             ↓
            x_out          acc (lecture parallèle)
```

### Ports
| Signal | Dir | Largeur | Description |
|---|---|---|---|
| `clk`, `rst_n` | in | 1 | horloge, reset synchrone actif bas |
| `w_in` | in | DW | poids venant de l'ouest |
| `x_in` | in | DW | activation venant du nord |
| `clear` | in | 1 | remise à zéro synchrone de l'accumulateur |
| `w_out` | out | DW | `w_in` retardé d'un cycle, vers l'est |
| `x_out` | out | DW | `x_in` retardé d'un cycle, vers le sud |
| `acc` | out | AW | valeur de l'accumulateur |

### Comportement (à chaque front montant)
```
si   !rst_n : w_r ← 0 ; x_r ← 0 ; acc ← 0
sinon       : w_r ← w_in ; x_r ← x_in
              si clear : acc ← 0                                  (clear est prioritaire, le produit du cycle est perdu)
              sinon    : acc ← acc + sext32(w_in × x_in)          (produit signé 8×8 → 16 bits)
w_out = w_r ; x_out = x_r
```
- Le dépassement de capacité boucle modulo 2^32. Il ne peut pas se produire tant que K ≤ 65 536.
- Pour qu'un cycle sans donnée utile n'ajoute rien, il suffit d'avoir 0 sur `w_in` ou sur `x_in`. C'est le `data_feeder` qui garantit ces zéros.

### Latences
| Chemin | L |
|---|---|
| `w_in` → `w_out`, `x_in` → `x_out` | 1 |
| `w_in`, `x_in`, `clear` → `acc` | 1 |

### Coût
48 bascules (8 + 8 + 32), 1 multiplieur signé 8×8, 1 additionneur 32 bits.
Le chemin critique passe par le multiplieur puis l'additionneur.
Fils vers les voisins : 8 bits vers l'est, 8 bits vers le sud.

---

## Variante WS (weight-stationary)

```
          psum_in   w_load_in
             ↓         ↓
 x_in  →  [ w_r stocké, psum + w·x ]  →  x_out
             ↓         ↓
          psum_out  w_load_out
```

### Ports
| Signal | Dir | Largeur | Description |
|---|---|---|---|
| `clk`, `rst_n` | in | 1 | horloge, reset synchrone actif bas |
| `x_in` | in | DW | activation venant de l'ouest |
| `psum_in` | in | AW | somme partielle venant du nord |
| `w_load_in` | in | DW | poids sur le chemin de préchargement, depuis le nord |
| `load` | in | 1 | décale le chemin de préchargement : `w_r ← w_load_in` |
| `x_out` | out | DW | `x_in` retardé d'un cycle, vers l'est |
| `psum_out` | out | AW | `psum_in + w_r·x_in`, enregistré, vers le sud |
| `w_load_out` | out | DW | poids stocké `w_r`, vers le PE du dessous (chemin de préchargement) |

### Comportement (à chaque front montant)
```
si   !rst_n : w_r ← 0 ; x_r ← 0 ; psum_r ← 0
sinon       : si load : w_r ← w_load_in
              x_r    ← x_in
              psum_r ← psum_in + sext32(w_r × x_in)
x_out = x_r ; psum_out = psum_r ; w_load_out = w_r
```
- Pendant un cycle où `load = 1`, le produit utilise l'ancien poids. Le `controller` garantit qu'aucune donnée utile ne passe à ce moment-là.

### Latences
| Chemin | L |
|---|---|
| `x_in` → `x_out` | 1 |
| `psum_in`, `x_in` → `psum_out` | 1 |
| `w_load_in` → `w_load_out` (si `load`) | 1 |

### Coût
48 bascules (8 + 8 + 32), 1 multiplieur signé 8×8, 1 additionneur 32 bits.
Fils vers les voisins : 8 bits vers l'est, **32 + 8 bits** vers le sud.

---

## Comparaison OS / WS
| | OS | WS |
|---|---|---|
| Bascules par PE | 48 | 48 |
| Fils entre PE | 16 | 48 |
| Commande | `clear` | `load` |
| Lecture du résultat | `acc` en parallèle | flux `psum_out` |

---

## Implémentation

- Fichier : [`pe.py`](pe.py) — classes `PE_OS` et `PE_WS`, en py4hw comportemental.
- Toutes les sorties sont des registres : l'état (`w_r`, `x_r`, `acc_r` ou `psum_r`) est mis à jour dans `clock()`
  et publié avec `prepare()`, qui réduit aussi la valeur modulo 2^AW.
- `DW` et `AW` se déduisent de la largeur des fils connectés (contrainte : AW ≥ 2·DW).

| Constructeur | Ports, dans l'ordre |
|---|---|
| `PE_OS(parent, name, …)` | `rst_n, w_in, x_in, clear, w_out, x_out, acc` |
| `PE_WS(parent, name, …)` | `rst_n, x_in, psum_in, w_load_in, load, x_out, psum_out, w_load_out` |

**Vérification** ([`test/pe/`](../../test/pe/)) : reset, latences, `clear` prioritaire, `load` qui utilise l'ancien poids,
valeurs extrêmes, débordement modulo 2^AW, colonne de 4 PE WS empilés, comparaison cycle par cycle avec un modèle de
référence sur 2000 cycles aléatoires, autres largeurs (int4, int16).

**Mesuré** : toutes les latences valent 1 ; en int8 / int32, aucun débordement possible tant que K ≤ 131 071.
