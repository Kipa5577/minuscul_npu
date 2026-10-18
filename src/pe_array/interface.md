# pe_array — Grille N×N de PE

Le bloc ne contient que de la structure : il instancie N×N `pe` et les relie à leurs voisins.
Il n'a aucune logique propre. Il existe une variante par dataflow.

## Paramètres
| Nom | Défaut | Rôle |
|---|---|---|
| `N` | 4 | côté de la grille |
| `DW` | 8 | largeur d'une donnée |
| `AW` | 32 | largeur des accumulateurs |

Les vecteurs « par voie » sont empaquetés : la voie `l` occupe les bits `[W·l + W−1 : W·l]`.

---

## Variante OS

```
            x_north[0]  x_north[1] ...
                ↓           ↓
w_west[0] → PE(0,0) →  PE(0,1) → ...
                ↓           ↓
w_west[1] → PE(1,0) →  PE(1,1) → ...
```

### Ports
| Signal | Dir | Largeur | Description |
|---|---|---|---|
| `clk`, `rst_n` | in | 1 | |
| `w_west` | in | N·DW | voie i → `w_in` de PE(i,0) |
| `x_north` | in | N·DW | voie j → `x_in` de PE(0,j) |
| `clear` | in | 1 | diffusé à tous les PE |
| `acc` | out | N·N·AW | accumulateur de PE(i,j) à l'indice `i·N + j` |

### Câblage interne
- `PE(i,j).w_in = PE(i,j−1).w_out`, et `PE(i,j).x_in = PE(i−1,j).x_out`.
- Les sorties est et sud des PE du bord ne sont pas connectées.

### Latences et contrat d'entrée
- Une valeur présentée sur la voie i de `w_west` au cycle t atteint PE(i,j) au cycle t + j.
- Une valeur présentée sur la voie j de `x_north` au cycle t atteint PE(i,j) au cycle t + i.
- **Contrat** : si `W[i][k]` est présenté sur la voie i au cycle `t0 + i + k`, et `X[k][j]` sur la voie j au cycle `t0 + j + k`,
  alors PE(i,j) calcule le terme k pendant le cycle `t0 + i + j + k`.
- Après un `clear`, `acc(i,j) = Σ_k W[i][k]·X[k][j]` est lisible **à partir du cycle `t0 + K + 2(N−1)`**.

### Coût (N = 4)
16 PE, soit 768 bascules, 16 multiplieurs et 512 fils de sortie (`acc`).

---

## Variante WS

```
              w_north[0]  w_north[1] ...   (préchargement)
psum = 0         ↓           ↓
x_west[0] →  PE(0,0) →  PE(0,1) → ...
                 ↓           ↓
x_west[1] →  PE(1,0) →  PE(1,1) → ...
                 ↓           ↓
            psum_south[0] psum_south[1]
```
PE(k,i) contient `W[i][k]` : la ligne k de la grille correspond à l'entrée k, la colonne i à la sortie i.

### Ports
| Signal | Dir | Largeur | Description |
|---|---|---|---|
| `clk`, `rst_n` | in | 1 | |
| `x_west` | in | N·DW | voie k → `x_in` de PE(k,0) |
| `w_north` | in | N·DW | voie i → `w_load_in` de PE(0,i) |
| `load` | in | 1 | diffusé à tous les PE |
| `psum_south` | out | N·AW | voie i = `psum_out` de PE(N−1,i) |

### Câblage interne
- `PE(k,i).x_in = PE(k,i−1).x_out`
- `PE(k,i).psum_in = PE(k−1,i).psum_out`, et 0 pour la ligne k = 0
- `PE(k,i).w_load_in = PE(k−1,i).w_load_out`, et `w_north[i]` pour la ligne k = 0

### Préchargement
- Il dure N cycles avec `load = 1`. La valeur présentée au n-ième cycle (n = 0..N−1) finit dans la ligne `N−1−n`.
- Il faut donc présenter **en premier la ligne de poids qui ira en bas**, c'est-à-dire la plus grande valeur de k.

### Latences et contrat d'entrée
- **Contrat** : si `X[k][j]` est présenté sur la voie k au cycle `t0 + j + k`, alors la voie i de `psum_south` porte
  `Σ_k W[i][k]·X[k][j]` (somme sur les N lignes de la tuile) **au cycle `t0 + j + i + N`**.
- La sortie est donc décalée en escalier (la voie i est en retard de i cycles), et c'est `output_drain` qui la remet d'équerre.

### Coût (N = 4)
16 PE, soit 768 bascules, 16 multiplieurs et 128 fils de sortie (`psum_south`).

---

## Implémentation

- Fichier : [`pe_array.py`](pe_array.py) — classes `PE_Array_OS` et `PE_Array_WS`.
- Bloc structurel : il instancie N×N PE et les câble ; les bus de voies sont découpés avec `Range` et réassemblés
  avec `ConcatenateLSBF`. En WS, la ligne 0 reçoit une constante 0 comme somme partielle.
- L'attribut `pes[r][c]` donne accès à chaque PE (utilisé par les bancs de test pour observer la grille).

| Constructeur | Ports, dans l'ordre |
|---|---|
| `PE_Array_OS(parent, name, …, N=4)` | `rst_n, w_west, x_north, clear, acc` |
| `PE_Array_WS(parent, name, …, N=4)` | `rst_n, x_west, w_north, load, psum_south` |

**Vérification** ([`test/pe_array/`](../../test/pe_array/)) : câblage comparé cycle par cycle à une grille de modèles de
référence du PE ; contrats de ce document vérifiés pour N = 1, 2, 3, 4, 8 — résultat juste **et** disponible exactement au
cycle annoncé (pas un cycle plus tôt) ; préchargement WS ; deux tuiles successives ; écart minimal avant rechargement.

**Mesuré** : une tuile prend L + 2N − 2 cycles en OS et L + 3N − 2 en WS (préchargement compris) pour un flux de
longueur L. En WS, il faut **au moins 2N − 2 cycles** entre la dernière colonne de X (voie 0) et le premier `load`
suivant : avec un cycle de moins, la dernière colonne de Y est fausse.
