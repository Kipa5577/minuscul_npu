# weight_buffer — SRAM des poids

C'est une mémoire **simple port** de 1024 × 32 bits (4 Kio). Elle est vue à travers deux ports logiques,
le port hôte et le port cœur, et le signal `busy` choisit lequel des deux est actif.
La disposition des données est décrite en D6 : le mot à l'adresse `{i_t, k}` contient `W[4·i_t + l][k]`.

## Paramètres
`N` = 4, `DW` = 8 (largeur de mot N·DW = 32), `BAW` = 10 (1024 mots)

## Ports
| Signal | Dir | Largeur | Description |
|---|---|---|---|
| `clk` | in | 1 | (le contenu n'est pas réinitialisé par `rst_n`) |
| `busy` | in | 1 | 0 : port hôte actif · 1 : port cœur actif |
| `host_addr` | in | BAW | adresse de mot |
| `host_wdata` | in | N·DW | |
| `host_we` | in | 1 | écriture d'un mot complet |
| `host_re` | in | 1 | |
| `host_rdata` | out | N·DW | latence 1 |
| `core_addr` | in | BAW | depuis le `controller` |
| `core_re` | in | 1 | |
| `core_rdata` | out | N·DW | latence 1, vers le `data_feeder` |

## Comportement
```
port actif = busy ? cœur : hôte          (les accès du port inactif sont ignorés)
écriture   : si we (port actif) : mem[addr] ← wdata
lecture    : si re (port actif) : rdata ← mem[addr]   (au front suivant, sinon rdata garde sa valeur)
```
- `host_rdata` et `core_rdata` peuvent être le même fil physique.

## Latences
Lecture : L = 1. Écriture : effective au front qui suit.

---

## Implémentation

- Fichier : [`weight_buffer.py`](weight_buffer.py) — classe `WeightBuffer`, `clock()` uniquement (pas de `rst_n`).
- Un seul registre de lecture, partagé par les deux ports logiques : `host_rdata` et `core_rdata` ont toujours la même
  valeur. Une lecture et une écriture à la même adresse dans le même cycle rendent l'ancienne valeur.
- La profondeur se déduit de la largeur de `host_addr`, la largeur de mot de celle de `host_wdata`.

| Constructeur | Ports, dans l'ordre |
|---|---|
| `WeightBuffer(parent, name, …)` | `busy, host_addr, host_wdata, host_we, host_re, host_rdata, core_addr, core_re, core_rdata` |

**Vérification** ([`test/weight_buffer/`](../../test/weight_buffer/)) : latence 1, maintien de la donnée lue, lecture et
écriture simultanées, port actif choisi par `busy`, modèle de référence sur 3000 accès aléatoires ; W rangé par l'hôte
selon D6 (`test/layout.py`) puis relu aux adresses émises par le planificateur du `controller` : chaque lecture rend
exactement le mot attendu par le cœur de calcul.

**Mesuré** : en OS, chaque mot de W est relu une fois par tuile de colonnes de X (8 fois le minimum pour un GEMM 32³,
port occupé 73 % du temps de calcul) ; en WS, chaque mot n'est lu qu'une fois (port occupé 10 % du temps).
