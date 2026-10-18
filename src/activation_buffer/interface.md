# activation_buffer — SRAM des activations

Le bloc contient deux bancs **simple port** indépendants de 1024 × 32 bits chacun :
- **IN** : X, lu par le cœur ;
- **OUT** : Y, écrit par `post_processing`.

Comme les bancs sont séparés, le cœur peut lire X et écrire Y dans le même cycle.
La disposition des données est décrite en D6, et dépend de la variante.

## Paramètres
`N` = 4, `DW` = 8, `BAW` = 10

## Ports
| Signal | Dir | Largeur | Description |
|---|---|---|---|
| `clk` | in | 1 | (le contenu n'est pas réinitialisé par `rst_n`) |
| `busy` | in | 1 | 0 : port hôte actif sur les deux bancs · 1 : ports cœur actifs |
| `host_addr` | in | BAW+1 | bit BAW : 0 = IN, 1 = OUT ; bits [BAW−1:0] : adresse de mot |
| `host_wdata` | in | N·DW | |
| `host_we` | in | 1 | écriture d'un mot complet |
| `host_re` | in | 1 | |
| `host_rdata` | out | N·DW | latence 1 |
| `core_rd_addr` | in | BAW | banc IN, depuis le `controller` |
| `core_rd_re` | in | 1 | |
| `core_rd_rdata` | out | N·DW | latence 1, vers le `data_feeder` |
| `core_wr_addr` | in | BAW | banc OUT, depuis `post_processing` (`y_addr`) |
| `core_wr_wdata` | in | N·DW | `y_wdata` |
| `core_wr_we` | in | 1 | `y_we` |
| `core_wr_wmask` | in | N | `y_wmask` : un bit d'autorisation d'écriture par octet (voie) |

## Comportement
```
banc IN  : busy ? lecture cœur : accès hôte (si host_addr[BAW] = 0)
banc OUT : busy ? écriture cœur (octets autorisés par wmask) : accès hôte (si host_addr[BAW] = 1)
host_rdata : mot du banc choisi par host_addr[BAW] (le choix est mémorisé un cycle)
```

## Latences
Lecture : L = 1. Écriture : effective au front qui suit.

---

## Implémentation

- Fichier : [`activation_buffer.py`](activation_buffer.py) — classe `ActivationBuffer`, `clock()` uniquement (pas de `rst_n`).
- Chaque banc a son registre de lecture ; `host_rdata` présente celui du banc lu en dernier par l'hôte
  (le choix est mémorisé avec la donnée). `core_rd_rdata` est le registre de lecture du banc IN.
- L'écriture du cœur dans le banc OUT ne modifie que les octets autorisés par `core_wr_wmask`.

| Constructeur | Ports, dans l'ordre |
|---|---|
| `ActivationBuffer(parent, name, …)` | `busy, host_addr, host_wdata, host_we, host_re, host_rdata, core_rd_addr, core_rd_re, core_rd_rdata, core_wr_addr, core_wr_wdata, core_wr_we, core_wr_wmask` |

**Vérification** ([`test/activation_buffer/`](../../test/activation_buffer/)) : accès hôte aux deux bancs, lecture IN
et écriture OUT dans le même cycle, masque par octet, côté actif choisi par `busy`, modèle de référence sur 3000 accès,
X rangé selon D6 puis relu aux adresses du planificateur. **De bout en bout avec les mémoires** : l'hôte charge W, X et
les biais mot par mot, le cœur calcule, l'hôte relit Y dans le banc OUT ; et **deux couches enchaînées** en recopiant
Y tel quel du banc OUT vers le banc IN, ce qui vérifie que Y sort bien au format X de la variante (D6).

**Mesuré** : sans DMA, l'hôte transfère un mot par cycle et le cœur attend : chargement et relecture représentent la
moitié du temps total pour un GEMM 4³, et encore 14 % pour un GEMM 64³. Entre deux couches, la recopie OUT → IN par
l'hôte coûte 3 cycles par mot de Y (578 cycles pour la couche 64 → 48 d'un MLP sur un lot de 16).
