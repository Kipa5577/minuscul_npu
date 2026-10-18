# host_interface — Bus hôte et registres

Le bloc décode le bus parallèle de 32 bits, contient les registres de configuration et de statut,
et redirige les accès vers les fenêtres mémoire (voir la carte mémoire en D7).

## Paramètres
`N` = 4 (la carte mémoire suppose un mot de buffer de 32 bits), `DIM_MAX` = 64, `BAW` = 10, `IW` = 6, `DATAFLOW`

## Ports côté bus
| Signal | Dir | Largeur | Description |
|---|---|---|---|
| `clk`, `rst_n` | in | 1 | |
| `bus_addr` | in | 13 | adresse de mot |
| `bus_wdata` | in | 32 | |
| `bus_we` | in | 1 | |
| `bus_re` | in | 1 | |
| `bus_rdata` | out | 32 | **latence 1** pour toutes les adresses |

## Ports internes
| Signal | Dir | Largeur | Vers / depuis |
|---|---|---|---|
| `dim_m`, `dim_k`, `dim_n` | out | 7 chacun | `controller` |
| `req_mult` | out | 16 | `post_processing` |
| `req_shift` | out | 5 | `post_processing` |
| `relu_en` | out | 1 | `post_processing` |
| `go` | out | 1 | impulsion vers le `controller` |
| `busy` | in | 1 | `controller` |
| `done` | in | 1 | impulsion du `controller` |
| `cycles` | in | 32 | `controller` |
| `host_wdata` | out | 32 | commun à toutes les mémoires (= `bus_wdata`) |
| `wb_addr`, `wb_we`, `wb_re` / `wb_rdata` | out / in | BAW, 1, 1 / 32 | `weight_buffer` (port hôte) |
| `ab_addr`, `ab_we`, `ab_re` / `ab_rdata` | out / in | BAW+1, 1, 1 / 32 | `activation_buffer` (port hôte) |
| `bias_addr`, `bias_we`, `bias_re` / `bias_rdata` | out / in | IW, 1, 1 / 32 | `post_processing` (port des biais) |

## Décodage d'adresse
| `bus_addr[12:10]` | Cible | Adresse transmise |
|---|---|---|
| `000` | registres | `bus_addr[3:0]` |
| `001` | `WEIGHT` | `wb_addr = bus_addr[9:0]` |
| `010`, `011` | `ACT_IN`, `ACT_OUT` | `ab_addr = bus_addr[10:0]` (le bit 10 choisit le banc) |
| `100` | `BIAS` | `bias_addr = bus_addr[5:0]` |
| autres | — | lecture de 0, écriture ignorée |

## Registres
| Adr. | Nom | Champs | Reset |
|---|---|---|---|
| 0 | `ID` (RO) | [3:0] N · [4] DATAFLOW (0 = OS, 1 = WS) · [15:8] DIM_MAX · [31:24] version = 1 | constante |
| 1 | `CTRL` | [0] `GO` (écrire 1 envoie l'impulsion, se relit 0) · [1] `RELU_EN` | 0 |
| 2 | `STATUS` (RO) | [0] `BUSY` · [1] `DONE` (mis à 1 par `done`, remis à 0 par `GO`) | 0 |
| 3–5 | `DIM_M/K/N` | [6:0] | 0 |
| 6 | `REQ_MULT` | [15:0] | 1 |
| 7 | `REQ_SHIFT` | [4:0] | 0 |
| 8 | `CYCLES` (RO) | [31:0] | 0 |

- Avec les valeurs de reset, la requantification calcule `y = sat8(acc + b)`.
- Le NPU est **occupé** quand le controller a `busy = 1` **ou** qu'une impulsion `go` est en cours. Le controller ne
  passe `busy` à 1 qu'un cycle après l'impulsion : sans cette règle, un accès de l'hôte juste après GO passerait
  avant le début du calcul (défaut trouvé par les tests de `npu_top`).
- Quand le NPU est occupé : les écritures de configuration, les `GO` et **tous les accès aux mémoires** (WEIGHT,
  ACT_IN, ACT_OUT, BIAS) sont ignorés, et `STATUS.BUSY` vaut 1. Les registres restent lisibles.
- La lecture de `bus_rdata` passe par un multiplexeur dont la sélection est mémorisée au cycle de `bus_re`,
  ce qui l'aligne sur la latence de 1 des mémoires.
- `irq = STATUS.DONE` est exporté par `npu_top`.

---

## Implémentation

- Fichier : [`host_interface.py`](host_interface.py) — classe `HostInterface(…, dataflow='OS', N=4)`.
- `clock()` : registres de configuration, impulsion `go` (registre d'un cycle), DONE mémorisé (GO prioritaire si les
  deux arrivent au même cycle), cible et valeur de registre mémorisées au cycle de `bus_re`.
- `propagate()` : décodage d'adresse vers les ports hôte des mémoires (combinatoire, forcé à 0 quand le NPU est
  occupé), sorties de configuration, `irq`, et multiplexeur de `bus_rdata` d'après la cible mémorisée.
- Une écriture de GO au cycle t produit `go` pendant le cycle t + 1 ; le calcul commence au cycle t + 2.

| Constructeur | Ports, dans l'ordre |
|---|---|
| `HostInterface(parent, name, …, dataflow, N)` | `rst_n, bus_addr, bus_wdata, bus_we, bus_re, bus_rdata, dim_m, dim_k, dim_n, req_mult, req_shift, relu_en, go, busy, done, cycles, host_wdata, wb_addr, wb_we, wb_re, wb_rdata, ab_addr, ab_we, ab_re, ab_rdata, bias_addr, bias_we, bias_re, bias_rdata, irq` |

**Vérification** ([`test/host_interface/`](../../test/host_interface/)) : ID et valeurs de reset, registres et largeur
de leurs champs, décodage de chaque fenêtre, impulsion `go`, DONE et `irq`, blocage pendant `busy` et pendant
l'impulsion `go`, latence de lecture 1 pour toutes les cibles.

**Mesuré sur la puce complète** : le passage par le bus ramène le débit d'une couche 64³ de 13,5 à 11,6 MAC/cycle
en OS (de 14,0 à 12,0 en WS) ; pour une couche 4³, l'hôte occupe plus de la moitié du temps.
