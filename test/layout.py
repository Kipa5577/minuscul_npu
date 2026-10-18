"""
Disposition des données en mémoire (architecture.md, D6) : ce que le logiciel de l'hôte doit faire
pour ranger W et X dans les buffers, et pour relire Y.

Un mot contient N valeurs int8 (voie l = octet l). L'adresse d'un mot est la concaténation
{indice de tuile, indice d'élément} = (t << IW) | u.

| Matrice | Variante | Mot à l'adresse {t, u} | contient            |
|---------|----------|------------------------|---------------------|
| W       | OS et WS | {i_t, k}               | W[N·i_t + l][k]     |
| X       | OS       | {j_t, k}               | X[k][N·j_t + l]     |
| X       | WS       | {k_t, j}               | X[N·k_t + l][j]     |
| Y       | OS       | {j_t, i}               | Y[i][N·j_t + l]     |
| Y       | WS       | {i_t, j}               | Y[N·i_t + l][j]     |

Chaque variante écrit Y au format X qu'elle lit : la sortie d'une couche se recopie telle quelle
dans le banc IN pour la couche suivante.
"""

N = 4
IW = 6


def addr(t: int, u: int) -> int:
    return (t << IW) | u


def tiles(D: int) -> int:
    return -(-D // N)


def _lanes(get, t, D):
    return [int(get(N * t + l)) if N * t + l < D else 0 for l in range(N)]


def w_image(W) -> dict:
    """{adresse: [N valeurs]} pour W (M × K), identique dans les deux variantes."""
    M, K = W.shape
    return {addr(it, k): _lanes(lambda i: W[i][k], it, M) for it in range(tiles(M)) for k in range(K)}


def x_image(X, variant: str) -> dict:
    """{adresse: [N valeurs]} pour X (K × N_dim), au format de la variante."""
    K, Nd = X.shape
    if variant == 'OS':
        return {addr(jt, k): _lanes(lambda j: X[k][j], jt, Nd) for jt in range(tiles(Nd)) for k in range(K)}
    return {addr(kt, j): _lanes(lambda k: X[k][j], kt, K) for kt in range(tiles(K)) for j in range(Nd)}


def y_addresses(M: int, Nd: int, variant: str) -> list:
    """Adresses du banc OUT qui contiennent Y (M × N_dim)."""
    if variant == 'OS':
        return [addr(jt, i) for jt in range(tiles(Nd)) for i in range(M)]
    return [addr(it, j) for it in range(tiles(M)) for j in range(Nd)]


def y_from_words(words: dict, M: int, Nd: int, variant: str):
    """Reconstruit Y (liste de listes) à partir de {adresse: [N valeurs]} lus dans le banc OUT."""
    Y = [[0] * Nd for _ in range(M)]
    for a, lanes in words.items():
        t, u = a >> IW, a & ((1 << IW) - 1)
        for l in range(N):
            i, j = (u, N * t + l) if variant == 'OS' else (N * t + l, u)
            if i < M and j < Nd:
                Y[i][j] = lanes[l]
    return Y
