"""
Data layout in memory (architecture.md, D6): what the host software must do
to store W and X in the buffers, and to read Y back.

A word holds N int8 values (lane l = byte l). The address of a word is the concatenation
{tile index, element index} = (t << IW) | u.

| Matrix  | Variant  | Word at address {t, u} | holds               |
|---------|----------|------------------------|---------------------|
| W       | OS & WS  | {i_t, k}               | W[N·i_t + l][k]     |
| X       | OS       | {j_t, k}               | X[k][N·j_t + l]     |
| X       | WS       | {k_t, j}               | X[N·k_t + l][j]     |
| Y       | OS       | {j_t, i}               | Y[i][N·j_t + l]     |
| Y       | WS       | {i_t, j}               | Y[N·i_t + l][j]     |

Each variant writes Y in the X format it reads: the output of a layer is copied as is
into the IN bank for the next layer.
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
    """{address: [N values]} for W (M × K), identical in both variants."""
    M, K = W.shape
    return {addr(it, k): _lanes(lambda i: W[i][k], it, M) for it in range(tiles(M)) for k in range(K)}


def x_image(X, variant: str) -> dict:
    """{address: [N values]} for X (K × N_dim), in the format of the variant."""
    K, Nd = X.shape
    if variant == 'OS':
        return {addr(jt, k): _lanes(lambda j: X[k][j], jt, Nd) for jt in range(tiles(Nd)) for k in range(K)}
    return {addr(kt, j): _lanes(lambda k: X[k][j], kt, K) for kt in range(tiles(K)) for j in range(Nd)}


def y_addresses(M: int, Nd: int, variant: str) -> list:
    """Addresses of the OUT bank that hold Y (M × N_dim)."""
    if variant == 'OS':
        return [addr(jt, i) for jt in range(tiles(Nd)) for i in range(M)]
    return [addr(it, j) for it in range(tiles(M)) for j in range(Nd)]


def y_from_words(words: dict, M: int, Nd: int, variant: str):
    """Rebuilds Y (list of lists) from {address: [N values]} read from the OUT bank."""
    Y = [[0] * Nd for _ in range(M)]
    for a, lanes in words.items():
        t, u = a >> IW, a & ((1 << IW) - 1)
        for l in range(N):
            i, j = (u, N * t + l) if variant == 'OS' else (N * t + l, u)
            if i < M and j < Nd:
                Y[i][j] = lanes[l]
    return Y
