import numpy as np
from random import choice


def diagonal(n, k):
    A_k = np.zeros(shape=(n, n))
    a_ij = [-1, -2, -3, -4]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rand = choice(a_ij)
            A_k[i, j] += rand + 10 ** (-k)

    for i in range(n):
        sum = 0
        for j in range(n):
            if i == j:
                continue
            sum += A_k[i, j]
        A_k[i, i] = sum

    return A_k


def hilbert(k):
    A_k = np.zeros(shape=(k, k))
    for i in range(k):
        for j in range(k):
            A_k[i, j] = 1.0 / (i + j + 1.0)

    return A_k