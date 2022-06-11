import numpy as np

from scipy.sparse import csc_array
from math import sqrt


def jacobi(A: csc_array, tol=1.0e-9):  # Jacobi method

    def maxElem(A: csc_array):  # Find largest off-diag. element A[i_max, j_max]
        n = A.shape[0]
        aMax = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if abs(A[i, j]) >= aMax:
                    aMax = abs(A[i, j])
                    i_max = i
                    j_max = j
        return aMax, i_max, j_max

    def rotate(A: csc_array, P: csc_array, i_max, j_max):  # Rotate to make a[i_max, j_max] = 0
        n = A.shape[0]
        aDiff = A[j_max, j_max] - A[i_max, i_max]

        if abs(A[i_max, j_max]) < abs(aDiff) * 1.0e-3:  # Seems better to test this hardcoded constant
            t = A[i_max, j_max] / aDiff
        else:
            phi = aDiff / (2.0 * A[i_max, j_max])
            t = 1.0 / (abs(phi) + sqrt(phi ** 2 + 1.0))
            if phi < 0.0:
                t = -t

        c = 1.0 / sqrt(t ** 2 + 1.0)
        s = t * c
        tau = s / (1.0 + c)

        temp = A[i_max, j_max]
        A[i_max, j_max] = 0.0
        A[i_max, i_max] = A[i_max, i_max] - t * temp
        A[j_max, j_max] = A[j_max, j_max] + t * temp

        for i in range(i_max):  # Case of i < k
            temp = A[i, i_max]
            A[i, i_max] = temp - s * (A[i, j_max] + tau * temp)
            A[i, j_max] = A[i, j_max] + s * (temp - tau * A[i, j_max])

        for i in range(i_max + 1, j_max):  # Case of k < i < l
            temp = A[i_max, i]
            A[i_max, i] = temp - s * (A[i, j_max] + tau * A[i_max, i])
            A[i, j_max] = A[i, j_max] + s * (temp - tau * A[i, j_max])

        for i in range(j_max + 1, n):  # Case of i > l
            temp = A[i_max, i]
            A[i_max, i] = temp - s * (A[j_max, i] + tau * temp)
            A[j_max, i] = A[j_max, i] + s * (temp - tau * A[j_max, i])

        for i in range(n):  # Update transformation matrix
            temp = P[i, i_max]
            P[i, i_max] = temp - s * (P[i, j_max] + tau * P[i, i_max])
            P[i, j_max] = P[i, j_max] + s * (temp - tau * P[i, j_max])

    n = A.shape[0]
    count_iterations = 0
    maxRot = 5 * (n ** 2)  # Set limit on number of rotations
    P = csc_array(np.identity(n, dtype=np.float_))  # Initialize transformation matrix
    for i in range(maxRot):  # Jacobi rotation loop
        count_iterations += 1
        aMax, i_max, j_max = maxElem(A)
        if aMax < tol:
            return np.diagonal(A.toarray()), P.toarray(), count_iterations
        rotate(A, P, i_max, j_max)
    print('Jacobi method did not converge')