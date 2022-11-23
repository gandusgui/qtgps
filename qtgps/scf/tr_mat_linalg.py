import numpy as np

def matmul(A_qii, A_qij, A_qji, B_qii, B_qij, B_qji):
    N = len(A_qii)
    # Diagonal sum
    ans_qii = [a @ b for a,b in zip(A_qii,B_qii)]
    # Upper diagonal sum
    for q in range(N-1):
        ans_qii[q][:] += A_qij[q] @ B_qji[q]
    # Lower diagonal sum
    for q in range(1,N):
        ans_qii[q][:] += A_qji[q-1] @ B_qij[q-1]

    return ans_qii

def matadd(A_qii, A_qij, A_qji, B_qii, B_qij, B_qji):
    ans_qii = [a + b for a,b in zip(A_qii,B_qii)]
    ans_qij = [a + b for a,b in zip(A_qij,B_qij)]
    ans_qji = [a + b for a,b in zip(A_qji,B_qji)]

    return ans_qii, ans_qij, ans_qji

def imatadd(A_qii, A_qij, A_qji, B_qii, B_qij, B_qji):
    N = len(A_qii)
    for q in range(N):
        A_qii[q] += B_qii[q]
    for q in range(N-1):
        A_qij[q] += B_qij[q]
    for q in range(N-1):
        A_qji[q] += B_qji[q]

def matsub(A_qii, A_qij, A_qji, B_qii, B_qij, B_qji):
    ans_qii = [a - b for a,b in zip(A_qii,B_qii)]
    ans_qij = [a - b for a,b in zip(A_qij,B_qij)]
    ans_qji = [a - b for a,b in zip(A_qji,B_qji)]

    return ans_qii, ans_qij, ans_qji

def imatsub(A_qii, A_qij, A_qji, B_qii, B_qij, B_qji):
    N = len(A_qii)
    for q in range(N):
        A_qii[q] -= B_qii[q]
    for q in range(N-1):
        A_qij[q] -= B_qij[q]
    for q in range(N-1):
        A_qji[q] -= B_qji[q]

def mul(A_qii, A_qij, A_qji, c):
    ans_qii = [m * c for m in A_qii]
    ans_qij = [m * c for m in A_qij]
    ans_qji = [m * c for m in A_qji]

    return ans_qii, ans_qij, ans_qji

def imul(A_qii, A_qij, A_qji, c):
    N = len(A_qii)
    for q in range(N):
        A_qii[q] *= c
    for q in range(N-1):
        A_qij[q] *= c
    for q in range(N-1):
        A_qji[q] *= c

def add(A_qii, A_qij, A_qji, c):
    ans_qii = [m + c for m in A_qii]
    ans_qij = [m + c for m in A_qij]
    ans_qji = [m + c for m in A_qji]

    return ans_qii, ans_qij, ans_qji

def iadd(A_qii, A_qij, A_qji, c):
    N = len(A_qii)
    for q in range(N):
        A_qii[q] += c
    for q in range(N-1):
        A_qij[q] += c
    for q in range(N-1):
        A_qji[q] += c

def sub(A_qii, A_qij, A_qji, c):
    ans_qii = [m - c for m in A_qii]
    ans_qij = [m - c for m in A_qij]
    ans_qji = [m - c for m in A_qji]

    return ans_qii, ans_qij, ans_qji

def isub(A_qii, A_qij, A_qji, c):
    N = len(A_qii)
    for q in range(N):
        A_qii[q] -= c
    for q in range(N-1):
        A_qij[q] -= c
    for q in range(N-1):
        A_qji[q] -= c

def diagonal(A_qii):
    nao = sum(len(A) for A in A_qii)
    A_i = np.zeros(nao, A_qii[0].dtype)
    # Loop over diagonal elements
    i0 = 0
    for A_ii in A_qii:
        i1 = i0 + len(A_ii)
        A_i[i0:i1] = A_ii.diagonal()
        i0 = i1
    return A_i
