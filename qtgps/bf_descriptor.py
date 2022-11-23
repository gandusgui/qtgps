import numpy as np

class BasisFunctionDescriptor:

    def __init__(self, M_a, nbf):
        self.M_a = M_a
        self.nbf = nbf
        self.na = len(M_a)

        self.nbf_a = np.zeros_like(M_a)
        self.nbf_a[:-1] = np.diff(M_a)
        self.nbf_a[-1] = nbf - M_a[-1]

    def expand(self, X_a, weights=None):
        X_i = np.repeat(X_a, self.nbf_a)
        if weights is None:
            return X_i
        return X_i * weights

    def sum(self, X_i, return_weights=True):
        dtype = X_i.dtype
        X_a = np.zeros(self.na, dtype=dtype)
        for a, i in enumerate(self.M_a):
            X_a[a] = sum(X_i[i:i+self.nbf_a[a]])
        if return_weights:
            return X_a, X_i / self.expand(X_a)
        return X_a
