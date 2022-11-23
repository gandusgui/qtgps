import numpy as np
from . import tr_mat_linalg as mla

class MatLists:

    def __init__(self, lengths=None, N=None, dtype=float):
        if lengths is not None:
            N = len(lengths)
            self.m_qii = [np.zeros((lengths[i], lengths[i]), dtype=dtype) for i in range(N)]
            self.m_qij = [np.zeros((lengths[i], lengths[i+1]), dtype=dtype) for i in range(N-1)]
            self.m_qji = [np.zeros((lengths[i+1], lengths[i]), dtype=dtype) for i in range(N-1)]
            self.N = N
        elif N is not None:
            self.m_qii = [None for _ in range(N)]
            self.m_qij = [None for _ in range(N-1)]
            self.m_qji = [None for _ in range(N-1)]
            self.N = N
        else:
            self.m_qii = None
            self.m_qij = None
            self.m_qji = None
            self.N = 0

    @classmethod
    def zeros_like(cls, other, dtype=None):
        if isinstance(other, MatLists):
            m_qii = other.m_qii
        elif isinstance(other, list):
            m_qii = other
        else:
            raise ValueError('{} not supported'.format(type(other)))
        lengths = [len(m) for m in m_qii]
        dtype = dtype or m_qii[0].dtype
        return cls(lengths=lengths, dtype=dtype)

    def __lshift__(self, ptr):
        self.m_qii = ptr[0]
        self.m_qij = ptr[1]
        self.m_qji = ptr[2]
        self.N = len(ptr[0])

    @property
    def imag(self):
        ans = MatLists()
        ans_qii = [m.imag for m in self.m_qii]
        ans_qij = [m.imag for m in self.m_qij]
        ans_qji = [m.imag for m in self.m_qji]
        ans << (ans_qii, ans_qij, ans_qji)
        return ans

    @property
    def real(self):
        ans = MatLists()
        ans_qii = [m.real for m in self.m_qii]
        ans_qij = [m.real for m in self.m_qij]
        ans_qji = [m.real for m in self.m_qji]
        ans << (ans_qii, ans_qij, ans_qji)
        return ans

    def diagonal(self):
        return mla.diagonal(self.m_qii)

    def __len__(self):
        return self.N

    def _args(self, other=None):
        if other is None:
            return self.m_qii, self.m_qij, self.m_qji
        # elif type(other) == type(self):
        elif isinstance(other, MatLists):
            return other.m_qii, other.m_qij, other.m_qji
        # elif type(other) in [list, tuple]:
        elif isinstance(other, (list, tuple)):
            return other
        else:
            raise ValueError('{} not supported'.format(type(other)))

    def __matmul__(self, other):
        A_args = self._args()
        B_args = self._args(other)
        ans = MatLists()
        ans << (mla.matmul(*A_args, *B_args), 0, 0)
        return ans

    def __mul__(self, c):
        A_args = self._args()
        ans = MatLists()
        ans << mla.mul(*A_args, c)
        return ans

    def __imul__(self, c):
        A_args = self._args()
        mla.imul(*A_args, c)
        return self

    def __add__(self, other):
        A_args = self._args()
        ans = MatLists()
        try:
            len(other)
        except:
            ans << mla.add(*A_args, other)
        else:
            B_args = self._args(other)
            ans << mla.matadd(*A_args, *B_args)
        return ans

    def __iadd__(self, other):
        A_args = self._args()
        try:
            len(other)
        except:
            mla.iadd(*A_args, other)
        else:
            B_args = self._args(other)
            mla.imatadd(*A_args, *B_args)
        return self

    def __sub__(self, other):
        A_args = self._args()
        ans = MatLists()
        try:
            len(other)
        except:
            ans << mla.sub(*A_args, other)
        else:
            B_args = self._args(other)
            ans << mla.matsub(*A_args, *B_args)
        return ans

    def __isub__(self, other):
        A_args = self._args()
        try:
            len(other)
        except:
            mla.isub(*A_args, other)
        else:
            B_args = self._args(other)
            mla.imatsub(*A_args, *B_args)
        return self
