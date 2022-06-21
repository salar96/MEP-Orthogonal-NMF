# Implemented based on the following paper:
# Megasthenis Asteris, Dimitris Papailiopoulos, and Alexandros G Dimakis. Orthogonal NMF through subspace exploration. In Advances in Neural Information Processing Systems, pages 343–351, 2015.

import numpy as np
from utils import divide, normalize

class ONMFS:
    name = "ONMFS"
    def func (M, k):
        M = M.T
        m, n = np.shape(M)
        
        r = k
        u, s, vh = np.linalg.svd(M, full_matrices=False, compute_uv=True)
        
        s = np.diag(s)[:, :r]
        best_W = np.zeros((m, k))
        best_H = np.zeros((n, k))
        best_Fro = 0
        for _ in range(2000):
            #print(_)
            C = np.random.normal(loc=0.0, scale=1.0, size=(r,k))
            for i in range(k):
                C = normalize(C, 0)
            A = u @ s @ C
            for j in range(200):
                W = ONMFS.LocalOptW(A)
                H = M.T @ W
                temp_Fro = np.linalg.norm(H, ord = "fro")
                if (temp_Fro > best_Fro):
                    bestFro = temp_Fro
                    best_W = W
                    best_H = H
        
        return best_H, best_W
    def LocalOptW (A):
        m, k = np.shape(A)
        W = np.zeros((m, k))
        s = np.random.randint(2,size = k)
        s = s * 2 - 1
        A_ = A * s.reshape((1, -1))
        argm = np.argmax(A_, axis = 1)

        for i in range(m):
            if A_[i, argm[i]] > 0:
                W[i, argm[i]] = A_[i, argm[i]]
        W = normalize(W, 0)
        return W
