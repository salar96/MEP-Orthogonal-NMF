# Implemented based on the following paper:
# Daniel D Lee and H Sebastian Seung. Algorithms for non-negative matrix factorization. In Advances in neural information processing systems, pages 556–562, 2001.

import numpy as np
from utils import divide, normalize

class NMF:
    name = "NMF"
    def func (X,k, double = False):
        m,n = np.shape(X)
        F = np.random.rand(m,k)
        G = np.random.rand(n,k)
        F_diff, G_diff = 1,1
        count = 0
        period = 50
        while F_diff >= 1e-5 or G_diff >= 1e-5:
            denominator = F @ G.T @ G
            P = divide(X @ G, denominator)
            F_next = F * P
            if count == 0:
                F_diff = np.linalg.norm(F - F_next, ord = "fro")/np.linalg.norm(F, ord = "fro")
            F = F_next
            denominator = G @ F.T @ F
            P = divide(X.T @ F, denominator)
            G_next = G * P
            if count == 0:
                G_diff = np.linalg.norm(G - G_next, ord = "fro")/np.linalg.norm(G, ord = "fro")
            G = G_next
            
            count += 1
            if count >= period:
                count = 0
        if double:
            return F, np.eye(k), G
        return F, G
