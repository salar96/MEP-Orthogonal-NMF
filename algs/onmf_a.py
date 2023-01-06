# Implemented based on the following paper:
# Seungjin Choi. Algorithms for orthogonal nonnegative matrix factorization. In 2008 ieee international joint conference on neural networks (ieee world congress on computational intelligence), pages 1828–1832. IEEE, 2008.

import numpy as np
from utils import divide, normalize

class ONMF_A:
    name = "ONMF-A"
    def func (X,k):
        X = X.T
        m,n = np.shape(X)
        F = np.random.rand(m,k)
        G = np.random.rand(n,k)
        F_diff, G_diff = 1, 1
        count = 0
        period = 5
        max_iter=1000
        i=0
        # while F_diff >= 1e-1 and G_diff >= 1e-1:
        while F_diff >= 1.5e-3 or G_diff >= 1.5e-3:
            denominator = F @ G.T @ X.T @ F
            P = divide (X @ G, denominator)
            F_next = F * P

            F_next /= np.linalg.norm(F_next, ord = 2)
            if count == 0:
                F_diff = np.linalg.norm(F - F_next, ord = "fro")/np.linalg.norm(F, ord = "fro")
            F = F_next

            denominator = G @ F.T @ F
            P = divide (X.T @ F, denominator)
            G_next = G * P

            if count == 0:
                G_diff = np.linalg.norm(G - G_next, ord = "fro")/np.linalg.norm(G, ord = "fro")
            G = G_next

            
            count += 1
            if count >= period:
                count = 0
            if i>max_iter:
                break
            i=i+1
        return G, F
