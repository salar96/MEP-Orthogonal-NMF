import numpy as np
from utils import divide, normalize

class ONMF_Ding_double:
    name = "ONMF-Ding"
    def func (X_original,k):
        X = X.T
        m,n = np.shape(X_original)
        F = np.random.rand(m,k)
        G = np.random.rand(n, k)
        S = np.random.rand(k, k)
        X = X_original + 0.01
        # F = np.random.exponential(1, size = (m,k))
        # G = np.random.exponential(1, size = (n,k))
        # S = np.random.exponential(1, size = (k,k))
        F_diff, G_diff = 1, 1
        _ = 0
        reconstruction_err = 100
        while F_diff >= 1e-5 or G_diff >= 1e-5 or reconstruction_err > 15:
            P = X @ G @ S.T
            temp = F @ F.T @ P
            P = divide(P, temp)
            F_next = F * np.sqrt(P)

            if _ % 100 == 0:
                F_diff = np.linalg.norm(F - F_next, ord = "fro") /np.linalg.norm(F, ord = "fro")
            F = F_next

            P = X.T @ F @ S
            temp = G @ G.T @ P
            P = divide(P, temp)
            G_next = G * np.sqrt(P)

            if _ % 100 == 0:
                G_diff = np.linalg.norm(G - G_next, ord = "fro") /np.linalg.norm(G, ord="fro")

            G = G_next

            P = F.T @ X @ G
            temp = F.T @ F @ S @ G.T @ G
            P = divide (P, temp)
            S *= np.sqrt(P)

            if _ % 100 == 0:
                reconstruction_err = np.linalg.norm(F @ S @ G.T - X, ord = "fro")
            if _ % 100 == 0:
                print("ONMF_Ding", F_diff, G_diff)
                print("reconstruction err", reconstruction_err)
            _ += 1
        # return F, S, G
        return G, S.T, F
