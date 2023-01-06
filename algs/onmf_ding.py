# Implemented based on the following paper:
# Chris Ding, Tao Li, Wei Peng, and Haesun Park. Orthogonal nonnegative matrix t-factorizations for clustering. In Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 126–135, 2006.


import numpy as np
from utils import divide, normalize

class ONMF_Ding:
    name = "ONMF-Ding"
    def func (X,k, double = False):
        if double:
            m,n = np.shape(X)
            F = np.random.rand(m,k)
            G = np.random.rand(n, k)
            S = np.random.rand(k, k)
            F_diff, G_diff = 1, 1
            count = 0
            period = 100
            while F_diff >= 1e-5 or G_diff >= 1e-5:
                P = X @ G @ S.T
                temp = F @ F.T @ P
                P = divide(P, temp)
                F_next = F * np.sqrt(P)

                if count == 0:
                    F_diff = np.linalg.norm(F - F_next, ord = "fro") /np.linalg.norm(F, ord = "fro")
                F = F_next

                P = X.T @ F @ S
                temp = G @ G.T @ P
                P = divide(P, temp)
                G_next = G * np.sqrt(P)

                if count == 0:
                    G_diff = np.linalg.norm(G - G_next, ord = "fro") /np.linalg.norm(G, ord="fro")

                G = G_next

                P = F.T @ X @ G
                temp = F.T @ F @ S @ G.T @ G
                P = divide (P, temp)
                S *= np.sqrt(P)

                
                count += 1
                if count >= period:
                    count = 0
            return F, S, G
        else:
            X = X.T
            m,n = np.shape(X)
            F = (np.random.rand(m,k))
            G = (np.random.rand(n, k))
            F_diff, G_diff = 1, 1
            count = 0
            period = 10
            while F_diff >= 5e-5 or G_diff >= 5e-5:
                P = X @ G
                temp = F @ F.T @ P
                P = divide(P, temp)
                F_next = F * np.sqrt(P)
                if count == 0:
                    F_diff = np.linalg.norm(F - F_next, ord = "fro") /np.linalg.norm(F, ord = "fro")
                F = F_next

                temp = G @ F.T @ F
                P = divide(X.T @ F, temp)
                G_next = G * P
                if count == 0:
                    G_diff = np.linalg.norm(G - G_next, ord = "fro") /np.linalg.norm(G, ord="fro")
                G = G_next
                
                count += 1
                if count >= period:
                    count = 0
            return G, F
