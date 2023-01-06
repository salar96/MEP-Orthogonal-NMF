# Implemented based on the following paper:
# Filippo Pompili, Nicolas Gillis, P-A Absil, and Francois Glineur. Two algorithms for orthogonal nonnegative matrix factorization with application to clustering. Neurocomputing, 141:15–25, 2014.

import numpy as np
from utils import divide, normalize

class ONMF_EM:
    name = "EM-ONMF"
    def func (X,k):
        m,n = np.shape(X)
        asgn_list, centers = ONMF_EM.spherical_k_means (X, k)
        W = np.zeros((k,n))
        for i in range(k):
            for j in asgn_list[i]:
                W[i,j] = centers[:, i].T @ X[:, j]
        return centers, W.T


    def spherical_k_means (X, k):
        m,n = np.shape(X)

        asgn_list = [[] for _ in range(k)]
        for i in range(n):
            asgn_list[np.random.randint(k)].append(i)

        asgn = []
        converged = False
        while not converged:
            centers = np.random.rand(m,k)
            centers = normalize(centers, 0)
            for i in range(k):
                if (len(asgn_list[i])):
                    subX = X[:, asgn_list[i]]
                    u, s, vh = np.linalg.svd(subX, full_matrices=False, compute_uv=True)
                    centers[:, i] = np.abs(u[:, 0])
                else:

                    r = np.random.randint(n)
                    centers[:, i] = divide(X[:, r], np.linalg.norm(X[:, r]))

            old_asgn = asgn
            asgn_list = [[] for _ in range(k)]
            dots = X.T @ centers
            dots_max = np.max(dots, axis = 1, keepdims = True)
            asgn = np.argmax(dots, axis = 1)
            asgn = np.argmax(np.isclose(dots, dots_max) * np.random.random((n, k)), axis = 1)
            if len(old_asgn):
                for i in range(n):
                    if np.isclose(dots[i][old_asgn[i]],dots_max[i]):
                        asgn[i] = old_asgn[i]
            for i in range(n):
                asgn_list[asgn[i]].append(i)
            if ONMF_EM.same_asgn(asgn, old_asgn):
                converged = True
        return asgn_list, centers

    def same_asgn(a, b):
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if a[i] != b[i]:
                return False
        return True
