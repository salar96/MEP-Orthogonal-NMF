# Implemented based on the following paper:
# Zhirong Yang and Jorma Laaksonen. Multiplicative updates for non-negative projections. Neurocomput- ing, 71(1-3):363–373, 2007.

import numpy as np
from utils import divide, normalize

class NHL:
    name = "NHL"
    def func (X,k):
        X = X.T
        m,n = np.shape(X)
        F = np.random.rand(m,k)
        F_diff = 1
        count = 0
        period = 5
        while F_diff > 1e-4:
            temp = X @ X.T @ F
            denominator = F @ F.T @ temp
            P = divide (temp, denominator)
            F_next = F * P
            F_next /= np.linalg.norm(F_next, ord = 2)
            if count == 0:
                F_diff = np.linalg.norm(F - F_next, ord = "fro")/np.linalg.norm(F, ord = "fro")
            F = F_next
            if count == 0:
                print("NHL", F_diff)
            count += 1
            if count >= period:
                count = 0
        # return F, X.T @ F
        return X.T @ F, F
