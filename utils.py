import numpy as np

def divide (N, P):
    out = np.copy(N)
    np.divide(out, P, where = np.logical_not(np.isclose(P, 0)), out = out)
    return out


def normalize(M, axis):
    M_sq = M * M
    M_norm = np.sqrt(np.sum(M_sq, axis = axis, keepdims = True))
    ret = np.copy(M)
    np.divide(ret, M_norm, where = np.logical_not(np.isclose(M_norm, 0)), out = ret)
    return ret
