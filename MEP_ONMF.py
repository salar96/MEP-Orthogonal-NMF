import numpy as np
from Deterministic_Annealing import DA

class ONMF_DA:
    name = "ONMF-DA"
    def func (X, k,alpha=2,purturb=0.0001,verbos=0,normalize=True,auto_weighting=True,tol=1e-3,express=True):
        m, n = np.shape(X)
        model=DA(k,alpha=alpha,purturb=purturb,verbos=verbos,normalize=normalize,tol=tol)
        if auto_weighting:
            model.fit(X,Px='auto');
        else:
            model.fit(X);
        Y,P=model.cluster(express=express)
        w=np.round(P.copy())
        for i in range(n):
            id=np.where(w[:,i]==1)[0]
            w[id,i]=np.divide(np.matmul(X[:,i].T , Y[:,id]),np.square(np.linalg.norm(Y[:,id])))

        return Y, w.T,model
