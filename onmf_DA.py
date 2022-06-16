import numpy as np
from utils import divide, normalize
from Deterministic_Annealing import DA

class ONMF_DA:
    name = "ONMF-DA"
    def func (X, k,GROWTH_RATE=2,PURTURB_RATIO=0.0001,patience=0.5,verbos=0,Normalize=True,auto_weighting=True):
        m, n = np.shape(X)
        model=DA(k,GROWTH_RATE=GROWTH_RATE,PURTURB_RATIO=PURTURB_RATIO,verbos=verbos,NORMALIZE=Normalize,PATIENCE=patience)
        if auto_weighting:
            model.fit(X,Px='auto');
        else:
            model.fit(X);
        Y,P=model.classify()
        w=np.round(P.copy())
        for i in range(n):
            id=np.where(w[:,i]==1)[0]
            norm1 = np.linalg.norm(Y[:,id])
            w[id,i]=(X[:,i].T @ Y[:,id])/norm1/norm1

        return Y, w,model

