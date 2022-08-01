# MCDA-ONMF
Mass-Constrained Deterministic Annealing Orthogonal Non-negative Matrix Factorization

This code is presented as supplimentary material for our paper:
"Orthogonal Non-negative Matrix Factorization: a Deterministic Annealing Approach"

------------------------------------------------------------
# Deterministic Annealing
Deterministic Annealing (DA) is a clustering, or in a more accurate definition, a facility allocation algorithm that clusters data points into several groups of different sizes, so that the cumulative distance of each data point to its assigned resource is minimized. To cluster data into k clusters, just use the code snippet here:


```python
{ 
  from Deterministic_Annealing import DA
  model=DA(k,alpha=alpha,purturb=purturb,verbos=verbos,normalize=normalize,tol=tol)
  model.fit(X,Px='auto')
  Y,P=model.cluster()
}
```
