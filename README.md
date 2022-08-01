# MCDA-ONMF
Mass-Constrained Deterministic Annealing Orthogonal Non-negative Matrix Factorization

This code is presented as supplimentary material for our paper:
"Orthogonal Non-negative Matrix Factorization: a Deterministic Annealing Approach"

------------------------------------------------------------
# Deterministic Annealing
Deterministic Annealing (DA) is a clustering, or in a more accurate definition, a facility allocation algorithm that clusters data points into several groups of different sizes, so that the cumulative distance of each data point to its assigned resource is minimized (for details on this algorithm, please refere to our paper mentioned above). To cluster data into $k$ clusters, just use the code snippet here:


| parameter      | description |
| ----------- | ----------- |
| k      | number of clusters we want to partition our data in       |
| tol   | The tolerence at which bifurcation happens for the critical cluster - smaller values more stable, but results in a slower performance        |
| max_iter   | maximum number of iterations thatthe code waits until the convergence of parameters        |
| alpha   | the value at which $\beta$ is multiplied by at the end of each convergence. Smaller values results in a more stable performance, but slows down the code        |
| purturb   | the rate at which we purturb new cluster centers. Too large or too small values results in instability        |
| beta_final   | the final $\beta$ value that we want to stop the algorithm at. If not specified, it is determined automatically        |
| verbos   | whether to print out iterations' information         |
| normalize   | whether to 'l2' normalize the input data        |

| input   | description        |
| ----------- | ----------- |
| X | Data matrix $X \in \mathbb{R}^{d \times n}$ where $d$ is each data point's dimension and $n$ is the number of data points|


| output   | description        |
| ----------- | ----------- |
| Y | cluster centers (resource locations) $Y \in \mathbb{R}^{d \times k}$ |
| P | associations matrix $P \in \mathbb{R}^{k \times n}$ determining the belonging of each data point to each cluster center (resource)|

```python
{ 
  from Deterministic_Annealing import DA
  model=DA(K,tol=1e-4,max_iter=1000,alpha=1.05,
              purturb=0.01,beta_final=None,verbos=0,normalize=False)
  model.fit(X,Px='auto')
  Y,P=model.cluster()
}
```
