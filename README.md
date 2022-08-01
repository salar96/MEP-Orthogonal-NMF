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
For example, for a dataset that consists of 16 psudu-symmetric clusters as illustrated if Fig 1, the DA finds the cluster centers:
```python
{ 
  model.plot()
}
```
![index](https://user-images.githubusercontent.com/50495107/182255163-d78a7d72-ea34-4a4f-ba32-5afc1fbfcd38.png)

Where in this animation below, you can see how by increasing the value of $\beta$ these cluster centers split. In the beginning, there is only one cluster center, and as the value of $\beta$ reaches a critical value for a cluster center, that cluster center will bifurcate, Here animation_frame is equivalent to $\beta$ values.
```python
{ 
  model.animation()
}
```
![ezgif com-gif-maker(1)](https://user-images.githubusercontent.com/50495107/182254523-c07d2473-0a44-4261-b90f-74c6b022b1d7.gif)

These critical $\beta$ values provide useful information about our dataset. By plotting them using the command below, we can determine the true number of clusters in our dataset:

```python
{ 
  model.plot_criticals()
}
```
![index](https://user-images.githubusercontent.com/50495107/182256886-e245ce07-2e2e-4fa5-9515-38abd7bbfef4.png)

By looking at this diagram, we can see that there are large gaps at 2,4,8 and 16 between these critical $\beta$ values. These show that, depending at the resolution you want to look at your dataset with, there are 2,4,8 or 16 clusters there. However, the largest gap occurs at 16, so it means that there are 16 clusters in this dataset, that can be taken as the true number of clusters.



