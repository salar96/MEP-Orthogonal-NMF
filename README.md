# MEP-ONMF
Maximum-Entropy-Principle Orthogonal Non-negative Matrix Factorization

[This code is presented as supplimentary material for our paper]:
["Orthogonal Non-negative Matrix Factorization: a Maximum-Entropy-Principle Approach"](https://arxiv.org/abs/2210.02672v2)
------------------------------------------------------------
# Deterministic Annealing Clustering (Maximum-Entropy-Principle Clustering)
Deterministic Annealing (DA) is a clustering, or in a more accurate definition, a facility location algorithm that clusters data points into several groups of different sizes, so that the cumulative distance of each data point to its assigned resource is minimized (for details on this algorithm, please refere to our paper mentioned above). To cluster data into $k$ clusters, just use the code snippet here:


| parameter      | description |
| ----------- | ----------- |
| k      | number of clusters we want to partition our data in       |
| tol   | The tolerence at which bifurcation happens for the critical cluster - smaller values more stable, but results in a slower performance        |
| max_iter   | maximum number of iterations that the code waits until the convergence of parameters        |
| alpha   | the value at which $\beta$ is multiplied by at the end of each convergence. Smaller values results in a more stable performance, but slows down the code        |
| purturb   | the rate at which we purturb new cluster centers        |
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
  print(model.return_cost())
}
```
For example, for a dataset that consists of 16 psudu-symmetric clusters as illustrated if Fig 1, the DA finds the cluster centers:
```python
{ 
  model.plot()
}
```
<img src="https://user-images.githubusercontent.com/50495107/182255163-d78a7d72-ea34-4a4f-ba32-5afc1fbfcd38.png" width="600" height="600" />

Where in this animation below, you can see how by increasing the value of $\beta$ these cluster centers split. In the beginning, there is only one cluster center, and as the value of $\beta$ reaches a critical value for a cluster center, that cluster center will bifurcate, Here animation_frame is equivalent to $\beta$ values.

```python
{ 
  model.animation()
}
```


https://user-images.githubusercontent.com/50495107/195967277-7348200e-4e6a-44a1-94ca-ed8e9d8aec15.mp4


These critical $\beta$ values provide useful information about our dataset. By plotting the log of these values using the command below, we can determine the true number of clusters in our dataset:

```python
{ 
  model.plot_criticals(log=True)
}
```
![index](https://user-images.githubusercontent.com/50495107/182256886-e245ce07-2e2e-4fa5-9515-38abd7bbfef4.png)

By looking at this diagram, we can see that there are large gaps at 2,4,8 and 16 between these critical $\beta$ values. These show that, depending at the resolution you want to look at your dataset with, there are 2,4,8 or 16 clusters there. However, the largest gap occurs at 16, so it means that there are 16 clusters in this dataset, that can be taken as the true number of clusters. To get an accurate plot, it is advised to use small values for alpha and purturb.
Also using :
```python
{ 
  model.return_true_number()
}
```
returns the value 16 for this dataset.

The command mesh_plot specifies which cluster center the data belong to. For example, for a given dataset of 15 clusters, we get:
```python
{ 
  model.mesh_plot()
}
```
![index](https://user-images.githubusercontent.com/50495107/182267869-cd768f93-43d9-4338-815c-4c171fe0c761.png)

The Mass-Constrained DA allows capturing unbalanced datasets where one or more clusters have significantly lower number of data points compared with the other clusters. The weight of each cluster can be shown using the pie_chart command. This is very useful for outlier detection applications. For example, for a given dataset like:

![index](https://user-images.githubusercontent.com/50495107/182269413-e9d65fe9-5eac-4e09-843d-4086d3c266a7.png)

```python
{ 
  model.pie_chart()
}
```
![index](https://user-images.githubusercontent.com/50495107/182269503-c564b9c9-0fb3-4b18-8d4d-a5eeb1c577f8.png)


The resulting pie chart indicates that the cluster Y_6 is an outlier.

------------------------------------------------------------
# Orthogonal Non-negative Matrix Factorization (ONMF)

The orthogonal NMF (ONMF) poses the following problem:

$$
\begin{align}
    \min \quad &D(X,WH) \\ 
    \textrm{s.t.} \quad &W^{\intercal}W=I \quad \text{or} \quad HH^{\intercal}=I \\ 
    & W_{ij},H_{ij} \geq 0 \quad \forall \ i,j 
\end{align}
$$


DA can also be used to solve the ONMF problem.
| input   | description        |
| ----------- | ----------- |
| X | Data matrix $X \in \mathbb{R}^{d \times n}_{+}$ where $d$ is each data point's dimension and $n$ is the number of data points|


| output   | description        |
| ----------- | ----------- |
| W | Features matrix $W \in \mathbb{R}^{d \times k}_{+}$ |
| H | Mixing matrix $H \in \mathbb{R}^{k \times n}_{+}$ |
| model | DA model associated with the data matrix. Can be used to identify the true number of features |
```python
{ 
  from onmf_DA import ONMF_DA
  W,H,model=ONMF_DA.func(X,k,alpha=alpha,purturb=p,tol=tol)
}
```
For $H$-orthogonal, put $X$ and for $W$-orthogonal, put $X^{\intercal}$ as input.


<img src="https://user-images.githubusercontent.com/50495107/182271297-015ab74c-69d6-4f79-b246-b5de6a709601.png" width="800" height="300" />

