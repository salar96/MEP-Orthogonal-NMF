#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
import numpy.matlib as matlib
import pandas as pd
import warnings
from sklearn.preprocessing import normalize
from scipy.linalg import eigh
import plotly.express as px
import time
from scipy.spatial.distance import cdist
import random
def has_changed(x,y,tol=1e-3):
    return np.linalg.norm(x-y)/np.linalg.norm(x)>tol
def has_stabled(x,y,tol=1e-3):
    return np.linalg.norm(x-y)/np.linalg.norm(x)<tol

def divide (X, l):
    out = np.copy(X)
    np.divide(out, l, where = np.logical_not(np.isclose(l, 0)), out = out)
    return out

def flatten(t):
    '''
    this finction is used for the purpose of flattening the Betas
    and data points to be ploted in the animation
    '''
    return [item for sublist in t for item in sublist]

def bifurcate(D,x,y,p,px,py,purturb):
    R=np.einsum('ij,ij->i',px*D,p)
    i=np.argmax(R)
    I=np.argpartition(p[i,:], kth=p.shape[1]//2, axis=-1)[p.shape[1]//2:]
    direction=np.matmul(np.multiply(x[:,I],px[I]),p[i,I].T)
    direction=purturb*np.sqrt(R[i])*direction/np.linalg.norm(direction)
    u=y[:,i][:,None]+direction
    # u=y[:,i][:,None]+(np.random.uniform(-purturb, purturb, size=y.shape))
    py_d=py.copy()
    py_d[i]=py_d[i]/2
    return np.append(y,u[:,0].reshape(-1,1),axis=1),np.append(py_d,py_d[i])


class DA:
    def __init__(self,K,tol=1e-4,max_iter=1000,alpha=1.05,
              purturb=0.01,beta_final=None,verbos=0,normalize=False):
        '''
         K: number of clusters norm: the type of norm applied as the distance measure
         tol: The relative tolerence applied as the stopping criteria
         max_iter: maximum number of iterations applied as the stopping criteria
         purturb: the magnitude of purturbation after each split
         beta_final: the maximum value of beta before the algorithm stops. If None,
         then it is detected automatically.
        '''
        self.K=K;self.tol=tol;self.max_iter=max_iter;
        self.alpha=alpha;self.purturb=purturb;self.beta_final=beta_final
        self.VERBOS=verbos;self.norm='l2';self.normalize=normalize

    #___________________Fitting the model on data___________________

    def fit(self,X,**kwargs):
        '''
         **kwargs:
         px: data points probability array. If 'auto', it is calculated automatically.
        '''
        flag=False
        m, self.n = np.shape(X)
        self.d=m
        l = np.sum(np.square(X), axis=0, keepdims=True);l_sqrt = np.sqrt(l);
        weights=l[0]
        weights=weights/weights.sum()
        if self.normalize:
            self.X = divide (X, l_sqrt)
        else:
            self.X=X;     
        if 'Px' in kwargs:
            if kwargs['Px']=='auto':
                self.Px=weights
                flag=True
            else:
                self.Px=kwargs['Px']
        else:
            self.Px=np.array([1 for i in range(self.n)])/self.n
        if not (flag == self.normalize):
            print('!!--normalization without auto weighting on Px--!!')
        self.Y=np.repeat((self.X@self.Px).reshape(m,1),1,axis=1)
        self.Py=np.array([1]) #all the points belong to this cluster



    def cluster(self,express=True):
            y_list=[] # a list to save codevectors over betas
            beta_list=[];Beta=0; # list of all betas
            y_list.append(self.Y);beta_list.append(0)    
            k_n=1 # at first we just have one codevector
            Y_old=np.random.rand(self.d,k_n*2)*1e6;P_old=np.random.rand(2,2);y_old=Y_old
            beta_devide=[] #list to store critical betas
            if not express:
              Beta=0.45/np.max(np.cov(self.X))
              changed=True;check_change=False
            if self.VERBOS:
                print(f'Classification started at Beta:{Beta}')
            while True: #beta loop
                counter=1
                while True: #y p loop
                    D=cdist(self.X.T, self.Y.T, metric='sqeuclidean').T
                    Dn=np.subtract(D,np.min(D,axis=0))
                    if express:
                      P=np.float64(Dn==0)
                      self.Py=np.dot(P,self.Px)
                      Beta+=1
                    else:
                      beta_list.append(Beta)   
                      delta=np.exp(-Dn*Beta)
                      counter2=1
                      while True:  
                          p=(delta.T*self.Py).T
                          P=p/p.sum(axis=0)
                          self.Py=P@self.Px
                          if P.shape==P_old.shape:
                              if has_stabled(P_old,P,1e-3):
                                  break
                          if counter>self.max_iter:
                              warnings.warn("MAX ITER REACHED: py LOOP")
                              break
                          P_old=P
                          counter2=counter2+1
                    self.Y=np.divide(np.matmul(np.multiply(self.X,self.Px),P.T),self.Py)
                    if self.Y.shape==Y_old.shape:
                        if has_stabled(Y_old,self.Y,self.tol):
                            break
                    if counter>self.max_iter:
                        warnings.warn("MAX ITER REACHED: Y LOOP")
                        break
                    Y_old=self.Y
                    counter=counter+1
                if not express:
                  com=(np.count_nonzero(np.abs(P-1)==0)/self.n)
                else:
                  com=1.0
                beta_list.append(Beta)
                y_list.append(self.Y)
                if self.VERBOS:
                  cost=np.linalg.norm(self.X-(self.Y@P))/np.linalg.norm(self.X)
                if (not (self.beta_final is None)) and Beta<1e100:
                  if Beta>self.beta_final and k_n==self.K:
                    print(f"Beta Max reached: {Beta} completeness:{com}")
                    Beta=1e100
                else:
                  if (1.0-com)<1e-18 and k_n==self.K: 
                    break     
      
                if self.VERBOS:
                    print(f'Beta: {Beta} cost:{cost}')
                if express:
                  self.Y,self.Py=bifurcate(D,self.X,self.Y,P,self.Px,self.Py,self.purturb)
                  k_n=self.Y.shape[1]
                else:
                  if check_change:
                      changed=has_changed(y_old,self.Y,self.tol)
                      if changed:
                          check_change=False
                  stabled=has_stabled(y_old,self.Y,self.tol)       
                  if not express:
                    Beta=Beta*self.alpha
                  if k_n<self.K:
                      if changed:
                          if stabled:
                              self.Y,self.Py=bifurcate(D,self.X,self.Y,P,self.Px,self.Py,self.purturb)
                              if self.VERBOS:
                                  print(f"\nDevision occured: to {self.Y.shape[1]} at {Beta}\n")
                              k_n=self.Y.shape[1]
                              check_change=True
                              beta_devide.append(Beta)
                y_old=self.Y               
            self.y_list=y_list
            self.beta_list=beta_list
            self.beta_devide=beta_devide
            self.P=np.round(P)

            return self.Y,self.P
    def plot(self,size=(12,10)):
        assert self.X.shape[0]==2,f'Can only plot 2-D data points, but the dimension here is {self.X.shape[0]}.'
        plt.figure(figsize=size)
        plt.scatter(self.X[0,:],self.X[1,:],marker='.',color='black')
        plt.scatter(self.Y[0,:],self.Y[1,:],marker='*',c='red',linewidths=2)
    def return_cost(self):
        return np.linalg.norm(self.X-(self.Y@self.P),'fro')/np.linalg.norm(self.X,'fro')

    def pie_chart(self,figsize=(6,8)):
        labels=['Y_'+str(i) for i in range(self.Py.shape[0])]
        explode=np.zeros_like(self.Py)
        explode[np.argmin(self.Py)]=0.5
        fig1, ax1 = plt.subplots(figsize=figsize)
        ax1.pie(self.Py, explode=explode,labels=labels,autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')
        ax1.set_title("Cluster Masses %")
        plt.show()
    def animation(self):    
        xx=np.array(flatten([list(i[0,:]) for i in self.y_list]))
        yy=np.array(flatten([list(i[1,:]) for i in self.y_list]))
        l=len(xx)
        Betas=[]
        ids=[]
        r=np.linalg.norm(self.Y)**0.2
        for i , y in enumerate(self.y_list):
            for j in range(y.shape[1]):
                Betas.append(np.round(self.beta_list[i],2))
                ids.append(int(j+1))
        data={'$Y_x$':xx,'$Y_y$':yy,'Beta':Betas,'cluster number':ids}
        df=pd.DataFrame(data=data)
        fig=px.scatter(df,x='$Y_x$', y='$Y_y$',animation_frame='Beta',
                    log_x=False,title='Cluster Centers over beta Values',range_x=[np.min(xx)-r,np.max(xx)+r],
                    range_y=[np.min(yy)-r,np.max(yy)+r]
                    )
        fig.show()
    def mesh_plot(self,size=(12,10)):
      plt.figure(figsize=size)
      for i in range(self.K):
          J=np.where(1-self.P[i,:]<=1e-5)
          plt.scatter(self.X[0,J],self.X[1,J],color=np.random.rand(3))
    def plot_criticals(self,log=False,size=(6,4)):
        plt.figure(figsize=size)
        if log:
            plt.scatter(range(1,self.K),np.log(self.beta_devide));plt.grid();plt.xticks(np.arange(1, self.K, 1.0))
            plt.ylabel('log(Beta)')
        else:
            plt.scatter(range(1,self.K),(self.beta_devide));plt.grid();plt.xticks(np.arange(1, self.K, 1.0))
            plt.ylabel('Beta')
        plt.title('Critical Betas over number of clusters')
        plt.xlabel('Cluster Number')
    def return_true_number(self):
      return np.argmax([(self.beta_devide[i+1]/self.beta_devide[i]) for i in range(len(self.beta_devide)-1)])+1
