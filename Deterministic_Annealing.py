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

def has_changed(x,y,tol=1e-3):
    return np.linalg.norm(x-y)/np.linalg.norm(x)>tol
def has_stabled(x,y,tol=1e-3):
    return np.linalg.norm(x-y)/np.linalg.norm(x)<tol
def divide (N, P):
    out = np.copy(N)
    np.divide(out, P, where = np.logical_not(np.isclose(P, 0)), out = out)
    return out

def flatten(t):
    '''
    this finction is used for the purpose of flattening the Betas
    and data points to be ploted in the animation
    '''
    return [item for sublist in t for item in sublist]

def bifurcate(D,x,y,p,px,py,beta,purturb):
    R=np.einsum('ij,ij->i',px*D,p)
    i=np.argmax(R)
    I=np.argsort(p[i,:])[int(p.shape[1]/2):p.shape[1]]
    direction=(x[:,I]*px[I])@p[i,I].T
    direction=direction/(np.sum(direction*direction,axis=0)**0.5)
    PURTURB=purturb*np.sqrt(np.max(R))*direction
    u=y[:,i][:,None]+PURTURB
    #y_h=np.insert(y,i,u[:,0],axis=1)
    y_h=np.append(y,u[:,0].reshape(-1,1),axis=1)
    py_d=py.copy()
    py_d[i]=py_d[i]/2
    py_h=np.append(py_d,py_d[i])
    return y_h,py_h


class DA:
    def __init__(self,K,norm='L2',tol=1e-4,max_iter=1000,alpha=1.05,
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
        self.VERBOS=verbos;self.norm=norm;self.normalize=normalize

    #___________________Fitting the model on data___________________

    def fit(self,X,**kwargs):
        '''
         **kwargs:
         px: data points probability array. If 'auto', it is calculated automatically.
        '''
        flag=False
        m, self.n = np.shape(X)
        self.d=m
        l = np.sum(X * X, axis = 0, keepdims = True);l_sqrt = np.sqrt(l);
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
        self.BETA_INIT=0.45/np.max(np.cov(self.X))


    def cluster(self):
        y_list=[] # a list to save codevectors over betas
        beta_list=[] # list of all betas
        y_list.append(self.Y);beta_list.append(0)    
        k_n=1 # at first we just have one codevector
        Y_old=np.random.rand(self.d,k_n*2)*1e6;P_old=np.random.rand(2,2);y_old=Y_old
        beta_devide=[] #list to store critical betas
        Beta=self.BETA_INIT
        changed=True;check_change=False
        if self.VERBOS:
            print(f'Classification started at Beta:{Beta}')
        while True: #beta loop
            counter=1
            while True: #y p etha loop
                self.Data_points=np.repeat(self.X[:,:,np.newaxis],self.Y.shape[1],axis=2)
                Cluster_points=np.repeat(self.Y[:,np.newaxis,:],self.n,axis=1)
                if self.norm=='L2':
                    D=(self.Data_points-Cluster_points)**2
                else:
                    raise Exception("Wrong norm!")
                D=np.sum(D,axis=0).T
                Dn=D-np.min(D,axis=0)
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
                self.Y=((self.X*self.Px)@P.T)/(self.Py)
                if self.Y.shape==Y_old.shape:
                    if has_stabled(Y_old,self.Y,self.tol):
                        break
                if counter>self.max_iter:
                    warnings.warn("MAX ITER REACHED: Y LOOP")
                    break
                Y_old=self.Y
                counter=counter+1
            com=(np.count_nonzero(np.abs(P-1)==0)/self.n)
            beta_list.append(Beta)
            y_list.append(self.Y)
            if self.VERBOS:
              cost=np.linalg.norm(self.X-(self.Y@P))/np.linalg.norm(self.X)
            if (not (self.beta_final is None)) and Beta<1e100:
              if Beta>self.beta_final and k_n==self.K:
                print(f"Beta Max reached: {Beta} completeness:{com}")
                Beta=1e100
            else:
              if (1-com)<1e-18 and k_n==self.K: 
                print(f"stopping criteria met at Beta: {Beta}")
                break     
            if self.VERBOS:
                print(f'Beta: {Beta} cost:{cost}')
            if check_change:
                changed=has_changed(y_old,self.Y,self.tol)
                if changed:
                    check_change=False
            stabled=has_stabled(y_old,self.Y,self.tol)       
            Beta=Beta*self.alpha
            if k_n<self.K:
                if changed:
                    if stabled:
                        self.Y,self.Py=bifurcate(D,self.X,self.Y,P,self.Px,self.Py,Beta,self.purturb)
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
        Betas=[]
        for i , y in enumerate(self.y_list):
            for j in range(y.shape[1]):
                Betas.append(self.beta_list[i])
        df = px.data.gapminder()
        fig=px.scatter(x=xx, y=yy,animation_frame=Betas,
                   log_x=False)
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

