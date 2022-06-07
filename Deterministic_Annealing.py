#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
from matplotlib import pyplot as plt
import numpy.matlib as matlib
import warnings
import timeit as dt
import pandas as pd
from sklearn.preprocessing import normalize
from utils import divide
from scipy.linalg import eigh
def a_split(a):
    
    if a.shape[1]==2:
        ss=[a]
    else:
        ss=np.hsplit(a, int(a.shape[1]/2))
    
    l=[np.linalg.norm(s[:,1]-s[:,0])/np.linalg.norm(s[:,0])<1e-1 for s in ss]
    
    for index , obj in enumerate(l):
        if obj==False:
            ss[index]=np.repeat(ss[index],2,axis=1); 
            
    out=np.concatenate(ss,axis=1)
    return out




class DA:
    def __init__(self,K,NORM='L2',TOL=1e-4,MAX_ITER=1000,GROWTH_RATE=1.05,
               PURTURB_RATIO=0.01,BETA_INIT=1e-6,BETA_TOL=1e-6,verbos=0,NORMALIZE=False):
        # K: number of clusters Norm: the type of norm applied as the distance measure
        # TOL: The relative tolerence applied as the stopping criteria
        # Max_ITER: maximum number of iterations applied as the stopping criteria
        self.K=K;self.TOL=TOL;self.MAX_ITER=MAX_ITER;self.BETA_INIT=BETA_INIT
        self.GROWTH_RATE=GROWTH_RATE;self.PURTURB_RATIO=PURTURB_RATIO;self.BETA_TOL=BETA_TOL
        self.CONSTRAINED=False;self.VERBOS=verbos;self.NORM=NORM;self.NORMALIZE=NORMALIZE
        print('DA model was created successfully')

    #___________________Fitting the model on data___________________

    def fit(self,X,**kwargs):
        # **kwargs:
        # px: data points probability array
        # lambdas: for the constrained codevectors probability array
        m, self.n = np.shape(X)
        self.X=X
        self.d=m
        if (self.X<0).any() and self.NORM=='KL':
            raise Exception('Your input matrix contains negative values. Try using another norm')
        
        if 'Px' in kwargs:
            self.Px=kwargs['Px']
        else:
            self.Px=np.array([1 for i in range(self.n)])/self.n
        self.Y=np.repeat((self.X@self.Px).reshape(m,1),2,axis=1)
        
        if 'Lambdas' in kwargs:
            self.CONSTRAINED=True
            self.Lambdas=kwargs['Lambdas']
            if np.sum(self.Lambdas) != 1.0:
                warnings.warn('Lambdas should sum up to one')
        else:
            self.Lambdas=np.array([1 for i in range(self.K)])/self.K
        self.Ethas=self.Lambdas
        ################### dealing with beta_init here
        #self.BETA_INIT=0.00000001/(2*np.max(np.abs(eigh(np.cov(X), eigvals_only=True))))

        print('Class DA fitted on DATA')
    
    def classify(self):
        end_break=0
        y_list=[] # a list to save codevectors over betas
        beta_list=[] # list of all betas
        y_list.append(self.Y);beta_list.append(0)
        start=dt.default_timer()
        
        k_n=1 # at first we just have one codevector
        self.Py=np.array([0.5,0.5]) #all the points belong to this cluster
        Y_old=np.random.rand(self.d,k_n*2)*1e6
        P_old=np.random.rand(2,2)
        Beta=self.BETA_INIT
        START_OK=0
        end=0
        cost_l=[]
        patience=5
        while True: #beta loop
            
            counter=1
            while True: #y p etha loop
                
                self.Data_points=np.repeat(self.X[:,:,np.newaxis],self.Y.shape[1],axis=2)
                Cluster_points=np.repeat(self.Y[:,np.newaxis,:],self.n,axis=1)
                if self.NORM=='KL':
                    d=np.log(Cluster_points/(self.Data_points+1e-6))
                    #d[np.where(d==-np.inf)]=0
                    D=d*Cluster_points-Cluster_points+self.Data_points
                elif self.NORM=='L2':
                    D=(self.Data_points-Cluster_points)**2
                else:
                    raise Exception("Wrong Norm!")
                D=np.sum(D,axis=0).T
                
                counter2=1
                if self.CONSTRAINED:
                    print('not ok')
                else:
                    
                    p=np.exp(-D*Beta)
                    if not START_OK:
                        print(f"beta init:{self.BETA_INIT} with com:{np.count_nonzero(np.abs(p-1)<1e-5)/(self.K*self.n)}")
                        START_OK=1
                    J=np.where(p.sum(axis=0)==0)[0]
                    #I=np.argmin(D[:,J],axis=0)
                    #p[I,J]=[1 for i in range(len(J))]
                    I=np.min(D[:,J],axis=0)
                    p[:,J]=np.logical_not(D[:,J]-matlib.repmat(I,D.shape[0],1)).astype(int)
                    
                    #p=(p.T*self.Py).T
                    P=p/p.sum(axis=0)
                    self.Py=P@self.Px
                    
                    if self.NORM=='KL':
                        self.Y=np.exp(((np.log(self.X+1e-6)*self.Px)@P.T)/(Py+1e-6))
                    elif self.NORM=='L2':
                        self.Y=((self.X*self.Px)@P.T)/(self.Py)
                    else:
                        raise Exception("Wrong Norm!")
                    
                    if self.Y.shape==Y_old.shape:
                        if np.linalg.norm(self.Y-Y_old)/np.linalg.norm(Y_old)<self.TOL:
                            break
                    if counter>self.MAX_ITER:
                        warnings.warn("MAX ITER REACHED: Y LOOP")
                        break
                    Y_old=self.Y
                    counter=counter+1
            
 
            com=(np.count_nonzero(np.abs(P-1)<1e-5)/self.n)
            beta_list.append(Beta)
            y_list.append(self.Y)
            if Beta>self.BETA_TOL:
            #if (1-com)<self.BETA_TOL:
                time=dt.default_timer()-start
                print(f"Beta Max reached: {Beta} completeness:{com} time:{time}")
                #break
           
            Beta=Beta*self.GROWTH_RATE
            
            ###################################################
            cost=np.linalg.norm(self.X-(self.Y@P))/np.linalg.norm(self.X)
            cost_l.append(cost)
            if self.VERBOS:
                print(f'Beta: {Beta} cost:{cost}')
            if not end_break:
                self.Y=a_split(self.Y)
            if self.Y.shape[1]/2>=self.K and not end_break:
                end_break=1
                print(f"reached the specified number of clusters: {self.Y.shape[1]/2}")
            if end_break:
                if len(cost_l)>=patience:
                    if max(cost_l[-patience:])-min(cost_l[-patience:]) <=1e-4:
                        break
                
            #print(f'number of codevectors is {self.Y.shape[1]/2}',Beta)
            """
            if len(set(np.round((self.Y*self.Y).sum(axis=0),1)))>self.K:
                print('setting the final stage')
                #self.Y=self.Y[:,[i for i in range(self.Y.shape[1]) if i%2==0]]
                self.Y=return_uniq(self.Y)
                end=1
                Beta=1e2
             """ 
   
            ###################################################            

            #___________________PURTURBATION__________________________

            #PURTURB=self.PURTURB_RATIO*(np.random.rand(self.d,self.K)-0.5)*(np.max(self.X)-np.min(self.X))
            sp=np.array_split(self.X, self.Y.shape[1],axis=1)
            PURTURB=self.PURTURB_RATIO*np.vstack([np.mean(i,axis=1) for i in sp]).T
            self.Y=self.Y+PURTURB
            #_________________________________________________________
            
        print(f'finished,{self.Y.shape[1]/2}')
        self.P=P
        return self.Y,self.P,beta_list,y_list
    def plot(self,size=(12,10),random_color=False):
        plt.figure(figsize=size)
        plt.scatter(self.X[0,:],self.X[1,:],marker='.');plt.grid()
        plt.scatter(self.Y[0,:],self.Y[1,:],marker='*',c='red',linewidths=2)
    def return_cost(self):
        return np.linalg.norm(self.X-(self.Y@self.P),'fro')/np.linalg.norm(self.X,'fro')

    
    
if __name__=='__main__':
    X=[]
    for i in range(100):
        X.append([0+(-1)**(int(np.random.rand()))*np.random.rand(),4+(-1)**(int(np.random.rand()))*np.random.rand()])
        X.append([1.5+(-1)**(int(np.random.rand()))*np.random.rand(),2+(-1)**(int(np.random.rand()))*np.random.rand()])
        X.append([-1.5+(-1)**(int(np.random.rand()))*np.random.rand(),2+(-1)**(int(np.random.rand()))*np.random.rand()])
        X.append([-3+(-1)**(int(np.random.rand()))*np.random.rand(),-3+(-1)**(int(np.random.rand()))*np.random.rand()])
        X.append([-4.5+(-1)**(int(np.random.rand()))*np.random.rand(),-5+(-1)**(int(np.random.rand()))*np.random.rand()])
        X.append([-1.5+(-1)**(int(np.random.rand()))*np.random.rand(),-5+(-1)**(int(np.random.rand()))*np.random.rand()])
        X.append([3+(-1)**(int(np.random.rand()))*np.random.rand(),-3+(-1)**(int(np.random.rand()))*np.random.rand()])
        X.append([4.5+(-1)**(int(np.random.rand()))*np.random.rand(),-5+(-1)**(int(np.random.rand()))*np.random.rand()])
        X.append([1.5+(-1)**(int(np.random.rand()))*np.random.rand(),-5+(-1)**(int(np.random.rand()))*np.random.rand()])
    X=(np.vstack(X).T+10)/10
    clus_num=9
    model=DA(clus_num,NORM='KL',GROWTH_RATE=1.5,BETA_INIT=1e-1,BETA_TOL=1e-16,PURTURB_RATIO=0.2,verbos=0)
    model.fit(X);Y,P=model.classify();model.plot()
    print(model.return_cost())
