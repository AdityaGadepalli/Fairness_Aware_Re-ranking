import numpy as np
import scipy
import pandas as pd
import math
from tqdm import tqdm
from fair_algortihms import DetGreedy, DetCons, DetRelax, DetConstSort
from metrics import *

class experiment:
    """ 
    This is the simulation framework for generating the data required to run the algorithms.
    It takes in the following variables as user inputs :-
    'max_a' = Maximum cardinality of protected class for which we want to generate synthetic data
    'tasks' = Number of desired probability distributions hypothesized as different queries
    'distribution' = Type of distribution to generate data from. Default-'Uniform' #Scope for Future Improvement
    'doctrine' = The Doctrine of Fairness to be ensured. Default- 'DI' #Scope for Future Improvement
    'datapoints' = The number of synthetic Scores to generate for different 'ai' values #Scope for Future Improvement
    """
    
    def __init__(self,max_a,tasks,rep,distribution,doctrine):
        self.max_a=max_a
        self.tasks=tasks
        self.rep=rep
        self.distribution=distribution
        self.doctrine=doctrine
        self.datapoints=100
    
    def setup(self):
        self.P = {}
        self.data={}
        for a in range (2,self.max_a):
            self.P[f'a={a}']=self.get_tasks(a)
            for i in self.P[f'a={a}']:
                for j in range(self.rep): 
                    self.data[(f'a={a}',i,j)] = self.gen_data(a)
        return self.P

    def get_tasks(self,a):
        T={}
        for i in range(self.tasks):
            T[i]=self.get_distribution(a)
        return T

    def get_distribution(self,a):
        d=[]
        for i in range(a):
            d.append(np.random.uniform())
        d_norm = [float(i)/sum(d) for i in d]
        return d_norm

    def gen_data(self,a):
        scores={}
        scoresgen=[]
        for i in range(a):
            for j in range(self.datapoints):
                scoresgen.append(np.random.uniform())
            scoresgen.sort(reverse=True)
            for j in range(self.datapoints):
                scores[(i,j)]=scoresgen[j]
        return scores
    
    def get_value(self,a,i,j):
        return self.data[(f'a={a}',i,j)]

class re_ranking:
    """ 
    This is the Reranking framework for applying the fair algorithms on the simulated ranked List and measure 
    different fairness metrics that are defined in the paper. The inputs for this are:-
    'demo' = Already simulated Experiment Object
    'a' = Cardinality of the protected attribute class
    """
    def __init__(self,demo,a):
        self.demo =demo
        self.a=a

        self.Min_skew_vanilla=[]
        self.Min_skew_DetGreedy=[]
        self.Min_skew_DetCons=[]
        self.Min_skew_DetRelax=[]
        self.Min_skew_DetConstSort=[]

        self.Inf_ind_vanilla=[]
        self.Inf_ind_DetGreedy=[]
        self.Inf_ind_DetCons=[]
        self.Inf_ind_DetRelax=[]
        self.Inf_ind_DetConstSort=[]

        self.ndcg_vanilla=[]
        self.ndcg_DetGreedy=[]
        self.ndcg_DetCons=[]
        self.ndcg_DetRelax=[]
        self.ndcg_DetConstSort=[]

        self.ndkl_vanilla=[]
        self.ndkl_DetGreedy=[]
        self.ndkl_DetCons=[]
        self.ndkl_DetRelax=[]
        self.ndkl_DetConstSort=[]


    def output(self,a,i,j,demo):
        data = demo.get_value(a,i,j)
        df = pd.Series(data).rename_axis(['ai', 'index']).reset_index(name='score')
        df.sort_values(by=['score'], inplace=True,ascending=False)
        df = df.reset_index(drop=True)
        Algo1=DetGreedy(data,self.demo.P[f'a={a}'][i],100)
        Algo2=DetCons(data,self.demo.P[f'a={a}'][i],100)
        Algo3=DetRelax(data,self.demo.P[f'a={a}'][i],100)
        Algo4=DetConstSort(data,self.demo.P[f'a={a}'][i],100)
   
        return df, Algo1, Algo2, Algo3, Algo4        
        

    def compute_metrics(self):

        for i in tqdm(range(self.demo.tasks)):
            for j in range(self.demo.rep):
                df, Algo1, Algo2, Algo3, Algo4 = self.output(self.a,i,j,self.demo) 

                self.Min_skew_vanilla.append(MinSkew(df['ai'],self.demo.P[f'a={self.a}'][i],100))
                self.Min_skew_DetGreedy.append(MinSkew(Algo1['ai'],self.demo.P[f'a={self.a}'][i],100))
                self.Min_skew_DetCons.append(MinSkew(Algo2['ai'],self.demo.P[f'a={self.a}'][i],100))
                self.Min_skew_DetRelax.append(MinSkew(Algo3['ai'],self.demo.P[f'a={self.a}'][i],100))
                self.Min_skew_DetConstSort.append(MinSkew(Algo4['ai'],self.demo.P[f'a={self.a}'][i],100))

                self.Inf_ind_vanilla.append(infeasibleIndex(df,self.demo.P[f'a={self.a}'][i])[0])
                self.Inf_ind_DetGreedy.append(infeasibleIndex(Algo1,self.demo.P[f'a={self.a}'][i])[0])
                self.Inf_ind_DetCons.append(infeasibleIndex(Algo2,self.demo.P[f'a={self.a}'][i])[0])
                self.Inf_ind_DetRelax.append(infeasibleIndex(Algo3,self.demo.P[f'a={self.a}'][i])[0])
                self.Inf_ind_DetConstSort.append(infeasibleIndex(Algo4,self.demo.P[f'a={self.a}'][i])[0])

                self.ndcg_vanilla.append(ndcg_at_k(df['score'],100))
                self.ndcg_DetGreedy.append(ndcg_at_k(Algo1['score'],100))
                self.ndcg_DetCons.append(ndcg_at_k(Algo2['score'],100))
                self.ndcg_DetRelax.append(ndcg_at_k(Algo3['score'],100))
                self.ndcg_DetConstSort.append(ndcg_at_k(Algo4['score'],100))
                   
                self.ndkl_vanilla.append(NDKL(df['ai'],self.demo.P[f'a={self.a}'][i]))
                self.ndkl_DetGreedy.append(NDKL(Algo1['ai'],self.demo.P[f'a={self.a}'][i]))
                self.ndkl_DetCons.append(NDKL(Algo2['ai'],self.demo.P[f'a={self.a}'][i]))
                self.ndkl_DetRelax.append(NDKL(Algo3['ai'],self.demo.P[f'a={self.a}'][i]))
                self.ndkl_DetConstSort.append(NDKL(Algo4['ai'],self.demo.P[f'a={self.a}'][i]))