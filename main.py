import numpy as np
import scipy
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from fair_algortihms import DetGreedy, DetCons, DetRelax, DetConstSort
from metrics import *
from experiment import experiment, re_ranking
#import plot_utlis


def main():
    demo = experiment(10,10,10,'uniform','DI')
    P = demo.setup()
    p=10

    ndcg_plot={}
    ndcg_plot["vanilla"]=[]
    ndcg_plot["DetGreedy"]=[]
    ndcg_plot["DetCons"]=[]
    ndcg_plot["DetRelax"]=[]
    ndcg_plot["DetConstSort"]=[]

    ndkl_plot={}
    ndkl_plot["vanilla"]=[]
    ndkl_plot["DetGreedy"]=[]
    ndkl_plot["DetCons"]=[]
    ndkl_plot["DetRelax"]=[]
    ndkl_plot["DetConstSort"]=[]

    Inf_ind_plot={}
    Inf_ind_plot["vanilla"]=[]
    Inf_ind_plot["DetGreedy"]=[]
    Inf_ind_plot["DetCons"]=[]
    Inf_ind_plot["DetRelax"]=[]
    Inf_ind_plot["DetConstSort"]=[]

    Min_skew_plot={}
    Min_skew_plot["vanilla"]=[]
    Min_skew_plot["DetGreedy"]=[]
    Min_skew_plot["DetCons"]=[]
    Min_skew_plot["DetRelax"]=[]
    Min_skew_plot["DetConstSort"]=[]

    for j in tqdm(range(2,p)):
        rank = re_ranking(demo,j)
        rank.compute_metrics()
        ndcg_plot["vanilla"].append(np.mean(rank.ndcg_vanilla))
        ndcg_plot["DetGreedy"].append(np.mean(rank.ndcg_DetGreedy))
        ndcg_plot["DetCons"].append(np.mean(rank.ndcg_DetCons))
        ndcg_plot["DetRelax"].append(np.mean(rank.ndcg_DetRelax))
        ndcg_plot["DetConstSort"].append(np.mean(rank.ndcg_DetConstSort))
        
        ndkl_plot["vanilla"].append(np.mean(rank.ndkl_vanilla))
        ndkl_plot["DetGreedy"].append(np.mean(rank.ndkl_DetGreedy))
        ndkl_plot["DetCons"].append(np.mean(rank.ndkl_DetCons))
        ndkl_plot["DetRelax"].append(np.mean(rank.ndkl_DetRelax))
        ndkl_plot["DetConstSort"].append(np.mean(rank.ndkl_DetConstSort))
        
        Inf_ind_plot["vanilla"].append(np.mean(rank.Inf_ind_vanilla))
        Inf_ind_plot["DetGreedy"].append(np.mean(rank.Inf_ind_DetGreedy))
        Inf_ind_plot["DetCons"].append(np.mean(rank.Inf_ind_DetCons))
        Inf_ind_plot["DetRelax"].append(np.mean(rank.Inf_ind_DetRelax))
        Inf_ind_plot["DetConstSort"].append(np.mean(rank.Inf_ind_DetConstSort))
    
        Min_skew_plot["vanilla"].append(np.mean(rank.Min_skew_vanilla))
        Min_skew_plot["DetGreedy"].append(np.mean(rank.Min_skew_DetGreedy))
        Min_skew_plot["DetCons"].append(np.mean(rank.Min_skew_DetCons))
        Min_skew_plot["DetRelax"].append(np.mean(rank.Min_skew_DetRelax))
        Min_skew_plot["DetConstSort"].append(np.mean(rank.Min_skew_DetConstSort))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1,figsize=(10,25))
    pd.Series(ndcg_plot["vanilla"]).reindex(index=np.arange(2,10)).plot(label='Vanilla',style='-o',kind='line',ax=ax1)
    pd.Series(ndcg_plot["DetGreedy"]).reindex(index=np.arange(2,10)).plot(label='DetGreedy',style='-<',kind='line',ax=ax1)
    pd.Series(ndcg_plot["DetCons"]).reindex(index=np.arange(2,10)).plot(label='DetCons',style='-s',kind='line',ax=ax1)
    pd.Series(ndcg_plot["DetRelax"]).reindex(index=np.arange(2,10)).plot(label='DetRelax',style='-*',kind='line',ax=ax1)
    pd.Series(ndcg_plot["DetConstSort"]).reindex(index=np.arange(2,10)).plot(label='DetConstSort',style='-+',kind='line',ax=ax1)
    ax1.legend()
    ax1.set_xlabel("Number of Attribute Values",)
    ax1.set_ylabel("Average NDCG",)
    ax1.set_title("NDCG vs Number of Attribute Values",)

    pd.Series(ndkl_plot["vanilla"]).reindex(index=np.arange(2,10)).plot(label='Vanilla',style='-o',kind='line',ax=ax2)
    pd.Series(ndkl_plot["DetGreedy"]).reindex(index=np.arange(2,10)).plot(label='DetGreedy',style='-<',kind='line',ax=ax2)
    pd.Series(ndkl_plot["DetCons"]).reindex(index=np.arange(2,10)).plot(label='DetCons',style='-s',kind='line',ax=ax2)
    pd.Series(ndkl_plot["DetRelax"]).reindex(index=np.arange(2,10)).plot(label='DetRelax',style='-*',kind='line',ax=ax2)
    pd.Series(ndkl_plot["DetConstSort"]).reindex(index=np.arange(2,10)).plot(label='DetConstSort',style='-+',kind='line',ax=ax2)
    ax2.legend()
    ax2.set_xlabel("Number of Attribute Values")
    ax2.set_ylabel("Average NDKl")
    ax2.set_title("NDKL vs Number of Attribute Values")

    pd.Series(Inf_ind_plot["vanilla"]).reindex(index=np.arange(2,10)).plot(label='Vanilla',style='-o',kind='line',ax=ax3)
    pd.Series(Inf_ind_plot["DetGreedy"]).reindex(index=np.arange(2,10)).plot(label='DetGreedy',style='-<',kind='line',ax=ax3)
    pd.Series(Inf_ind_plot["DetCons"]).reindex(index=np.arange(2,10)).plot(label='DetCons',style='-s',kind='line',ax=ax3)
    pd.Series(Inf_ind_plot["DetRelax"]).reindex(index=np.arange(2,10)).plot(label='DetRelax',style='-*',kind='line',ax=ax3)
    pd.Series(Inf_ind_plot["DetConstSort"]).reindex(index=np.arange(2,10)).plot(label='DetConstSort',style='-+',kind='line',ax=ax3)
    ax3.legend()
    ax3.set_xlabel("Number of Attribute Values")
    ax3.set_ylabel("Average Infeasible Index")
    ax3.set_title("Infeasible Index vs Number of Attribute Values")

    pd.Series(Min_skew_plot["vanilla"]).reindex(index=np.arange(2,10)).plot(label='Vanilla',style='-o',kind='line',ax=ax4)
    pd.Series(Min_skew_plot["DetGreedy"]).reindex(index=np.arange(2,10)).plot(label='DetGreedy',style='-<',kind='line',ax=ax4)
    pd.Series(Min_skew_plot["DetCons"]).reindex(index=np.arange(2,10)).plot(label='DetCons',style='-s',kind='line',ax=ax4)
    pd.Series(Min_skew_plot["DetRelax"]).reindex(index=np.arange(2,10)).plot(label='DetRelax',style='-*',kind='line',ax=ax4)
    pd.Series(Min_skew_plot["DetConstSort"]).reindex(index=np.arange(2,10)).plot(label='DetConstSort',style='-+',kind='line',ax=ax4)
    ax4.legend()
    ax4.set_xlabel("Number of Attribute Values")
    ax4.set_ylabel("Average MinSkew@100")
    ax4.set_title("MinSkew@100 vs Number of Attribute Values")
    plt.savefig('results/fair_ranking_plot.png',dpi=300)
    plt.show()
    #plot_utlis.plot_metrics()

if __name__=='__main__':
    print("Running Fair Algorithm for Re-Ranking!!")
    main()
    
