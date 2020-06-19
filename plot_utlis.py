import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.style.use('ggplot')

def plot_metrics():
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1,figsize=(10,25))
    pd.Series(ndcg_plot["vanilla"]).reindex(index=np.arange(2,10)).plot(label='Vanilla',style='-o',kind='line ',ax=ax1)
    pd.Series(ndcg_plot["DetGreedy"]).reindex(index=np.arange(2,10)).plot(label='DetGreedy',style='-<',kind='line',ax=ax1)
    pd.Series(ndcg_plot["DetCons"]).reindex(index=np.arange(2,10)).plot(label='DetCons',style='-s',kind='line',ax=ax1)
    pd.Series(ndcg_plot["DetRelax"]).reindex(index=np.arange(2,10)).plot(label='DetRelax',style='-*',kind='line',ax=ax1)
    pd.Series(ndcg_plot["DetConstSort"]).reindex(index=np.arange(2,10)).plot(label='DetConstSort',style='-+',kind='line',ax=ax1)
    ax1.legend()
    ax1.set_xlabel("Number of Attribute Values",)
    ax1.set_ylabel("Average NDCG",)
    ax1.set_title("NDCG vs Number of Attribute Values",)

    pd.Series(ndkl_plot["vanilla"]).reindex(index=np.arange(2,10)).plot(label='Vanilla',style='-o',kind='line ',ax=ax2)
    pd.Series(ndkl_plot["DetGreedy"]).reindex(index=np.arange(2,10)).plot(label='DetGreedy',style='-<',kind='line',ax=ax2)
    pd.Series(ndkl_plot["DetCons"]).reindex(index=np.arange(2,10)).plot(label='DetCons',style='-s',kind='line',ax=ax2)
    pd.Series(ndkl_plot["DetRelax"]).reindex(index=np.arange(2,10)).plot(label='DetRelax',style='-*',kind='line',ax=ax2)
    pd.Series(ndkl_plot["DetConstSort"]).reindex(index=np.arange(2,10)).plot(label='DetConstSort',style='-+',kind='line',ax=ax2)
    ax2.legend()
    ax2.set_xlabel("Number of Attribute Values")
    ax2.set_ylabel("Average NDKl")
    ax2.set_title("NDKL vs Number of Attribute Values")

    pd.Series(Inf_ind_plot["vanilla"]).reindex(index=np.arange(2,10)).plot(label='Vanilla',style='-o',kind='line ',ax=ax3)
    pd.Series(Inf_ind_plot["DetGreedy"]).reindex(index=np.arange(2,10)).plot(label='DetGreedy',style='-<',kind='line',ax=ax3)
    pd.Series(Inf_ind_plot["DetCons"]).reindex(index=np.arange(2,10)).plot(label='DetCons',style='-s',kind='line',ax=ax3)
    pd.Series(Inf_ind_plot["DetRelax"]).reindex(index=np.arange(2,10)).plot(label='DetRelax',style='-*',kind='line',ax=ax3)
    pd.Series(Inf_ind_plot["DetConstSort"]).reindex(index=np.arange(2,10)).plot(label='DetConstSort',style='-+',kind='line',ax=ax3)
    ax3.legend()
    ax3.set_xlabel("Number of Attribute Values")
    ax3.set_ylabel("Average Infeasible Index")
    ax3.set_title("Infeasible Index vs Number of Attribute Values")

    pd.Series(Min_skew_plot["vanilla"]).reindex(index=np.arange(2,10)).plot(label='Vanilla',style='-o',kind='line ',ax=ax4)
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