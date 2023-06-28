import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import loompy
from sklearn.neighbors import NearestNeighbors
from matplotlib.collections import LineCollection

def motif_plot(ds: loompy.LoomConnection, dsr: loompy.LoomConnection, out_file: str, N:int = 5) -> None:
    '''
    Generates a multi-panel plot to inspect Motif enrichment scores.
    
    Args:
        ds                    Connection to the .agg.loom motif file to use
        dsr                   Connection to the .agg.loom RNA file to use
        out_file              Name and location of the output file
        N                     The number of motifs to include per cluster (top N)
        
    Remarks:
    
    '''
    layer = '-log_pval'
    if '-log_pval_filtered' in ds.layers:
        layer = '-log_pval_filtered'

    if N:
        valids = []
        for i in range(ds.shape[1]):
            q = np.quantile(ds[layer][:,i], (ds.shape[0]-N)/ds.shape[0])
            valids.append(np.where(ds[layer][:,i]>q)[0])
        valids = np.unique([i for s in valids for i in s])
    else:
        valids = np.repeat(True,ds.shape[0])
    ds.ra.valids = np.zeros(ds.shape[0])
    ds.ra.valids[valids] = True
    
    mask = np.zeros(ds.shape[0], dtype=bool)
    mask[valids] = 1
    
    TF_order = np.zeros(ds.shape[0], dtype='int')
    TF_order[mask] = np.argmax(ds.layer[layer][np.where(mask)[0], :], axis=1)
    TF_order[~mask] = np.argmax(ds.layer[layer][np.where(~mask)[0], :], axis=1) + ds.shape[1]
    TF_order = np.argsort(TF_order)[::-1]
    ds.permute(TF_order, axis=0)
    
    shape_factor = len(valids)/ds.shape[1]

    layer = '-log_pval'
    x = np.where(ds.ra.valids)[0]
    pval = ds[layer][x,:]

    df = pd.DataFrame([])
    TF = [f'{ds.ra.TF[i]}' for i in x]
    if 'ClusterName' in ds.ca:
        xlabels = [f'{i}: {name}' for i, name in zip(ds.ca.Clusters,ds.ca.ClusterName)]
    elif 'Class' in ds.ca:
        xlabels = [f'{i}: {name}' for i, name in zip(ds.ca.Clusters,ds.ca.Class)]
    elif 'AutoAnnotation' in ds.ca:
        xlabels = [f'{i}: {name}' for i, name in zip(ds.ca.Clusters,ds.ca.AutoAnnotation)]
    else:
        xlabels = np.arange(ds.shape[1])
    rx = np.where(np.isin(dsr.ra.Gene, TF))
    genes = dsr.ra.Gene[rx]
    s = [np.where(genes==x)[0][0] for x in TF]
    for i in range(ds.shape[1]): 
        pvals = ds[layer][:,i][x]
        trin = dsr['trinaries'][:,i][rx][s]
        data = pd.DataFrame({'Cluster': i, 'TF': TF, 'p_vals': pvals, 'trinaries': trin}, columns=['Cluster', 'TF', 'p_vals', 'trinaries'])
        df = df.append(data)

    factor = np.max(df['p_vals']) / 200
    df['Bubble_size'] = df['p_vals'] / factor
    order = sorted(np.unique(df['Cluster']))
    df['Cluster'] = [order.index(x) for x in df['Cluster']]
    df = df.set_index(np.arange(0,df.shape[0])) 

    h, w = int(np.ceil(len(x)/3)),int(np.ceil(ds.shape[1]/3))
    fig, ax = plt.subplots(figsize=(w,h), dpi=200)
    h = lambda c: plt.Line2D([], [], color=c, ls="", marker="o")

    scatter = ax.scatter('Cluster', 'TF', s='Bubble_size', c='trinaries', cmap='Reds', data=df, edgecolors='black', linewidth=.5)
    handles, labels = scatter.legend_elements(prop="sizes", num=5, color='lightgrey') 
    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', shrink=.25)
    cbar.ax.tick_params(size=0)
    cbar.outline.set_visible(False)
    legend2 = ax.legend(handles, labels, bbox_to_anchor=(1.05,1), 
                        labelspacing=1, title=f"-log pval", title_fontsize=16, frameon=False, fontsize=12)
    plt.xticks(np.arange(len(xlabels)), xlabels, rotation=90,  fontsize=10) 
    plt.yticks(range(len(TF)), TF, fontsize=12) 
    plt.xlim(-1, ds.shape[1])
    plt.ylim(-1, len(x))
    ax.set_axisbelow(True) 
    plt.title('Motif enrichment and gene expression', fontsize=20, pad=20)
    plt.savefig(out_file, bbox_inches='tight')