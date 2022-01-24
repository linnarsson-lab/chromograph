## Imports
import loompy
import numpy as np
import matplotlib.pyplot as plt

def Motif_heatmap(ds, outfile, N:int=None):
    '''
    args:
        ds
        outfile
        N
    '''
    name = ds.filename.split('/')[-2]
    if '-log_pval_trinaries' in ds.layers:
        layer = '-log_pval_trinaries'
    else:
        layer = '-log_pval'

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
    TF_order = np.argsort(TF_order)
    ds.permute(TF_order, axis=0)
    
    Shape = (len(valids), ds.shape[1])

    fig, ax = plt.subplots(figsize=Shape)
    x = np.where(ds.ra.valids)[0]
    mat = ds[layer][x,:]
    vmax = np.max(mat)
    im = ax.imshow(mat, cmap = 'Reds', vmin=2, vmax=vmax)
    labels = [f'{ds.ra.TF[i]}' for i in x]
    ax.set_title(f'TF motif enrichment by Cluster in {name}')

    ax.set_yticks(range(len(x)))
    ax.set_yticklabels(labels, fontsize=16)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=.25)
    cbar.ax.tick_params(labelsize=18) 
    cbar.set_label('-log10 p-val', fontsize=18)
    
    plt.savefig(outfile, dpi=144)
