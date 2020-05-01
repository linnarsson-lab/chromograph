import matplotlib.pyplot as plt
import numpy as np
import loompy
from sklearn.neighbors import NearestNeighbors
from matplotlib.collections import LineCollection

def marker_plot(ds: loompy.LoomConnection, out_file: str, markers: list, lay: str = 'smooth', embedding: str = "TSNE") -> None:
    '''
    Generates a multi-panel plot to inspect UMI and Bin counts.
    
    Args:
        ds                    Connection to the .loom file to use
        out_file              Name and location of the output file
        embedding             The embedding to use for UMI manifold plot (TSNE or UMAP)
        
    Remarks:
    
    '''
    n_cells = ds.shape[1]
    
    has_edges = False
    if "RNN" in ds.col_graphs:
        g = ds.col_graphs.RNN
        has_edges = True
    elif "MKNN" in ds.col_graphs:
        g = ds.col_graphs.MKNN
        has_edges = True
    if embedding in ds.ca:
        pos = ds.ca[embedding]
    else:
        raise ValueError("Embedding not found in the file")
    labels = ds.ca["Clusters"]
    if "Outliers" in ds.col_attrs:
        outliers = ds.col_attrs["Outliers"]
    else:
        outliers = np.zeros(ds.shape[1])
        
    # Compute a good size for the markers, based on local density
    min_pts = 50
    eps_pct = 60
    nn = NearestNeighbors(n_neighbors=min_pts, algorithm="ball_tree", n_jobs=4)
    nn.fit(pos)
    knn = nn.kneighbors_graph(mode='distance')
    k_radius = knn.max(axis=1).toarray()
    epsilon = (2500 / (pos.max() - pos.min())) * np.percentile(k_radius, eps_pct)
    
    nM = len(markers)
    
    fig = plt.figure(figsize=(32,np.ceil(nM/5)*8))
    
    i = 1
    for m in markers:
        ax = fig.add_subplot(np.ceil(nM/4), 4, i)

        ## GA score    
        v = ds[lay][ds.ra['Gene'] == m, :][0]
        k = v > 0
        GA = v[k]
        q = np.quantile(v, 0.99)
        
        ax.scatter(pos[:,0], pos[:,1],s=epsilon, c = 'lightgrey', alpha=.5)
        im = ax.scatter(pos[k,0], pos[k,1], cmap='viridis', c=GA, vmax = q, marker='.', lw=0, s=epsilon)
        
        fig.colorbar(im, ax=ax, orientation='vertical', shrink = 0.5)
        ax.set_title(f'{m}:  {ds.ra.loc[ds.ra.Gene == m]}', fontdict={'fontsize': 8, 'fontweight': 'medium'})
        ax.axis("off")
        
        i += 1
    
    fig.savefig(out_file, format="png", dpi=144, bbox_inches='tight')