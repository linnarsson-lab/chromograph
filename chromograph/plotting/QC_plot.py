import matplotlib.pyplot as plt
import numpy as np
import loompy
from sklearn.neighbors import NearestNeighbors
from matplotlib.collections import LineCollection

import matplotlib.pyplot as plt
import numpy as np
import loompy
from sklearn.neighbors import NearestNeighbors
from matplotlib.collections import LineCollection

## Import from cytograph
from cytograph.plotting.colors import colorize

def QC_plot(ds: loompy.LoomConnection, out_file: str, embedding: str = "TSNE", attrs: list = None) -> None:
    '''
    Generates a multi-panel plot to inspect UMI and Bin counts.
    
    Args:
        ds                    Connection to the .loom file to use
        out_file              Name and location of the output file
        embedding             The embedding to use for UMI manifold plot (TSNE or UMAP)
        attrs                 List of column attributes to plot
        
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
        
    if attrs == None:
        n_axes = 4
    else:
        n_axes = 4 + len(attrs)
        
    nrows = int(np.ceil(n_axes/2))
        
    # Compute a good size for the markers, based on local density
    min_pts = 50
    eps_pct = 60
    nn = NearestNeighbors(n_neighbors=min_pts, algorithm="ball_tree", n_jobs=4)
    nn.fit(pos)
    knn = nn.kneighbors_graph(mode='distance')
    k_radius = knn.max(axis=1).toarray()
    epsilon = (2500 / (pos.max() - pos.min())) * np.percentile(k_radius, eps_pct)
    
    fig, ax = plt.subplots(nrows, 2, figsize = (20, 10*nrows))
    ax = ax.flatten()
    
    ## Histogram of features per cell    
    if 'Chr' in ds.ra:
        ax[0].hist(ds.ca['NPeaks'], bins=100, alpha=0.5)
        ax[0].set_title("Number of positive peaks per cell")
        ax[0].set_ylabel("Number of Cells")
        ax[0].set_xlabel("Number of positive peaks")

        ax[1].scatter(np.log10(ds.ca['passed_filters']), np.log10(ds.ca['NPeaks']), s=1)
        ax[1].set_title("Fragments per cell v. positive peaks per cell")
        ax[1].set_ylabel("Log10 Positive peaks")
        ax[1].set_xlabel("Log10 fragments")

        ## Plot the variance an clustermeans used for feature selection
        ax[2].scatter(ds.ra.mu, ds.ra.sd, s=1, c='grey', marker='.')
        ax[2].scatter(ds.ra.mu[np.where(ds.ra.Valid)], ds.ra.sd[np.where(ds.ra.Valid)], s=1, c='red', marker='.')
        ax[2].set_title("Selection of peaks by variance")
        ax[2].set_ylabel("Standard deviation across preclusters")
        ax[2].set_xlabel("Mean peak count (CPM) across preclusters")

    else:
        ax[0].hist(ds.ca['NBins'], bins=100, alpha=0.5)
        ax[0].set_title("Number of positive bins per cell")
        ax[0].set_ylabel("Number of Cells")
        ax[0].set_xlabel("Number of positive bins")
    
        ax[1].scatter(np.log10(ds.ca['passed_filters']), np.log10(ds.ca['NBins']), s=1)
        ax[1].set_title("Fragments per cell v. positive bins per cell")
        ax[1].set_ylabel("Log10 Positive Bins")
        ax[1].set_xlabel("Log10 fragmentss")
    
        ## Histogram of Feature Coverage
        ax[2].hist(np.log10(ds.ra['NCells']+1), bins=100, alpha=0.5, range=(0, np.log10(ds.shape[1])+0.5))    
        ## Plot min and max coverage
        ax[2].axvline(np.log10(np.min(ds.ra['NCells'][ds.ra['Valid']==1])+1), color="r")
        ax[2].axvline(np.log10(np.max(ds.ra['NCells'][ds.ra['Valid']==1])+1), color="r")
        ax[2].set_title("Coverage")
        ax[2].set_ylabel("Number of features")
        ax[2].set_xlabel("Log10 Coverage")
    
    ## Plot the number of fragments per cell
    # Draw edges
    if has_edges:
        lc = LineCollection(zip(pos[g.row], pos[g.col]), linewidths=0.25, zorder=0, color='thistle', alpha=0.1)
        ax[3].add_collection(lc)
    
    im = ax[3].scatter(ds.ca[embedding][:,0],ds.ca[embedding][:,1], cmap='viridis', c=np.log10(ds.ca['passed_filters']), marker='.', lw=0, s=epsilon)
    fig.colorbar(im, ax=ax[3], orientation='vertical', shrink=.5)
    ax[3].set_title('Log10 fragments')
    ax[3].axis("off")
    
    ## Plot the attributes on the embedding
    if attrs is not None:
        for n, attr in enumerate(attrs):
            x = n +4
            # Draw edges
            if has_edges:
                lc = LineCollection(zip(pos[g.row], pos[g.col]), linewidths=0.25, zorder=0, color='thistle', alpha=0.1)
                ax[x].add_collection(lc)
            
            ax[x].scatter(ds.ca[embedding][:,0],ds.ca[embedding][:,1], c='lightgrey', marker='.', lw=0, s=epsilon)
            
            names, labels = np.unique(ds.ca[attr], return_inverse=True)
            colors = colorize(names)
            ax[x].scatter(ds.ca[embedding][:,0],ds.ca[embedding][:,1], c=colors[labels], marker='.', lw=0, s=epsilon)

            def h(c):
                return plt.Line2D([], [], color=c, ls="", marker="o")
            ax[x].legend(handles=[h(colors[i]) for i in range(len(names))], labels=list(names), loc='lower left', markerscale=1, frameon=False, fontsize=10)
            ax[x].set_title(f'{attr}')
            ax[x].axis("off")
            
    
    fig.savefig(out_file, format="png", dpi=144, bbox_inches='tight')
