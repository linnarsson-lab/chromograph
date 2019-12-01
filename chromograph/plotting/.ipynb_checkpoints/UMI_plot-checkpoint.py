import matplotlib.pyplot as plt
import numpy as np
import loompy
from sklearn.neighbors import NearestNeighbors
from matplotlib.collections import LineCollection

def UMI_plot(ds: loompy.LoomConnection, out_file: str, embedding: str = "TSNE") -> None:
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
    
    fig = plt.figure(figsize=(25, 20))
    ax = fig.add_axes([0, 0, 0.40, 0.45])
    
    # Draw edges
    if has_edges:
        lc = LineCollection(zip(pos[g.row], pos[g.col]), linewidths=0.25, zorder=0, color='thistle', alpha=0.1)
        ax.add_collection(lc)
    
    im = ax.scatter(ds.ca[embedding][:,0],ds.ca[embedding][:,1], cmap='viridis', c=np.log10(ds.ca['passed_filters']), marker='.', lw=0, s=epsilon)
    
    cax = fig.add_axes([0.45, 0.05, 0.005, 0.4])
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title('Log10 UMIs')
    ax.axis("off")
    
    ## Histogram of Bin Coverage
    ax2 = fig.add_axes([0, 0.5, 0.45, 0.2])

    ax2.hist(np.log10(ds.ra['NCells'][ds.ra['NCells'] > 0]), bins=100, alpha=0.5, range=(0, np.log10(ds.shape[1])+0.5))
    ax2.set_title("Bin Coverage")
    ax2.set_ylabel("Number of Bins")
    ax2.set_xlabel("Log10 Coverage")
    
    ## Histogram of Bins per cell
    ax3 = fig.add_axes([0.5, 0.5, 0.45, 0.2])
    
    ax3.hist(ds.ca['NBins'], bins=100, alpha=0.5)
    ax3.set_title("Number of positive bins per cell")
    ax3.set_ylabel("Number of Cells")
    ax3.set_xlabel("Number of positive bins")
    
    ax4 = fig.add_axes([0.5, 0, 0.40, 0.45])
    ax4.scatter(np.log10(ds.ca['passed_filters']), np.log10(ds.ca['NBins']), s=1)
    ax4.set_title("Fragments per cell v. positive bins per cell")
    ax4.set_ylabel("Log10 Positive Bins")
    ax4.set_xlabel("Log10 UMIs")
    
    fig.savefig(out_file, format="png", dpi=144, bbox_inches='tight')
