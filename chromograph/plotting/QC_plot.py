import matplotlib.pyplot as plt
import numpy as np
import loompy
from sklearn.neighbors import NearestNeighbors
from matplotlib.collections import LineCollection

## Import from cytograph
import cytograph.visualization.colors as colors
from cytograph.visualization.scatter import *

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
    xy = ds.ca[embedding]
    labels = ds.ca["Clusters"]
    if "Outliers" in ds.col_attrs:
        outliers = ds.col_attrs["Outliers"]
    else:
        outliers = np.zeros(ds.shape[1])

    if 'Class' in ds.ca:
        if attrs != None:
            if not 'Class' in attrs:
                attrs.append('Class')

    if attrs == None:
        n_axes = 8
    else:
        attrs = [x for x in attrs if x in ds.ca]
        n_axes = 8 + len(attrs)
        
    nrows = int(np.ceil(n_axes/3))
        
    # Compute a good size for the markers, based on local density
    min_pts = 50
    eps_pct = 60
    nn = NearestNeighbors(n_neighbors=min_pts, algorithm="ball_tree", n_jobs=4)
    nn.fit(xy)
    knn = nn.kneighbors_graph(mode='distance')
    k_radius = knn.max(axis=1).toarray()
    epsilon = (1000 / (xy.max() - xy.min())) * np.percentile(k_radius, eps_pct)
    
    plt.figure(figsize = (30, 10*nrows))
    
    ## Histogram of features per cell    
    if 'Chr' in ds.ra:
        plt.subplot(nrows, 3, 1)
        plt.hist(ds.ca['NPeaks'], bins=100, alpha=0.5)
        plt.title("Number of positive peaks per cell")
        plt.ylabel("Number of Cells")
        plt.xlabel("Number of positive peaks")

        ax = plt.subplot(nrows, 3, 2)
        x,y = np.log10(ds.ca['passed_filters']+1), np.log10(ds.ca['NPeaks']+1)
        m, b = np.polyfit(x, y, 1)
        plt.scatter(x,y, s=1)
        plt.plot(x, m*x + b, color="black", lw=0.75, linestyle='--')
        plt.title(f"Fragments per cell v. positive peaks per cell")
        plt.ylabel("Log10 Positive peaks")
        plt.xlabel("Log10 fragments")

        ## Plot the variance an clustermeans used for feature selection
        if 'preCluster_residuals' in ds.ra:
            metric = ds.ra.preCluster_residuals
            ylab = "Pearson residuals variance across preclusters"
        else:
            metric = np.log10((ds.ra.preCluster_sd/ds.ra.preCluster_mu)+1)
            ylab = "Log10(Coefficient of variance)"
        plt.subplot(nrows, 3, 3)
        plt.scatter(np.log10(ds.ra.preCluster_mu+1), metric, s=1, c='grey', marker='.', lw=0)
        if 'Valid' in ds.ra:
            plt.scatter(np.log10(ds.ra.preCluster_mu[np.where(ds.ra.Valid)]+1), metric[np.where(ds.ra.Valid)], s=1, c='red', marker='.', lw=0)
            plt.title("Selection of variable peaks")
        else:
            plt.title("Cluster level variance of peaks")
        plt.ylabel(ylab)
        plt.xlabel("Log10(Mean peak count (CPM) across preclusters)")

        ## Plot FRIP
        plt.subplot(nrows, 3, 4)
        scattern(xy, cmap='viridis', c=ds.ca.FRIP, s=epsilon)
        plt.colorbar(shrink=.5)
        plt.title('Fraction of fragments in peaks')
        plt.axis("off")

    else:
        plt.subplot(nrows, 3, 1)
        plt.hist(ds.ca['NBins'], bins=100, alpha=0.5)
        plt.title("Number of positive bins per cell")
        plt.ylabel("Number of Cells")
        plt.xlabel("Number of positive bins")
    
        plt.subplot(nrows, 3, 2)
        x,y = np.log10(ds.ca['passed_filters']+1), np.log10(ds.ca['NBins']+1)
        m, b = np.polyfit(x, y, 1)
        plt.scatter(x,y, s=1)
        plt.plot(x, m*x + b, color="black", lw=0.75, linestyle='--')
        plt.title("Fragments per cell v. positive bins per cell")
        plt.ylabel("Log10 Positive Bins")
        plt.xlabel("Log10 fragments")
    
        ## Histogram of Feature Coverage
        plt.subplot(nrows, 3, 3)
        plt.hist(np.log10(ds.ra['NCells']+1), bins=100, alpha=0.5, range=(0, np.log10(ds.shape[1])+0.5))    
        ## Plot min and max coverage
        plt.axvline(np.log10(np.min(ds.ra['NCells'][ds.ra['Valid']==1])+1), color="r")
        plt.axvline(np.log10(np.max(ds.ra['NCells'][ds.ra['Valid']==1])+1), color="r")
        plt.title("Coverage")
        plt.ylabel("Number of features")
        plt.xlabel("Log10 Coverage")

        ## Plot TSS fraction
        plt.subplot(nrows, 3, 4)
        scattern(xy, cmap='viridis', c=ds.ca.FRtss, s=epsilon)
        plt.colorbar(shrink=.5)
        plt.title('TSS fraction')
        plt.axis("off")
    
    ## Plot Age
    age = 'PseudoAge' if 'PseudoAge' in ds.ca else 'Age'
    plt.subplot(nrows, 3, 5)
    scattern(xy, c=ds.ca[age], cmap=colors.Colorizer("age").cmap, vmin=5, vmax=14, s=epsilon)
    plt.title(age)
    plt.colorbar(label="Age (p.c.w.)", shrink=.5)
    plt.axis("off")
    
    ## Plot the number of fragments per cell
    plt.subplot(nrows, 3, 6)
    scattern(xy, c=np.log10(ds.ca['passed_filters']), cmap='viridis', s=epsilon)
    plt.colorbar(label='Log10 fragments', shrink=.5)
    plt.title('Log10 fragments')
    plt.axis("off")
    
    ## Regions
    plt.subplot(nrows, 3, 7)
    labels = ds.ca.regions
    try:
        scatterc(xy, c=labels, colors='regions', s=epsilon)
    except:
        scatterc(xy, c=labels, colors='tube', s=epsilon)
    plt.title('Regions')
    plt.axis("off")
    
    ## Sex
    if 'SEX' not in ds.ca:
        try:
            ds.ca.Y = np.sum(ds[np.where(ds.ra.Chr == 'chrY')[0],:], axis=0) / ds.ca.passed_filters
        except:
            ds.ca.Y = np.sum(ds[np.where(ds.ra.chrom == 'chrY')[0],:], axis=0) / ds.ca.passed_filters
        SEX = 'F' if np.median(ds.ca.Y) < .0005 else 'M'
        ds.ca.SEX = np.repeat(SEX, ds.shape[1])
    plt.subplot(nrows, 3, 8)
    labels = ds.ca.SEX
    scatterc(xy, c=labels, colors='sex', s=epsilon)
    plt.title('Sex')
    plt.axis("off")

    ## Plot the attributes on the embedding
    if attrs is not None:
        for n, attr in enumerate(attrs):
            x = n + 9
            
            plt.subplot(nrows, 3, x)
            labels = ds.ca[attr]
            scatterc(xy, c=labels, colors='tube', s=epsilon)
            plt.title(attr)
            plt.axis("off")
            
    plt.savefig(out_file, format="png", dpi=300, bbox_inches='tight')