import numpy as np
import os
import sys
import collections
import matplotlib.pyplot as plt
import gzip
import loompy
import scipy.sparse as sparse
import urllib.request
import pybedtools
import warnings
import logging

from cytograph.decomposition import HPF
from scipy.stats import poisson
from cytograph.manifold import BalancedKNN
from cytograph.metrics import jensen_shannon_distance
from cytograph.embedding import tsne
from cytograph.clustering import PolishedLouvain, PolishedSurprise
from cytograph.plotting import manifold

sys.path.append('/home/camiel/chromograph/')
from chromograph.plotting.UMI_plot import UMI_plot

from umap import UMAP
import sklearn.metrics
from scipy.spatial import distance
import community
import networkx as nx
from scipy import sparse
from typing import *

class bin_analysis:
    def __init__(self) -> None:
        """
        Perform Dimensional Reduction and Clustering on a binned loom-file   
        Args:
            steps                    Which steps to include in the analysis

        Remarks:
            --- PUNCHCARD SUPPORT NEEDS TO INTEGRATED, FOR NOW PASS PARAMETER TO FIT FUNCTION ---
            # All parameters are obtained from the config object, which comes from the default config
            # and can be overridden by the config in the current punchcard
        """
    #   self.config = config
        logging.info("Bin_Analysis initialised")
    
    def fit(self, ds: loompy.LoomConnection, outdir) -> None:
        blayer = '{}kb_bins'.format(int(ds.attrs['bin_size'] / 1000))
        logging.info("Running Bin-analysis on {} cells with {}".format(ds.shape[1], blayer))
        
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        
        ## Bin selection  --RIGHT NOW WE ONLY FILTER OUT BINS WITH TOTAL COUNTS < 10 AND THE TOP 5% OF BINS--
        ## nonzero (nnz) counts per bin
        ds.ra['NCells'] = ds.map([np.count_nonzero], axis=0)[0]
        ds.ca['NBins'] = ds.map([np.count_nonzero], axis=1)[0]
        nnz_l = np.log10(ds.ra['NCells'] + 1)
        bin_cutoff = np.quantile(ds.ra['NCells'], 0.95)
        
        bins = np.logical_and(ds.ra['NCells'] > 10, ds.ra['NCells'] < bin_cutoff)
        logging.info("Using {} out of {} bins for manifold learning".format(sum(bins), ds.shape[0]))
        
        ## Create binary layer
        logging.info("Binarizing the matrix")
        ds.layers[blayer] = np.clip(ds[:,:], 0, 1)

        # Load the data for the selected genes
        logging.info("Selecting data for HPF factorization: {} cells and {} bins".format(ds.shape[1], sum(bins)))
        data = ds[blayer].sparse(rows=bins).T

        # HPF factorization
        k = 96
        cpus = 8
        logging.info("Performing HPF factorization with {} factors".format(k))
        hpf = HPF(k=k, validation_fraction=0.05, min_iter=10, max_iter=200, compute_X_ppv=False, n_threads=cpus)
        hpf.fit(data)
        
        logging.info("Adding Betas and Thetas to loom file")
        beta_all = np.zeros((ds.shape[0], hpf.beta.shape[1]))
        beta_all[bins] = hpf.beta
        # Save the unnormalized factors
        ds.ra.HPF_beta = beta_all
        ds.ca.HPF_theta = hpf.theta
        # Here we normalize so the sums over components are one, because JSD requires it
        # and because otherwise the components will be exactly proportional to cell size
        theta = (hpf.theta.T / hpf.theta.sum(axis=1)).T
        beta = (hpf.beta.T / hpf.beta.sum(axis=1)).T
        beta_all[bins] = beta
        # Save the normalized factors
        ds.ra.HPF = beta_all
        ds.ca.HPF = theta

        logging.info("Calculating posterior probabilities")
        # Expected values
        exp = "{}_expected".format(blayer)
        n_samples = ds.shape[1]

        ds[exp] = 'float32'  # Create a layer of floats
        log_posterior_proba = np.zeros(n_samples)
        theta_unnormalized = hpf.theta
        data = data.toarray()
        start = 0
        batch_size = 6400
        beta_all = ds.ra.HPF_beta  # The unnormalized beta

        while start < n_samples:
            # Compute PPV (using normalized theta)
            ds[exp][:, start: start + batch_size] = beta_all @ theta[start: start + batch_size, :].T
            # Compute PPV using raw theta, for calculating posterior probability of the observations
            ppv_unnormalized = beta @ theta_unnormalized[start: start + batch_size, :].T
            log_posterior_proba[start: start + batch_size] = poisson.logpmf(data.T[:, start: start + batch_size], ppv_unnormalized).sum(axis=0)
            start += batch_size
        ds.ca.HPF_LogPP = log_posterior_proba
        
        ## Construct nearest-neighbor graph
        logging.info("Constructing nearest-neighbor graph")
        bnn = BalancedKNN(k=25, metric="js", maxl=2 * 25, sight_k=2 * 25, n_jobs=-1)
        bnn.fit(theta)
        knn = bnn.kneighbors_graph(mode='distance')
        knn.eliminate_zeros()
        mknn = knn.minimum(knn.transpose())
        # Convert distances to similarities
        knn.data = 1 - knn.data
        mknn.data = 1 - mknn.data
        ds.col_graphs.KNN = knn
        ds.col_graphs.MKNN = mknn
        # Compute the effective resolution
        d = 1 - knn.data
        d = d[d < 1]
        radius = np.percentile(d, 90)
        ds.attrs.radius = radius
        knn = knn.tocoo()
        knn.setdiag(0)
        inside = knn.data > 1 - radius
        rnn = sparse.coo_matrix((knn.data[inside], (knn.row[inside], knn.col[inside])), shape=knn.shape)
        ds.col_graphs.RNN = rnn
        
        ## Perform tSNE and UMAP
        logging.info("Generating tSNE from thetas")
        ds.ca.TSNE = tsne(theta, metric="js", radius=radius)

        logging.info("Generating UMAP from thetas")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)  # Suppress an annoying UMAP warning about meta-embedding
            ds.ca.UMAP = UMAP(n_components=2, metric=jensen_shannon_distance, n_neighbors=25 // 2, learning_rate=0.3, min_dist=0.25).fit_transform(theta)

        logging.info("Generating 3D UMAP from thetas")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            ds.ca.UMAP3D = UMAP(n_components=3, metric=jensen_shannon_distance, n_neighbors=25 // 2, learning_rate=0.3, min_dist=0.25).fit_transform(theta)

        ## Perform Clustering
        logging.info("Performing Polished Louvain clustering")
        pl = PolishedLouvain(outliers=False)
        labels = pl.fit_predict(ds, graph="RNN", embedding="UMAP3D")
        ds.ca.ClustersModularity = labels + min(labels)
        ds.ca.OutliersModularity = (labels == -1).astype('int')
        ds.ca.Clusters = labels + min(labels)
        ds.ca.Outliers = (labels == -1).astype('int')

        logging.info("Performing Louvain Polished Surprise clustering")
        ps = PolishedSurprise(embedding="TSNE")
        labels = ps.fit_predict(ds)
        ds.ca.ClustersSurprise = labels + min(labels)
        ds.ca.OutliersSurprise = (labels == -1).astype('int')
        logging.info(f"Found {ds.ca.Clusters.max() + 1} clusters")
        
        ## Plot results on manifold
        logging.info("Plotting UMAP")
        manifold(ds, os.path.join(outdir, 'manifold_UMAP.png'), embedding = 'UMAP')
        logging.info("Plotting TSNE")
        manifold(ds, os.path.join(outdir, 'manifold_TSNE.png'), embedding = 'TSNE')
        logging.info("plotting the number of UMIs")
        UMI_plot(ds, os.path.join(outdir, 'manifold_UMI.png'), embedding = 'TSNE')