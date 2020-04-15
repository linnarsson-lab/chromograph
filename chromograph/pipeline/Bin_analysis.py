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
from cytograph.embedding import art_of_tsne
from cytograph.clustering import PolishedLouvain, PolishedSurprise
from cytograph.plotting import manifold

# sys.path.append('/home/camiel/chromograph/')
from chromograph.plotting.QC_plot import QC_plot
from chromograph.features.bin_annotation import Bin_annotation
from chromograph.pipeline.TF_IDF import TF_IDF
from chromograph.pipeline import config

from umap import UMAP
import sklearn.metrics
from scipy.spatial import distance
import community
import networkx as nx
from scipy import sparse
from typing import *

from sklearn.decomposition import IncrementalPCA

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
        self.config = config.load_config()
        self.outdir = os.path.join(self.config.paths.build, 'exported')
        self.blayer = 'TF_IDF'
        logging.info("Bin_Analysis initialised")
    
    def fit(self, ds: loompy.LoomConnection) -> None:
        logging.info(f"Running Chromograph on {ds.shape[1]} cells")
        
        try:
            self.blayer = '{}kb_bins'.format(int(ds.attrs['bin_size'] / 1000))
        except:
            ds.attrs['bin_size'] = int(int(ds.ra.end[0]) - int(ds.ra.start[0]) + 1)
            self.blayer = '{}kb_bins'.format(int(ds.attrs['bin_size'] / 1000))
        logging.info(f'Running Bin-analysis on {ds.shape[1]} cells with {self.blayer}')
        
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        
        ## nonzero (nnz) counts per bin
        logging.info('Calculating bin and cell coverage')
        ds.ra['NCells'] = ds.map([np.count_nonzero], axis=0)[0]
        ds.ca['NBins'] = ds.map([np.count_nonzero], axis=1)[0]
        
        ## Calculate coverage
        cov = np.log10(ds.ra['NCells']+1)
        mu = np.mean(cov)
        sd = np.std(cov)
        ds.ra['Coverage'] = (cov - mu) / sd
        
        ## Create binary layer
        logging.info("Binarizing the matrix")
        nnz = ds[:,:] > 0
        nnz.dtype = 'int8'
        ds.layers[self.blayer] = nnz

        del nnz, cov, mu, sd

        ## Term-Frequence Inverse-Data-Frequency ##
        if 'TF-IDF' in self.config.params.Normalization:
            logging.info(f'Performing TF-IDF')
            tf_idf = TF_IDF()
            tf_idf.fit(ds)
            X = np.zeros((ds.shape[0], ds.shape[1]))
            for (ix, selection, view) in ds.scan(axis=1):
                X[:,selection] = tf_idf.transform(view[:,:], selection)
                logging.info(f'transformed {max(selection)} cells')
            ds.layers['TF_IDF'] = X.astype('float16')
            self.blayer = 'TF_IDF'

            del X

        if 'PCA' in self.config.params.factorization:
            ## Select bins for PCA fitting
            # bins = (ds.ra['Coverage'] > -self.config.params.cov) & (ds.ra['Coverage'] < self.config.params.cov)

            ds.ra.Valid = np.array((ds.ra['Coverage'] > 0) & (ds.ra['Coverage'] < self.config.params.cov)==1)
            ds.attrs['bin_max_cutoff'] = max(ds.ra['NCells'][ds.ra.Valid==1])
            ds.attrs['bin_min_cutoff'] = min(ds.ra['NCells'][ds.ra.Valid==1])
            PCA = IncrementalPCA(n_components=self.config.params.n_factors)
            logging.info(f'Fitting {sum(ds.ra.Valid)} bins from {ds.shape[1]} cells to {self.config.params.n_factors} components')
            for (ix, selection, view) in ds.scan(axis=1):
                PCA.partial_fit(view[self.blayer][ds.ra.Valid==1, :].T)
                logging.info(f'Fitted {ix} cells to PCA')

            logging.info(f'Transforming data')
            x = []
            for (ix, selection, view) in ds.scan(axis=1):
                X.append(PCA.transform(view[self.blayer][ds.ra.Valid==1,:].T))
                logging.info(f'Transformed {ix} cells')
            X = np.vstack(X)
            # X = PCA.transform(ds[self.blayer][ds.ra.Valid==1,:].T)

            logging.info(f'Shape X is {X.shape}')
            ds.ca.PCA = X
            logging.info(f'Added PCA components')
            del X, PCA
        
        if 'HPF' in self.config.params.factorization:
            logging.info("Performing HPF")
            
            ## Bin selection  --RIGHT NOW WE FITLER THE ABSOLUTE TOP 1% BINS and bottom 80%--
            ds.attrs['bin_max_cutoff'] = np.quantile(ds.ra['NCells'], 0.99)
            ds.attrs['bin_min_cutoff'] = np.quantile(ds.ra['NCells'], 0.80)
            
            logging.info(f"Selected max cut_off {ds.attrs['bin_max_cutoff']} and min cut_off {ds.attrs['bin_min_cutoff']}")

            ds.ra.Valid = np.array((ds.ra['NCells'] > ds.attrs['bin_min_cutoff']) & (ds.ra['NCells'] < ds.attrs['bin_max_cutoff']))
            logging.info("Using {} out of {} bins for manifold learning".format(sum(ds.ra.Valid), ds.shape[0]))
            # Load the data for the selected genes
            logging.info(f"Selecting data for HPF factorization: {ds.shape[1]} cells and {sum(ds.ra.Valid)} bins")
            data = ds[self.blayer].sparse(rows=np.array(ds.ra.Valid==1)).T
            logging.info(f"Shape data: {data.shape}")

            # HPF factorization
            cpus = 4
            logging.info("Performing HPF factorization with {} factors".format(k))
            hpf = HPF(k=self.config.params.HPF_factors, validation_fraction=0.05, min_iter=10, max_iter=200, compute_X_ppv=False, n_threads=cpus)
            hpf.fit(data)

            logging.info("Adding Betas and Thetas to loom file")
            beta_all = np.zeros((ds.shape[0], hpf.beta.shape[1]))
            beta_all[ds.ra.Valid==1] = hpf.beta
            # Save the unnormalized factors
            ds.ra.HPF_beta = beta_all
            ds.ca.HPF_theta = hpf.theta
            # Here we normalize so the sums over components are one, because JSD requires it
            # and because otherwise the components will be exactly proportional to cell size
            theta = (hpf.theta.T / hpf.theta.sum(axis=1)).T
            beta = (hpf.beta.T / hpf.beta.sum(axis=1)).T
            beta_all[ds.ra.Valid==1] = beta
            # Save the normalized factors
            ds.ra.HPF = beta_all
            ds.ca.HPF = theta

            logging.info("Calculating posterior probabilities")
            # Expected values
            exp = "{}_expected".format(self.blayer)
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
        
        if 'PCA' in self.config.params.factorization:
            decomp = ds.ca['PCA']
            metric = "euclidean"
        elif 'HPF' in self.config.params.factorization:
            decomp = ds.ca['HPF']
            metric = "js"

        ## Construct nearest-neighbor graph
        logging.info(f"Computing balanced KNN (k = {self.config.params.k}) in {self.config.params.nn_space} space using the '{metric}' metric")
        bnn = BalancedKNN(k=self.config.params.k, metric=metric, maxl=2 * self.config.params.k, sight_k=2 * self.config.params.k, n_jobs=-1)
        bnn.fit(decomp)
        knn = bnn.kneighbors_graph(mode='distance')
        knn.eliminate_zeros()
        mknn = knn.minimum(knn.transpose())
        # Convert distances to similarities
        logging.info(f'Converting distances to similarities knn shape.')
        max_d = knn.data.max()
        knn.data = (max_d - knn.data) / max_d
        mknn.data = (max_d - mknn.data) / max_d
        ds.col_graphs.KNN = knn
        ds.col_graphs.MKNN = mknn
        mknn = mknn.tocoo()
        mknn.setdiag(0)
        # Compute the effective resolution
        logging.info(f'Computing resolution')
        d = 1 - knn.data
        radius = np.percentile(d, 90)
        logging.info(f"  90th percentile radius: {radius:.02}")
        ds.attrs.radius = radius
        inside = mknn.data > 1 - radius
        rnn = sparse.coo_matrix((mknn.data[inside], (mknn.row[inside], mknn.col[inside])), shape=mknn.shape)
        ds.col_graphs.RNN = rnn

        del knn, mknn, rnn
    
        ## Perform tSNE and UMAP
        logging.info(f"Computing 2D and 3D embeddings from latent space")
        metric_f = (jensen_shannon_distance if metric == "js" else metric)  # Replace js with the actual function, since OpenTSNE doesn't understand js
        # logging.info(f"  Art of tSNE with {metric} distance metric")
        # ds.ca.TSNE = np.array(art_of_tsne(decomp, metric=metric_f))  # art_of_tsne returns a TSNEEmbedding, which can be cast to an ndarray (its actually just a subclass)

        logging.info(f'Using sklearn TSNE for the time being')
        from sklearn.manifold import TSNE
        TSNE = TSNE(init='pca') ## TSNE uses a random seed to initiate, meaning that the results don't always look the same!
        ds.ca.TSNE = TSNE.fit(decomp).embedding_

        logging.info("Generating UMAP from decomposition")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)  # Suppress an annoying UMAP warning about meta-embedding
            ds.ca.UMAP = UMAP(n_components=2, metric=metric_f, n_neighbors=25 // 2, learning_rate=0.3, min_dist=0.25).fit_transform(decomp)

        logging.info("Generating 3D UMAP from decomposition")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            ds.ca.UMAP3D = UMAP(n_components=3, metric=metric_f, n_neighbors=25 // 2, learning_rate=0.3, min_dist=0.25).fit_transform(decomp)

        ## Perform Clustering
        logging.info("Performing Polished Louvain clustering")
        pl = PolishedLouvain(outliers=False, graph="RNN", embedding="TSNE")
        labels = pl.fit_predict(ds)
        ds.ca.ClustersModularity = labels + min(labels)
        ds.ca.OutliersModularity = (labels == -1).astype('int')
        ds.ca.Clusters = labels + min(labels)
        ds.ca.Outliers = (labels == -1).astype('int')

        logging.info("Performing Louvain Polished Surprise clustering")
        ps = PolishedSurprise(graph="RNN", embedding="TSNE")
        labels = ps.fit_predict(ds)
        ds.ca.ClustersSurprise = labels + min(labels)
        ds.ca.OutliersSurprise = (labels == -1).astype('int')
        logging.info(f"Found {ds.ca.Clusters.max() + 1} clusters")
        
        ## Annotate bins
        logging.info(f"Annotating Bins")
        Bin_annotation(ds, self.config.paths.ref)
        
        ## Plot results on manifold
        logging.info("Plotting UMAP")
        manifold(ds, os.path.join(self.outdir, f"{ds.attrs['tissue']}_bins_manifold_UMAP.png"), embedding = 'UMAP')
        logging.info("Plotting TSNE")
        manifold(ds, os.path.join(self.outdir, f"{ds.attrs['tissue']}_bins_manifold_TSNE.png"), embedding = 'TSNE')
        logging.info("plotting the number of UMIs")
        QC_plot(ds, os.path.join(self.outdir, f"{ds.attrs['tissue']}_bins_manifold_QC.png"), embedding = 'TSNE')