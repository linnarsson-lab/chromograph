import numpy as np
import os
import sys
import collections
import matplotlib.pyplot as plt
import gzip
import loompy
import scipy.sparse as sparse
import urllib.request
import warnings
import logging
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

from cytograph.manifold import BalancedKNN
from cytograph.metrics import jensen_shannon_distance
from cytograph.embedding import art_of_tsne
from cytograph.clustering import PolishedLouvain
from cytograph.plotting import manifold

from chromograph.plotting.QC_plot import QC_plot
from chromograph.features.bin_annotation import Bin_annotation
from chromograph.plotting.sample_distribution_plot import sample_distribution_plot
from chromograph.pipeline.TF_IDF import TF_IDF
from chromograph.pipeline.PCA import PCA
from chromograph.pipeline.SVD import SVD
from chromograph.pipeline.utils import *
from chromograph.pipeline import config

from umap import UMAP
from joblib import parallel_backend
import sklearn.metrics
from scipy.spatial import distance
import community
import networkx as nx
from scipy import sparse
from typing import *
from tqdm import tqdm

class Bin_analysis:
    def __init__(self, outdir, do_UMAP=True) -> None:
        """
        Perform Dimensional Reduction and Clustering on a binned loom-file   
        Args:
            steps                    Which steps to include in the analysis

        Remarks:
            # All parameters are obtained from the config object, which comes from the default config
            # and can be overridden by the config in the current punchcard
        """
        self.config = config.load_config()
        self.outdir = os.path.join(outdir, 'exported')
        self.blayer = '5kb_bins'
        self.UMAP = do_UMAP
        logging.info(f"Bin_Analysis initialised, saving plots to {self.outdir}")
    
    def fit(self, ds: loompy.LoomConnection) -> None:        
        try:
            self.blayer = '{}kb_bins'.format(int(ds.attrs['bin_size'] / 1000))
        except:
            ds.attrs['bin_size'] = int(int(ds.ra.end[0]) - int(ds.ra.start[0]) + 1)
            self.blayer = '{}kb_bins'.format(int(ds.attrs['bin_size'] / 1000))
        logging.info(f'Running Chromograph Bin-analysis on {ds.shape[1]} cells with {ds.shape[0]} bins of size {self.blayer}')

        ## Get the output folder
        name = ds.filename.split("/")[-1].split(".")[0].split("_")[0]
    
        if not os.path.isdir(self.outdir):
            logging.info(f'Creating dir {self.outdir}')
            os.mkdir(self.outdir)
        
        if 'NCells' not in ds.ra or 'NBins' not in ds.ca:
            ## nonzero (nnz) counts per bin
            logging.info('Calculating bin and cell coverage')
            ds.ra['NCells'] = ds.map([np.count_nonzero], axis=0)[0]
            ds.ca['NBins'] = ds.map([np.count_nonzero], axis=1)[0]

        if not 'FRtss' in ds.ca:
            ds.ca.FRtss = div0(ds.ca.TSS_fragments, ds.ca.passed_filters)
        
        ## Calculate coverage
        cov = np.log10(ds.ra['NCells']+1)
        mu = np.mean(cov)
        sd = np.std(cov)
        ds.ra['Coverage'] = (cov - mu) / sd
        del cov, mu, sd
        
        ## Select bins for PCA fitting
        autosom = ~np.isin(ds.ra.chrom, ['chrX', 'chrY'])
        ds.ra.Valid = np.array((ds.ra['NCells'] > np.quantile(ds.ra['NCells'][autosom], self.config.params.bin_quantile)) & 
                                (ds.ra['NCells'] < (0.6*ds.shape[1])) & 
                                autosom)
        ds.attrs['bin_max_cutoff'] = max(ds.ra['NCells'][ds.ra.Valid==1])
        ds.attrs['bin_min_cutoff'] = min(ds.ra['NCells'][ds.ra.Valid==1])

        ## Create binary layer
        if self.blayer not in ds.layers:
            logging.info("Binarizing the matrix")
            ds.layers[self.blayer] = 'int8'

            ## Binarize in loop
            progress = tqdm(total=ds.shape[1])
            for (_, selection, view) in ds.scan(axis=1, batch_size=self.config.params.batch_size):
                ds[self.blayer][:,selection] = view[:,:] > 0
                progress.update(self.config.params.batch_size)
            progress.close()

        ## Faster LSI
        f_temp = ds.filename + '.tmp'
        if os.path.isfile(f_temp):
            os.remove(f_temp)
        with loompy.new(f_temp) as dst:
            logging.info(f'Creating temp file for faster LSI')
            x = np.where(ds.ra.Valid)[0]
            for (ix, selection, view) in tqdm(ds.scan(layers = [self.blayer], axis=1)):
                dst.add_columns(view[self.blayer][x,:], col_attrs=view.ca, row_attrs={'loc': ds.ra.loc[x]})
            dst.ra.Valid = np.ones(dst.shape[0])

            ## Term-Frequence Inverse-Data-Frequency ##
            logging.info(f'Performing TF-IDF on {dst.shape}')
            tf_idf = TF_IDF(layer='')
            tf_idf.fit(dst)
            dst.layers['TF-IDF'] = 'float16'
            progress = tqdm(total=dst.shape[1])
            for (_, selection, view) in dst.scan(axis=1, batch_size=self.config.params.batch_size):
                dst['TF-IDF'][:,selection] = tf_idf.transform(view[:,:], selection)
                progress.update(self.config.params.batch_size)
            progress.close()
            del tf_idf
            logging.info(f'Finished fitting TF-IDF')

            ## Fit PCA
            logging.info(f'Fitting PCA')
            pca = PCA(max_n_components = self.config.params.n_factors, layer= '', key_depth= 'NBins', batch_keys = self.config.params.batch_keys)
            pca.fit(dst)

            ## Decompose data
            ds.ca.LSI_b = pca.transform(dst)

            logging.info(f'Finished PCA transformation')
            del pca

            ## Get correct embedding and metric
            decomp = ds.ca.LSI_b
        os.remove(f_temp) ## Remove temp file

        ## Construct nearest-neighbor graph
        metric = self.config.params.f_metric # jaccard js euclidean correlation cosine 
        logging.info(f"Computing balanced KNN (k = {self.config.params.k}) space using the '{metric}' metric")
        metric_f = (jensen_shannon_distance if metric == "js" else metric)  # Replace js with the actual function, since OpenTSNE doesn't understand js
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
        logging.info(f"  Art of tSNE with {metric} distance metric")
        ds.ca.TSNE = np.array(art_of_tsne(decomp, metric=metric_f))  # art_of_tsne returns a TSNEEmbedding, which can be cast to an ndarray (its actually just a subclass)
        if self.UMAP==True:
            logging.info("Generating UMAP from decomposition")
            ds.ca.UMAP = UMAP(n_components=2, metric=metric_f, verbose=True).fit_transform(decomp)
            logging.info("Generating 3D UMAP from decomposition")
            ds.ca.UMAP3D = UMAP(n_components=3, metric=metric_f, verbose=True).fit_transform(decomp)

        ## Perform Clustering
        logging.info("Performing Polished Louvain clustering")
        if name == 'All':
            mn = self.config.params.min_cells_precluster
        else:
            mn = 10
        pl = PolishedLouvain(outliers=False, graph="RNN", embedding="TSNE", resolution = self.config.params.resolution, min_cells=mn)
        labels = pl.fit_predict(ds)
        ds.ca.ClustersModularity = labels + min(labels)
        ds.ca.OutliersModularity = (labels == -1).astype('int')
        ds.ca.Clusters = labels + min(labels) ## Will be overwritten
        ds.ca.preClusters = ds.ca.Clusters
        ds.ca.Outliers = (labels == -1).astype('int')
        
        ## Annotate bins
        logging.info(f"Annotating Bins")
        Bin_annotation(ds, self.config.paths.ref)
        
        ## Plot results on manifold
        logging.info("Plotting TSNE")
        manifold(ds, os.path.join(self.outdir, f"{name}_bins_TSNE.png"), embedding = 'TSNE')
        QC_plot(ds, os.path.join(self.outdir, f"{name}_bins_TSNE_QC.png"), embedding = 'TSNE', attrs=self.config.params.plot_attrs)

        if name == 'All':
            logging.info(f'Plotting sample distribution')
            sample_distribution_plot(ds, os.path.join(self.outdir, f"{name}_cell_counts.png"))
