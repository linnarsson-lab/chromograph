import numpy as np
import pandas as pd
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
from tqdm import tqdm
from numba.core.errors import NumbaPerformanceWarning
import multiprocessing as mp
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

from cytograph.decomposition import HPF
from cytograph.manifold import BalancedKNN
from cytograph.metrics import jensen_shannon_distance
from cytograph.embedding import art_of_tsne
from cytograph.clustering import UnpolishedLouvain, PolishedLouvain
from cytograph.plotting import manifold

from chromograph.plotting.QC_plot import QC_plot
from chromograph.pipeline import config
from chromograph.pipeline.utils import *
from chromograph.pipeline.TF_IDF import TF_IDF
from chromograph.pipeline.PCA import PCA
from chromograph.peak_analysis.feature_selection_by_variance import FeatureSelectionByVariance
from chromograph.peak_analysis.feature_selection_by_PearsonResiduals import FeatureSelectionByPearsonResiduals
from chromograph.peak_analysis.utils import *

from pynndescent import NNDescent
from umap import UMAP
import sklearn.metrics
from scipy.spatial import distance
from harmony import harmonize
import community
import networkx as nx
from scipy.stats import poisson
from sknetwork.clustering import Louvain
from scipy import sparse
from typing import *

class Peak_analysis:
    def __init__(self, outdir, do_UMAP=True) -> None:
        """
        Perform Dimensional Reduction and Clustering on a Peak loom-file   
        Args:
            steps                    Which steps to include in the analysis

        Remarks:
            --- PUNCHCARD SUPPORT NEEDS TO INTEGRATED, FOR NOW PASS PARAMETER TO FIT FUNCTION ---
            # All parameters are obtained from the config object, which comes from the default config
            # and can be overridden by the config in the current punchcard
        """
        self.config = config.load_config()
        self.outdir = os.path.join(outdir, 'exported')
        self.layer = ''
        self.depth_key = 'NPeaks'
        self.do_UMAP = do_UMAP
        logging.info("Peak_analysis initialised")
    
    def fit(self, ds: loompy.LoomConnection) -> None:
        logging.info(f"Running Peak_analysis on {ds.shape[1]} cells with {ds.shape[0]} peaks")
        name = '_'.join(ds.filename.split("/")[-1].split(".")[0].split("_")[:-1])
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        
        ## nonzero (nnz) counts per peak
        logging.info('Calculating peak and cell coverage')
        ds.ra['NCells'] = ds.map([np.count_nonzero], axis=0)[0]
        if not 'NPeaks' in ds.ca:
            ds.ca['NPeaks'] = ds.map([np.count_nonzero], axis=1)[0]
            ds.ca['Fragments_in_peaks'] = ds['Counts'].map([np.sum], axis=1)[0]
            ds.ca['FRIP'] = div0(ds.ca.Fragments_in_peaks, ds.ca.passed_filters)

        ## Select peaks for manifold learning based on variance between pre-clusters
        logging.info('Select Peaks for manifold learning by variance in preclusters')

        valid_clusters = select_preclusters(ds, min_cells=self.config.params.min_cells_cluster, min_clusters=25, Always_iterative=self.config.params.Always_iterative)
        valid = np.array([x in valid_clusters for x in ds.ca.preClusters])

        ## Aggregate
        logging.info(f'Aggregating file')
        temporary_aggregate = os.path.join(self.config.paths.build, name, name + '_tmp.agg.loom')
        if np.sum(valid) != ds.shape[1]:
            temporary_file = os.path.join(self.config.paths.build, name, name + '_tmp.loom')
            ds.aggregate(temporary_file, None, "preClusters", "sum", {"preClusters": "first"})
            with loompy.connect(temporary_file) as ds_temp:
                with loompy.new(temporary_aggregate) as dsagg:
                    valid = np.array([x in valid_clusters for x in ds_temp.ca.preClusters])
                    for (ix, selection, view) in ds_temp.scan(items=np.where(valid)[0], axis=1):
                        dsagg.add_columns(view.layers, col_attrs=view.ca, row_attrs=view.ra)
            os.remove(temporary_file)

        else:
            ds.aggregate(temporary_aggregate, None, "preClusters", "sum", {"preClusters": "first"})
        
        ## Feature selection
        with loompy.connect(temporary_aggregate) as dsout:
            ## Normalize peak counts by total fragments per cluster
            dsout.ca.Total = dsout.map([np.sum], axis=1)[0]
            logging.info('Convert to CPMs')
            dsout.layers['CPM'] = div0(dsout[''][:,:], dsout.ca.Total * 1e-6)
            logging.info('Calculating variance')
            (ds.ra.preCluster_mu, ds.ra.preCluster_sd) = dsout['CPM'].map((np.mean, np.std), axis=0)
            logging.info(f'Selecting {self.config.params.N_peaks_decomp} peaks using {self.config.params.feature_selection} for clustering')
            dsout.ra.Valid = ((ds.ra.NCells / ds.shape[1]) > self.config.params.peak_fraction) & (ds.ra.NCells < np.quantile(ds.ra.NCells, 0.99))
            if self.config.params.feature_selection == 'Pearson_Residuals':
                ds.ra.Valid, ds.ra.preCluster_residuals = FeatureSelectionByPearsonResiduals(n_genes=self.config.params.N_peaks_decomp, mask=np.isin(ds.ra.Chr, ['chrX', 'chrY'])).fit(dsout)
            elif self.config.params.feature_selection == 'Variance':
                ds.ra.Valid = FeatureSelectionByVariance(n_genes=self.config.params.N_peaks_decomp, layer='CPM', mask=np.isin(ds.ra.Chr, ['chrX', 'chrY'])).fit(dsout)
        ## Delete temporary file
        os.remove(temporary_aggregate)
        
        ## HPF for manifold learning
        if self.config.params.peak_factorization == 'HPF':
            logging.info(f'Performing HPF, layer = {self.layer}')
            
            # Load the data for the selected genes
            logging.info(f"Selecting data for HPF factorization: {ds.shape[1]} cells and {sum(ds.ra.Valid)} peaks")
            data = ds[self.layer].sparse(rows=np.array(ds.ra.Valid==1)).T
            logging.info(f"Shape data: {data.shape} with {data.nnz} values")

            # HPF factorization
            
            cpus = 4
            logging.info(f'Performing HPF factorization with {self.config.params.HPF_factors} factors')
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

            ## Correct the normalized theta values using Harmony
            logging.info(f'Batch correcting using Harmony')
            keys_df = pd.DataFrame.from_dict({k: ds.ca[k] for k in self.config.params.batch_keys})
            theta = harmonize(theta, keys_df, batch_key=self.config.params.batch_keys, n_jobs_kmeans=1)
            theta = (theta.T / theta.sum(axis=1)).T

            # Save the normalized factors
            ds.ra.HPF = beta_all
            ds.ca.HPF = theta
            del hpf

            decomp = ds.ca.HPF

        ## Term-Frequence Inverse-Data-Frequency ##
        if self.config.params.peak_factorization == 'LSI':
            ## Faster LSI
            f_temp = ds.filename + '.tmp'
            if os.path.isfile(f_temp):
                os.remove(f_temp)
            logging.info(f'Making temp file')

            x = np.where(ds.ra.Valid)[0]
            loompy.create(f_temp, {'':ds.sparse(rows=x).tocsr()}, {'ID': ds.ra.ID[x]}, col_attrs=ds.ca)
            with loompy.connect(f_temp) as dst:
                dst.ra.Valid = np.ones(dst.shape[0])

                ## Term-Frequence Inverse-Data-Frequency ##
                logging.info(f'Performing TF-IDF')
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
                pca = PCA(max_n_components = self.config.params.n_factors, layer= '', key_depth= self.depth_key, batch_keys = self.config.params.batch_keys)
                pca.fit(dst)

                ## Decompose data
                ds.ca.LSI = pca.transform(dst)

                logging.info(f'Finished PCA transformation')
                del pca

                ## Get correct embedding and metric
                decomp = ds.ca.LSI
            os.remove(f_temp)

        ## Construct nearest-neighbor graph
        metric = self.config.params.f_metric # jaccard js euclidean correlation cosine 
        add_graphs(ds, decomp, metric, self.config.params.k)

        ## Calculate Pseudo-age
        if 'Age' in ds.ca:
            knn = ds.col_graphs.KNN
            k = knn.nnz / knn.shape[0]
            ds.ca.PseudoAge = (knn.astype("bool") @ ds.ca.Age) / k
            del knn

        ## Perform tSNE and UMAP
        logging.info(f"Computing 2D and 3D embeddings from latent space")
        logging.info(f"Art of tSNE with distance metrid: {metric}")
        ds.ca.TSNE = np.array(art_of_tsne(decomp, metric=metric, exaggeration=2))  # art_of_tsne returns a TSNEEmbedding, which can be cast to an ndarray (its actually just a subclass)
        
        if self.do_UMAP:
            logging.info(f'Generating UMAP from decomposition using metric {metric}')
            ds.ca.UMAP = UMAP(n_components=2, metric=metric, verbose=True).fit_transform(decomp)
            logging.info(f'Generating 3D UMAP from decomposition using metric {metric}')
            ds.ca.UMAP3D = UMAP(n_components=3, metric=metric, verbose=True).fit_transform(decomp)

        ## Perform Clustering
        if self.config.params.clusterer == 'unpolished':
            logging.info("Performing unpolished Louvain clustering")
            pl = UnpolishedLouvain(graph=self.config.params.graph, embedding=self.config.params.main_emb, min_cells=self.config.params.min_cells_cluster)
            labels = pl.fit_predict(ds)
            ds.ca.Clusters = labels + min(labels)
            ds.ca.Outliers = (labels == -1).astype('int')
        elif self.config.params.clusterer == 'sknetwork':
            logging.info("Performing unpolished Louvain clustering using SKnetwork")
            G = nx.from_scipy_sparse_matrix(ds.col_graphs[self.config.params.graph])
            adj = nx.linalg.graphmatrix.adjacency_matrix(G)
            labels = Louvain(random_state=0).fit_transform(adj)
            ds.ca.Clusters = labels + min(labels)
            ds.ca.Outliers = (labels == -1).astype('int')
        elif self.config.params.clusterer == 'polished':
            logging.info("Clustering by polished Louvain")
            pl = PolishedLouvain(outliers=False, graph=self.config.params.graph, embedding=self.config.params.main_emb, min_cells=self.config.params.min_cells_cluster)
            labels = pl.fit_predict(ds)
            ds.ca.Clusters = labels + min(labels)
            ds.ca.Outliers = (labels == -1).astype('int')
        else:
            raise Exception("No clustering set!!!")

        logging.info(f"Found {ds.ca.Clusters.max() + 1} clusters")

        ## Plot results on manifold
        logging.info("Plotting TSNE")
        manifold(ds, os.path.join(self.outdir, f"{name}_peaks_TSNE.png"), embedding = 'TSNE')
        try:
            QC_plot(ds, os.path.join(self.outdir, f"{name}_peaks_TSNE_QC.png"), embedding = 'TSNE', attrs=self.config.params.plot_attrs)        
        except:
            pass
        if 'UMAP' in ds.ca:
            logging.info("Plotting UMAP")
            manifold(ds, os.path.join(self.outdir, f"{name}_peaks_UMAP.png"), embedding = 'UMAP')
            try:
                QC_plot(ds, os.path.join(self.outdir, f"{name}_peaks_UMAP_QC.png"), embedding = 'UMAP', attrs=self.config.params.plot_attrs)
            except:
                pass
