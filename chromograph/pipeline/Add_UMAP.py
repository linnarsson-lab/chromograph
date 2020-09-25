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
from cytograph.plotting import manifold

from chromograph.plotting.QC_plot import QC_plot
from chromograph.pipeline import config

from umap import UMAP
import sklearn.metrics
from scipy.spatial import distance
import networkx as nx
from typing import *
from tqdm import tqdm

class Add_UMAP:
    def __init__(self, outdir, feature) -> None:
        """
        Add UMAP embedding to existing analysis

        Args:
            feature      name of feature matrix
        """
        self.config = config.load_config()
        self.outdir = os.path.join(outdir, 'exported')
        self.feature = feature
        logging.info(f"Bin_Analysis initialised, saving plots to {self.outdir}")
    
    def fit(self, ds: loompy.LoomConnection) -> None:        
        name = ds.filename.split("/")[-1].split(".")[0].split("_")[0]
        
        if self.feature == 'bins':
            decomp = ds.ca.LSI_b
        elif self.features == 'peaks':
            decomp = ds.ca.LSI
        metric = 'euclidean'

        ## Perform tSNE and UMAP
        logging.info(f"Computing 2D and 3D embeddings from latent space")
        metric_f = (jensen_shannon_distance if metric == "js" else metric)  # Replace js with the actual function, since OpenTSNE doesn't understand js
        if self.UMAP==True:
            logging.info("Generating UMAP from decomposition")
            ds.ca.UMAP = UMAP(n_components=2, metric=metric_f, n_neighbors=self.config.params.k // 2, learning_rate=0.3, min_dist=0.25, init='random', verbose=True).fit_transform(decomp)
            logging.info("Generating 3D UMAP from decomposition")
            ds.ca.UMAP3D = UMAP(n_components=3, metric=metric_f, n_neighbors=self.config.params.k // 2, learning_rate=0.3, min_dist=0.25, init='random', verbose=True).fit_transform(decomp)
       
        ## Plot results on manifold
        logging.info("Plotting UMAP")
        manifold(ds, os.path.join(self.outdir, f"{name}_{self.feature}_manifold_UMAP.png"), embedding = 'UMAP')
        QC_plot(ds, os.path.join(self.outdir, f"{name}_{self.feature}_manifold_QC_UMAP.png"), embedding = 'UMAP', attrs=['Age', 'Shortname', 'Chemistry', 'Tissue'])
