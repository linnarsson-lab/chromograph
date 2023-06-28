import numpy as np
import os
import sys
import loompy
import pickle as pkl

import chromograph
from chromograph.peak_calling.utils import *
from chromograph.pipeline.utils import *
from chromograph.pipeline import config

from cytograph.embedding import art_of_tsne
from cytograph.manifold import BalancedKNN
from umap import UMAP

import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import NearestNeighbors
from sknetwork.clustering import Louvain
from scipy.stats import mode
from pynndescent import NNDescent
import networkx as nx
import community
import igraph as ig
import logging


logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')


class peak_clustering:
    def __init__(self, k:int=25, min_feat:int=50) -> None:
        '''
        Identify cCRE modules
        '''
        self.k = k
        self.min_feat = min_feat
        self.config = config.load_config()

    def fit(self, ds: loompy.LoomConnection):
        '''
        Fit peak clustering
        '''
        X = ds['residuals'][:,:].astype(np.float32)

        logging.info(f'Building graph')
        bnn = BalancedKNN(k=self.k, metric='euclidean', maxl= 50, sight_k=50, n_jobs=-1)
        bnn.fit(X)
        knn = bnn.kneighbors_graph(mode='distance')
        knn.eliminate_zeros()

        ## TSNE
        logging.info(f'Running art of tsne')
        ds.ra.TSNE = np.array(art_of_tsne(X))

        ## UMAP
        logging.info(f'Generating UMAP')
        ds.ra.UMAP = UMAP(n_components=2, metric='euclidean', verbose=True).fit_transform(X)

        ## Cluster
        logging.info(f'Clustering peaks')
        G = nx.from_scipy_sparse_matrix(knn)
        adj = nx.linalg.graphmatrix.adjacency_matrix(G)
        labels = Louvain(random_state=0).fit_transform(adj)
        labels = labels + min(labels)
        logging.info(f'Found: {len(np.unique(labels))} clusters')

        # Set the local cluster label to the local majority vote
        logging.info("Smoothing cluster identity on the embedding")
        xy = ds.ra.TSNE
        nn = NNDescent(data=xy, n_jobs=-1, random_state=0, n_neighbors=self.k)
        indices, distances = nn.query(xy, k=25)
        labels = mode(labels[indices], axis=1)[0].flatten()

        # Mark tiny clusters as outliers
        logging.info("Marking tiny clusters as outliers")
        ix, counts = np.unique(labels, return_counts=True)
        labels[np.isin(labels, ix[counts < self.min_feat])] = -1

        # Renumber the clusters (since some clusters might have been lost in poor neighborhoods)
        retain = list(set(labels))
        if -1 not in retain:
            retain.append(-1)
        retain = sorted(retain)
        d = dict(zip(retain, np.arange(-1, len(set(retain)))))
        labels = np.array([d[x] if x in d else -1 for x in labels])

        if np.any(labels == -1):
            # Assign each outlier to the same cluster as the nearest non-outlier
            logging.info(f'Assigning outliers to closest neighbors')
            nn = NearestNeighbors(n_neighbors=self.k, algorithm="ball_tree")
            nn.fit(xy[labels >= 0])
            nearest = nn.kneighbors(xy[labels == -1], n_neighbors=1, return_distance=False)
            labels[labels == -1] = labels[labels >= 0][nearest.flat[:]]

        ds.ra.module = labels
        logging.info(f'finished, total clusters{len(np.unique(labels))}')