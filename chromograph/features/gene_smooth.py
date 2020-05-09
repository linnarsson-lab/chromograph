import numpy as np
import os
import sys
import loompy
import warnings
from tqdm import tqdm

sys.path.append('/home/camiel/chromograph/')
import chromograph
from chromograph.pipeline import config
from chromograph.pipeline.utils import div0
import cytograph as cg
from cytograph.manifold import BalancedKNN

import sklearn.metrics
from scipy.spatial import distance
import networkx as nx
from scipy import sparse
from typing import *
import pickle

import logging
logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

class GeneSmooth:
    def __init__(self) -> None:
        """
        Smoothes Gene Accessibility values over the K-Nearest Neighbour network 
        Args:
            steps                    Which steps to include in the analysis

        Remarks:

        """
        self.config = config.load_config()
        logging.info("GeneSmooth initialised")

    def fit(self, ds: loompy.LoomConnection) -> None:
        """
        Smooth the GA data
        """
        ## Get the rowsums and colsums
        ds.ra['GA_rowsum'] = ds.map([np.sum], axis=0)[0]
        ds.ca['GA_colsum'] = ds.map([np.sum], axis=1)[0]

        logging.info(f'Converting to FPKM')  # divide by BPs/1e3 and divide by GA_colsum/1e6
        ds['FPKM'] = 'float32'
        progress = tqdm(total = ds.shape[1])
        for (ix, selection, view) in ds.scan(axis=1, batch_size=self.config.params.batch_size):
            ds['FPKM'][:,selection] = div0(div0(view[''][:,:], 1e-6 * ds.ca['GA_colsum'][selection]), (1e-3*ds.ra['BPs'].reshape(ds.shape[0], 1)))
            progress.update(self.config.params.batch_size)
        progress.close()

        logging.info(f'Loading the network')
        bnn = BalancedKNN(k=self.config.params.k, metric='euclidean', maxl=2 * self.config.params.k, sight_k=2 * self.config.params.k, n_jobs=-1)
        bnn.bknn = ds.col_graphs.KNN

        logging.info('Smoothing over the graph')
        ds['smooth'] = 'float32'
        progress = tqdm(total = ds.shape[0])
        for (ix, selection, view) in ds.scan(axis=0, batch_size=self.config.params.batch_size):
            ds['smooth'][selection,:] = bnn.smooth_data(view['FPKM'][:,:], only_increase=False)
            progress.update(self.config.params.batch_size)
        progress.close()

        ## Set FPKM as main matrix
        ds['raw'] = ds[''][:,:]
        ds[''] = np.nan_to_num(ds['FPKM'][:,:])
        logging.info(f'Finished smoothing') 
        return
