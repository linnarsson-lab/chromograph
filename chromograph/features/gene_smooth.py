import numpy as np
import os
import sys
import loompy
import warnings
from numba import jit, njit
import numba

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
        ds.ra['GA_rowsum'] = ds.map([np.count_nonzero], axis=0)[0]
        ds.ca['GA_colsum'] = ds.map([np.count_nonzero], axis=1)[0]

        logging.info(f'Converting to FPKM')  # divide by BPs/1e3 and divide by GA_colsum/1e6
        ds['FPKM'] = 'float16'
        for (ix, selection, view) in ds.scan(axis=1):
            ds['FPKM'][:,selection] = div0(div0(view[''][:,:], 1e-3*ds.ra['BPs'].reshape(ds.shape[0], 1)), (1e-6 * ds.ca['GA_colsum'][selection]))
            logging.info(f"FPKM for: {max(selection)} cells out of {ds.shape[1]}")

        logging.info(f'Loading the network')
        bnn = BalancedKNN(k=self.config.params.k, metric='euclidean', maxl=2 * self.config.params.k, sight_k=2 * self.config.params.k, n_jobs=-1)
        bnn.bknn = ds.col_graphs.KNN

        logging.info('Smoothing over the graph')
        ds['smooth'] = 'float16'
        for (ix, selection, view) in ds.scan(axis=0):
            ds['smooth'][selection,:] = bnn.smooth_data(view['FPKM'][:,:], only_increase=False)
            logging.info(f'Smoothed {max(selection)} rows out of {ds.shape[0]}')

        logging.info(f'Finished smoothing')      
