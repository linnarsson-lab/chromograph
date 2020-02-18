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
import cytograph as cg

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

@njit()
def smooth_jit(data, g) -> None:
    '''
    '''
    y = []
    for x in numba.prange(data.shape[1]):
        nn = g[:,x]>0
        nn_n = np.sum(nn)
        v = data[:,x] + np.sum(data[:,nn], axis=1) / (nn_n+1)
        y.append(v)
        
    return y

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

        """
        ds['smooth'] = 'float32'
        for (ix, selection, view) in ds.scan(axis=0):
            g = view.col_graphs['KNN'].toarray()
            vals = smooth_jit(view[:,:], g)
            vals = np.vstack(vals)
            ds['smooth'][ix:vals.shape[1],:] = vals.T
            logging.info(f"finished: {ix} rows")

        logging.info(f"Finished smoothing and saving to file")         
