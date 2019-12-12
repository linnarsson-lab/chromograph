import numpy as np
import os
import sys
import collections
import matplotlib.pyplot as plt
import loompy
# import scipy.sparse as sparse
import urllib.request
import pybedtools
from pybedtools import BedTool
import warnings
# from sklearn.neighbors import NearestNeighbors
import statsmodels as sm
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.stats.proportion import proportions_ztest
from numba import jit, njit
import numba

sys.path.append('/home/camiel/chromograph/')
# from chromograph.plotting.UMI_plot import UMI_plot
import chromograph
from chromograph.peak_calling.utils import *

import cytograph as cg

from umap import UMAP
import sklearn.metrics
from scipy.spatial import distance
import community
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

## Connect to file
f = '/data/proj/scATAC/chromograph/build_20191205/Midbrain.loom'
ref = '/data/ref/cellranger-atac/refdata-cellranger-atac-GRCh38-1.2.0/'
ds = loompy.connect(f, mode='r')
blayer = str(int(ds.attrs['bin_size']/1000)) + 'kb_bins'
logging.info(ds.shape)

g = ds.col_graphs['MKNN'].toarray()
smooth = []
batch = 10000
i = 0

# Parallel version
# @jit(parallel=True)
@njit
def smooth_bins_jit(data, g) -> None:
    '''
    '''
    vals = []
    for x in numba.prange(data.shape[1]):
        nn = g[:,x]>0
        nn_n = np.sum(nn)
        v = data[:,x] + np.sum(data[:,nn], axis=1) / (nn_n+1)
        vals.append(v)
        
    return vals


for (ix, selection, view) in ds.scan(axis=0, batch_size=batch):
    g = view.col_graphs['MKNN'].toarray()
    vals = smooth_bins_jit(view[blayer][:,:], g)
    vals = np.array(vals)
    smooth.append(vals)
    i+= batch
    if i %(batch*5) == 0:
        logging.info(f"finished{i} rows")

pickle.dump(smooth, 'smooth.pkl')