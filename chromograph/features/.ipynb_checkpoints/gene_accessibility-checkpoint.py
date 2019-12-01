import numpy as np
import os
import sys
import collections
import matplotlib.pyplot as plt
import gzip
import loompy
# import scipy.sparse as sparse
import urllib.request
import pybedtools
from pybedtools import BedTool
import warnings
from sklearn.neighbors import NearestNeighbors
from matplotlib.collections import LineCollection

sys.path.append('/home/camiel/chromograph/')
# from chromograph.plotting.UMI_plot import UMI_plot
import chromograph
from chromograph.peak_calling.utils import *
from chromograph.features.count_genes import *

from umap import UMAP
import sklearn.metrics
from scipy.spatial import distance
import community
import networkx as nx
from scipy import sparse
from typing import *
import multiprocessing as mp
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

## Import path to the relevant 10X reference dataset
ref = '/data/ref/cellranger-atac/refdata-cellranger-atac-mm10-1.2.0'
indir = '/data/proj/scATAC/chromograph/mouse_test2/'
f = os.path.join(indir, '10X_test_10kb.loom')

## Connect to loompy session
ds = loompy.connect(f)

logging.info('Connected to loom file')

## Load transcripts
genes = BedTool(os.path.join(ref, 'regions', 'transcripts.bed'))

## Extract unique protein coding transcripts
coding = []
for x in genes:
    if x[6] == 'protein_coding':
        coding.append(x)
coding = BedTool(coding)
coding = coding.groupby(g = [4,1], c = [2,3,4], o = ['min', 'max', 'count'], full=True)

## Generate row features
rows = {'Gene': [], 'Chromosome': [], 'Start': [], 'End': []}
for x in coding:
    rows['Gene'].append(x[3])
    rows['Chromosome'].append(x[0])
    rows['Start'].append(int(x[1]))
    rows['End'].append(int(x[2]))
    
logging.info(f"Found {len(rows['Gene'])} genes")
    
if __name__ == '__main__':
    manager = mp.Manager()
    res = manager.dict()
    jobs = []
    
    ## Chunk by cell
    frags = [strFrags_to_list(x) for x in ds.ca['fragments']]
    chunks = [[k,v] for k,v in zip(ds.ca['cell_id'], frags)]

    logging.info(f"Chunked data")
    
    i = 0
    for chunk in chunks:
        p = mp.Process(target=count_genes, args=(chunk, rows, res,))
        jobs.append(p)
        p.start()
        
        i += 1
        
        if i%100 == 0:
            logging.info(f"processed {i} cells")
        
    for proc in jobs:
        proc.join()
        
    import pickle
    logging.info('save as pickle')
    pickle.dump(res, open('res_GAM.pkl', 'wb'))
