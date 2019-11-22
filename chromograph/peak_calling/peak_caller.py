import numpy as np
import os
import sys
import logging
import collections
import matplotlib.pyplot as plt
import gzip
import glob
import loompy
import urllib.request
import pybedtools
from pybedtools import BedTool
import MACS2

import sklearn.metrics
from scipy.spatial import distance
import community
import networkx as nx
from scipy import sparse
from typing import *

sys.path.append('/home/camiel/chromograph/')
from chromograph.peak_calling.utils import *
import chromograph.peak_calling.call_MACS

## Parameters
indir = '/data/proj/scATAC/chromograph/mouse_test3/'
f = os.path.join(indir, '10X_test_10kb.loom')
pf = os.path.join(indir, 'peaks')
macs_path = '/home/camiel/anaconda3/envs/chromograph/bin/macs2'
gf = None
    
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if not os.path.isdir(pf):
    os.mkdir(pf)

## Load data
ds = loompy.connect(f)

## Check organism
if ds.attrs['reference_organism'] == 'Homo_sapiens':
    g = 'hs'
elif ds.attrs['reference_organism'] == 'Mus_musculus':
    g = 'mm'
    
import multiprocessing as mp
 
# def call_MACS(data, pf):

#     clus = data[0]
#     fragments = data[1]
#     frags = [strFrags_to_list(x) for x in fragments]
#     frags = [x for l in frags for x in l]
#     logging.info("Total fragments in cluster {}:  {}".format(clus, len(frags)))

#     fbed = os.path.join(pf, "fragments_cluster_{}.bed.gz".format(clus))
#     fpeaks = os.path.join(pf, "cluster_{}".format(clus))

#     logging.info("{},  {}".format(g, fpeaks))
#     bed = BedTool(frags)
#     bed.saveas(fbed)

#     ## Call Peaks
#     cmd = "{} callpeak -t {} -f BEDPE -g {} --nomodel --shift 100 --ext 200 --qval 5e-2 -B --SPMR -n {}".format(macs_path, fbed, g, fpeaks)
#     os.system(cmd)

#     logging.info('Called peaks for cluster {} out of {}'.format(clus, np.unique(ds.ca['Clusters'])))
    
#     ## We only need the narrowPeak file, so clean up the rest
#     os.system("rm {}".format(os.path.join(pf, 'cluster_' + str(clus) + '_peaks.xls')))
#     os.system("rm {}".format(os.path.join(pf, 'cluster_' + str(clus) + '_control_lambda.bdg')))
#     os.system("rm {}".format(os.path.join(pf, 'cluster_' + str(clus) + '_summits.bed')))
 
#     return "Cluster {} completed".format(clus)

if __name__ == '__main__':
    jobs = []
    chunks = [[i, ds.ca['fragments'][ds.ca['Clusters']==i]] for i in np.unique(ds.ca['Clusters']) ]
    for chunk in chunks:
        p = mp.Process(target=call_MACS, args=(chunk, pf,))
        jobs.append(p)
        p.start()
        
## Compound the peak lists
peaks = [BedTool(x) for x in glob.glob(os.path.join(pf, '*.narrowPeak'))]
logging.info('Identified on average {} peaks per cluster'.format(np.int(np.mean([len(x) for x in peaks]))))
peaks_all = peaks[0].cat(*peaks[1:])
logging.info('Identified {} peaks after compounding list'.format(len(peaks_all)))

