import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import loompy
# import scipy.sparse as sparse
import warnings

import chromograph
from chromograph.pipeline import config
from chromograph.pipeline.utils import div0
from chromograph.peak_calling.utils import *
from chromograph.peak_analysis.utils import *
from chromograph.plotting.motif_heatmap import motif_heatmap
import pandas as pd

import cytograph as cg
from typing import *

import logging
logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

class motif_aggregator():
    def __init__(self, name):
        self.name = name
        self.config = config.load_config()
        self.subset_dir = os.path.join(self.config.paths.build, self.name)
        self.peak_file = os.path.join(self.subset_dir, f'{self.name}_peaks.agg.loom')
        self.motif_dir = os.path.join(self.subset_dir, f'motifs/')
        self.motif_file = os.path.join(self.subset_dir, f'{self.name}_motifs.agg.loom')
        self.RNA_file = os.path.join(self.subset_dir, f'{self.name}_RNA.agg.loom')
        self.motif_plot = os.path.join(self.subset_dir, f'exported/{name}_motif_heatmap.png')
    
    def fit(self):
        '''

        '''        
        motif_outputs = {}
        for cluster in os.listdir(self.motif_dir):
            motif_outputs[cluster] = np.genfromtxt(os.path.join(self.motif_dir, cluster, 'knownResults.txt'), dtype = str, skip_header=1)
            
        TFs = [motif_outputs[k][:,0] for k in motif_outputs]
        TFs = np.array([x for s in TFs for x in s])
        consensus = [motif_outputs[k][:,1] for k in motif_outputs]
        consensus = np.array([x for s in consensus for x in s])
        TF_unique = np.unique(TFs)
        consensus_unique = []

        for TF in TF_unique:
            consensus_unique.append(consensus[np.where(TFs==TF)[0][0]])
        TF, Cluster, Domain = [x.flatten() for x in np.split(np.array([x.split('.') for x in TF_unique]), 3, axis=1)]
            
        if os.path.exists(self.motif_file):
            os.remove(self.motif_file)
        with loompy.connect(self.peak_file) as ds:
            matrix = sparse.coo_matrix((len(TF_unique), ds.shape[1]))
            row_attrs = {'TF': TF, 'Consensus': consensus_unique, 'Cluster': Cluster, 'Domain': Domain}
            col_attrs = ds.ca
            loompy.create(self.motif_file, matrix, row_attrs, col_attrs)

        with loompy.connect(self.motif_file) as ds:
            ds.layers['log_pval'] = 'float32'
            ds.layers['qval'] = 'float32'
            for k in motif_outputs:
                cl = int(k.split('_')[-1])
                mat = pd.DataFrame(motif_outputs[k], columns=('TF', 'Consensus', 'P_val', 'Log P_val', 'q_val', '# of Target Sequences with Motif(of 2000)', 
                                                '% of Target Sequences with Motif', '# of Background Sequences with Motif(of 46941)', '% of Background Sequences with Motif'))
                mat = mat.sort_values('TF')
                
                ds[:,cl] = np.array(mat['P_val']).astype(float)
                ds['log_pval'][:,cl] = mat['Log P_val'].astype(float)
                ds['qval'][:,cl] = mat['q_val'].astype(float)

            ds['-log_pval'] = -ds['log_pval'][:,:]

            if os.path.isfile(self.RNA_file):
                logging.info(f'Removing motifs of TFs that are not expressed')
                with loompy.connect(self.RNA_file) as dsa:
                    valid_genes = np.array([g for g in ds.ra.TF if g in dsa.ra.Gene])
                    skipped = np.array([g for g in ds.ra.TF if g not in dsa.ra.Gene])
                    if len(skipped)> 0:
                        logging.info(f'No expression data for {skipped}')

                    x = np.where(np.isin(ds.ra.TF, valid_genes))[0]
                    
                    X = ds['-log_pval'][:,:]
                    select = np.array([np.where(dsa.ra.Gene==TF)[0] for TF in ds.ra.TF[x]])
                    trinaries = np.ones(ds.shape)
                    v = np.ones((len(x),ds.shape[1]))
                    v[:,dsa.ca.Clusters] = dsa['trinaries'][:,:][select,:].reshape((len(x),dsa.shape[1]))
                    trinaries[x,:] = v
                        
                    ds['-log_pval_trinaries'] = X * trinaries
            else:
                logging.info('No RNA file')

            motif_heatmap(ds, self.motif_plot, N=5)