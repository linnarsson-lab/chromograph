import loompy
import os
import subprocess
import gc
import sys
import numpy as np
from datetime import datetime
import logging
from typing import *
from tqdm import tqdm

import pickle as pkl
import gzip
import glob
import pybedtools
from pybedtools import BedTool
import shutil
import multiprocessing as mp
from pynndescent import NNDescent

## Import chromograph
import chromograph
from chromograph.pipeline.Bin_analysis import *
from chromograph.pipeline import config
from chromograph.pipeline.utils import transfer_ca, div0
from chromograph.features.GA_Aggregator import GA_Aggregator
from chromograph.peak_calling.peak_caller import *
from chromograph.peak_calling.utils import *
from chromograph.features.utils import *

## Setup logger and load config
logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

class Generate_promoter:
    def __init__(self, outdir, poisson_pooling=True) -> None:
        """
        Use Gene reference to generate Gene accessibility scores as a gene expression proxy
        
        Args:
            ds                    Loom connection

        Remarks:
        
        """
        self.config = chromograph.pipeline.config.load_config()
        self.outdir = outdir
        self.peakdir = os.path.join(outdir, 'peaks')
        self.gene_ref = os.path.join(chromograph.__path__[0], 'references/GRCh38_2kbprom.bed')
        self.poisson_pooling = poisson_pooling
        self.loom = ''
    
    def fit(self, ds: loompy.LoomConnection) -> None:
        '''
        Use Gene reference to generate Gene accessibility scores as a gene expression proxy
        
        Args:

        Returns:
        
        Remarks:
        
        '''
        ## Get sample name from loom-file
        name = ds.filename.split('/')[-2]
        self.loom = os.path.join(self.outdir, f'{name}_prom.loom')

        ## Check if location for peaks and compounded fragments exists
        if not os.path.isdir(self.peakdir):
            os.mkdir(self.peakdir)   

        if not os.path.exists(self.loom):
            ## Check All_peaks.loom exists, get subset
            all_prom_loom = os.path.join(self.config.paths.build, 'All', 'All_prom.loom')

            if os.path.exists(all_prom_loom) & (all_prom_loom != self.loom):
                logging.info(f'Main promoter matrix already exists')
                
                with loompy.connect(all_prom_loom) as dsp:
                    selection = np.array([x in ds.ca.CellID for x in dsp.ca.CellID])
                
                loompy.combine_faster([all_prom_loom], self.loom, selections=[selection])
                
                with loompy.connect(self.loom) as ds2:
                    transfer_ca(ds, ds2, 'CellID')
                logging.info(f'Finished creating promoter file')
            
            else:
                logging.info(f'Check if promoter files exist')
                inputfiles = [os.path.join(config.paths.samples, sample, f"{sample}_prom.loom") for sample in np.unique(ds.ca.Name)]
                selections = []
                for file in inputfiles:
                    ## Check if file with right binning exists
                    if not os.path.exists(file):
                        logging.info(f"Generating promoter file for {file.split('/')[-2]}")
                        file_5kb = glob.glob(f"/{os.path.join(*file.split('/')[:-1])}/*5kb.loom")[0]
                        generate_prom_matrix(file_5kb, file, self.gene_ref, self.peakdir, self.config.paths.samples)

                    ## Get cells passing filters
                    with loompy.connect(file, 'r') as ds2:
                        good_cells = np.array([x in ds.ca.CellID for x in ds2.ca.CellID])
                        selections.append(good_cells)
                        
                logging.info(f'Combining samples')
                loompy.combine_faster(inputfiles, self.loom, selections=selections, key = 'loc')

                logging.info(f'Transferring column attributes')
                
                with loompy.connect(self.loom) as ds2:
                    transfer_ca(ds, ds2, 'CellID')
                logging.info(f'Loom promoter file saved as {self.loom}')

                ## Clean up stranded pybedtools tmp files
                pybedtools.helpers.cleanup(verbose=True, remove_all=True)
    
        ## Generate normalized layer
        with loompy.connect(self.loom, 'r+') as dsp:
            
            logging.info(f'Converting to CPM')  # divide by GA_colsum/1e6
            dsp.ca['GA_colsum'] = dsp[''].map([np.sum], axis=1)[0]
            
            ## Poisson pooling
            if self.poisson_pooling:
                logging.info(f"Poisson pooling")
                if 'HPF' in dsp.ca:
                    decomp = dsp.ca.HPF
                    decomp_type = 'HPF'
                elif 'LSI' in dsp.ca:
                    decomp = dsp.ca.LSI
                    decomp_type = 'LSI'
                logging.info(f'Create NN graph using {decomp_type}')
                nn = NNDescent(data=decomp, metric="euclidean", n_neighbors=self.config.params.k_pooling, verbose=True, n_jobs=1)
                logging.info(f'Query NN graph')
                indices, distances = [x.copy() for x in nn.neighbor_graph]
                # Note: we convert distances to similarities here, to support Poisson smoothing below
                knn = sparse.csr_matrix(
                    (np.ravel(distances), np.ravel(indices), np.arange(0, distances.shape[0] * distances.shape[1] + 1, distances.shape[1])), (decomp.shape[0], decomp.shape[0]))
                max_d = knn.data.max()
                knn.data = (max_d - knn.data) / max_d
                knn.setdiag(1)  # This causes a sparse efficiency warning, but it's not a slow step relative to everything else
                knn = knn.astype("bool")

                ## Start pooling over the network
                logging.info(f'Start pooling over network')
                dsp["pooled"] = 'int32'
                for (_, indexes, view) in dsp.scan(axis=0, layers=[""], what=["layers"]):
                    dsp["pooled"][indexes.min(): indexes.max() + 1, :] = view[:, :] @ knn.T
                dsp.ca['GA_pooled_colsum'] = dsp['pooled'].map([np.sum], axis=1)[0]

            dsp['CPM'] = 'float32'
            logging.info(f'Start conversion')
            progress = tqdm(total = dsp.shape[1])
            for (ix, selection, view) in dsp.scan(axis=1, batch_size=self.config.params.batch_size):
                dsp['CPM'][:,selection] = div0(view[''][:,:], 1e-6 * dsp.ca['GA_colsum'][selection])
                if self.poisson_pooling:
                    dsp['pooled_CPM'][:,selection] = div0(view['pooled'][:,:], 1e-6 * dsp.ca['GA_pooled_colsum'][selection])
                progress.update(self.config.params.batch_size)
            progress.close()
            logging.info(f'Finished calculating CPM values')

        return self.loom