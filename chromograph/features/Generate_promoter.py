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
sys.path.append('/home/camiel/chromograph/')
import chromograph
from chromograph.pipeline.Bin_analysis import *
from chromograph.pipeline import config
from chromograph.pipeline.utils import transfer_ca, div0
from chromograph.features.GA_Aggregator import GA_Aggregator
from chromograph.peak_calling.peak_caller import *
from chromograph.peak_calling.utils import *

## Setup logger and load config
logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

class Generate_promoter:
    def __init__(self, outdir) -> None:
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
        self.loom = ''
    
    def fit(self, ds: loompy.LoomConnection) -> None:
        '''
        Use Gene reference to generate Gene accessibility scores as a gene expression proxy
        
        Args:

        Returns:
        
        Remarks:
        
        '''
        ## Get sample name from loom-file
        name = ds.filename.split(".")[0]
        self.loom = os.path.join(self.outdir, f'{name}_prom.loom')

        ## Check if location for peaks and compounded fragments exists
        if not os.path.isdir(self.peakdir):
            os.mkdir(self.peakdir)   

        ## Check All_peaks.loom exists, get subset
        all_prom_loom = os.path.join(self.config.paths.build, 'All', 'All_prom.loom')

        if os.path.exists(all_prom_loom) & (all_prom_loom != self.loom):
            logging.info(f'Main promoter matrix already exists')
            
            with loompy.connect(all_prom_loom) as dsp:
                selection = np.array([x in ds.ca.CellID for x in dsp.ca.CellID])
            
            loompy.combine_faster([all_prom_loom], self.loom, selections=[selection], key = 'CellID')
            
            with loompy.connect(self.loom) as ds2:
                transfer_ca(ds, ds2, 'CellID')
            logging.info(f'Finished creating promoter file')
        
        else:
            logging.info(f'Start counting peaks')
            pool = mp.Pool(20, maxtasksperchild=1)
            chunks = np.array_split(ds.ca['CellID'], np.int(np.ceil(ds.shape[1]/1000)))
            for i, cells in enumerate(chunks):
                pool.apply_async(Count_peaks, args=(i, cells, self.config.paths.samples, self.peakdir, self.gene_ref, 'genes',))
            pool.close()
            pool.join()

            ## Generate row attributes
            row_attrs = {k: [] for k in ['Accession', 'Gene', 'loc', 'BPs']}
            for x in BedTool(self.gene_ref):
                row_attrs['Accession'].append(x.attrs['gene_id'])
                row_attrs['Gene'].append(x.attrs['gene_name'])
                row_attrs['loc'].append(f'{x[0]}:{x[3]}-{x[4]}')
                row_attrs['BPs'].append(int(abs(int(x[3])-int(x[4]))))

            r_dict = {k: v for v,k in enumerate(row_attrs['Accession'])} 

            logging.info("Generating Sparse matrix")
            col = []
            row = []
            v = []

            cix = 0
            IDs = []
            dict_files = glob.glob(os.path.join(self.peakdir, '*.pkl'))
            for file in dict_files:
                Counts = pkl.load(open(file, 'rb'))
                for cell in Counts:
                    if len(Counts[cell]) > 0:
                        for key in (Counts[cell]):
                            col.append(cix)
                            row.append(r_dict[key])
                            v.append(np.int8(Counts[cell][key]))
                        cix+=1
                        IDs.append(cell)
            logging.info(f'CellID order is maintained: {np.array_equal(ds.ca.CellID, np.array(IDs))}')
            matrix = sparse.coo_matrix((v, (row,col)), shape=(len(r_dict.keys()), len(ds.ca['CellID']))).tocsc()
            logging.info(f'Matrix has shape {matrix.shape} with {matrix.nnz} elements')

            ## Create loomfile
            logging.info("Constructing loomfile")
            loompy.create(filename=self.loom, 
                        layers=matrix, 
                        row_attrs=row_attrs, 
                        col_attrs={'CellID': np.array(IDs)},
                        file_attrs=dict(ds.attrs))
            logging.info(f'Transferring column attributes')
            with loompy.connect(self.loom) as ds2:
                transfer_ca(ds, ds2, 'CellID')
            logging.info(f'Loom promoter file saved as {self.loom}')

            for file in glob.glob(os.path.join(self.peakdir, '*.pkl')):
                os.system(f'rm {file}')

            ## Clean up stranded pybedtools tmp files
            pybedtools.helpers.cleanup(verbose=True, remove_all=True)
    
        ## Generate pooled layer
        with loompy.connect(self.loom) as ds:
            ## Poisson pooling
            logging.info(f"Poisson pooling")
            if 'HPF' in ds.ca:
                decomp = ds.ca.HPF
            elif 'LSI' in ds.ca:
                decomp = ds.ca.LSI
            logging.info(f'Create NN graph')
            nn = NNDescent(data=decomp, metric="euclidean", n_neighbors=self.config.params.k_pooling)
            logging.info(f'Query NN graph')
            indices, distances = nn._neighbor_graph
            # Note: we convert distances to similarities here, to support Poisson smoothing below
            knn = sparse.csr_matrix(
                (np.ravel(distances), np.ravel(indices), np.arange(0, distances.shape[0] * distances.shape[1] + 1, distances.shape[1])), (decomp.shape[0], decomp.shape[0]))
            max_d = knn.data.max()
            knn.data = (max_d - knn.data) / max_d
            knn.setdiag(1)  # This causes a sparse efficiency warning, but it's not a slow step relative to everything else
            knn = knn.astype("bool")

            ## Start pooling over the network
            logging.info(f'Start pooling over network')
            ds["pooled"] = 'int32'
            for (_, indexes, view) in ds.scan(axis=0, layers=[""], what=["layers"]):
                ds["pooled"][indexes.min(): indexes.max() + 1, :] = view[:, :] @ knn.T

            logging.info(f'Converting to CPM')  # divide by GA_colsum/1e6
            ds.ca['GA_colsum'] = ds[''].map([np.sum], axis=1)[0]
            ds.ca['GA_pooled_colsum'] = ds['pooled'].map([np.sum], axis=1)[0]
            
            ds['CPM'] = 'float32'
            ds['pooled_CPM'] = 'float32'
            progress = tqdm(total = ds.shape[1])
            for (ix, selection, view) in ds.scan(axis=1, batch_size=self.config.params.batch_size):
                ds['CPM'][:,selection] = div0(view[''][:,:], 1e-6 * ds.ca['GA_colsum'][selection])
                ds['pooled_CPM'][:,selection] = div0(view['pooled'][:,:], 1e-6 * ds.ca['GA_pooled_colsum'][selection])
                progress.update(self.config.params.batch_size)
            progress.close()
            
        return self.loom