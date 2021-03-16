import loompy
import os
import gc
import sys
import numpy as np
from datetime import datetime
import logging
from typing import *

import pickle as pkl
import glob
import pybedtools
from pybedtools import BedTool
import shutil
import multiprocessing as mp

## Import chromograph
import chromograph
from chromograph.pipeline import config
from chromograph.peak_calling.utils import *

config = config.load_config()

## Setup logger and load config
logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

def generate_prom_matrix(file_5kb, gene_ref, peakdir, sample_dir, verbose=False):
    name = file_5kb.split('/')[-1].split('.')[0]
    # loom_file = f"/{os.path.join(*file_5kb.split('/')[:-1])}/{name}_prom.loom"
    loom_file = f"/{os.path.join(*file_5kb.split('/')[:-1], '_'.join(file_5kb.split('/')[-1].split('_')[:2]))}_prom.loom"

    if verbose:
        logging.info(f"Start counting TSS enrichment for {name}")
    
    with loompy.connect(file_5kb, 'r') as ds:
        chunks = np.array_split(ds.ca['CellID'], np.int(np.ceil(ds.shape[1]/100)))
        dict_files = glob.glob(os.path.join(peakdir, '*.pkl'))

        if len(chunks) > len(dict_files):
            if verbose:
                logging.info(f'Using {min(mp.cpu_count(), len(chunks))} cores')
            with mp.get_context().Pool(min(mp.cpu_count(), len(chunks)), maxtasksperchild=10) as pool:
                for i, cells in enumerate(chunks):
                    pool.apply_async(Count_peaks, args=(i, cells, sample_dir, peakdir, gene_ref, 'genes',))
                pool.close()
                pool.join()

        ## Generate row attributes
        if verbose:
            logging.info("Generating row attributes")
        row_attrs = {k: [] for k in ['Accession', 'Gene', 'loc', 'BPs', 'Chr', 'Start', 'End']}
        for x in BedTool(gene_ref):
            row_attrs['Accession'].append(x.attrs['gene_id'])
            row_attrs['Gene'].append(x.attrs['gene_name'])
            row_attrs['loc'].append(f'{x[0]}:{x[3]}-{x[4]}')
            row_attrs['BPs'].append(int(abs(int(x[3])-int(x[4]))))
            row_attrs['Chr'].append(f'{x[0]}')
            row_attrs['Start'].append(f'{x[3]}')
            row_attrs['End'].append(f'{x[4]}')

        r_dict = {k: v for v,k in enumerate(row_attrs['Accession'])} 

        if verbose:
            logging.info("Generating Sparse matrix")
        col = []
        row = []
        v = []

        cix = 0
        IDs = []
        dict_files = sorted(glob.glob(os.path.join(peakdir, '*.pkl')))
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

        matrix = sparse.coo_matrix((v, (row,col)), shape=(len(r_dict.keys()), len(ds.ca['CellID']))).tocsc()        
        if verbose:
            logging.info(f'Matrix has shape {matrix.shape} with {matrix.nnz} elements')

        ## Create loomfile
        if verbose:
            logging.info("Constructing loomfile")
        
        loompy.create(filename=loom_file, 
                    layers=matrix, 
                    row_attrs=row_attrs, 
                    col_attrs={'CellID': np.array(IDs)},
                    file_attrs=dict(ds.attrs))
        
        ## Remove pkls
        if verbose:
            logging.info("Deleting pkl files")
        for f in dict_files:
            os.system(f'rm {f}') 
        if verbose:
            logging.info(f"Finished with {name}")