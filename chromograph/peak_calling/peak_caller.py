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
import shutil

import sklearn.metrics
from scipy.spatial import distance
from scipy import sparse
import community
import networkx as nx
from scipy import sparse
from typing import *
import multiprocessing as mp

sys.path.append('/home/camiel/chromograph/')
from chromograph.peak_calling.utils import *
from  chromograph.peak_calling.call_MACS import call_MACS
from chromograph.pipeline import config

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

# ## Parameters
# ds = loompy.connect('/data/proj/scATAC/chromograph/build_20191205/Midbrain.loom', 'r')
# logging.info(f'ds connection: {ds.shape}')

class Peak_caller:
    def __init__(self) -> None:
        """
        Generate fragments piles based on cluster identities and use MACS to call peaks
        
        Args:
            ds                    Loom connection

        Remarks:
        
        """
        self.config = config.load_config()
        self.peakdir = os.path.join(self.config.paths.build, 'peaks')
        self.loom = ''
        logging.info("Peak Caller initialised")
    
    def fit(self, ds: loompy.LoomConnection) -> None:
        ''''
        Generate fragments piles based on cluster identities and use MACS to call peaks
        
        Args:

        Returns:
        
        Remarks:
        
        '''
        # if __name__ == '__main__':
        ## Check if location for peaks and compounded fragments exists
        if not os.path.isdir(self.peakdir):
            os.mkdir(self.peakdir)   

        logging.info(f'Saving peaks to folder {self.peakdir}')

        chunks = []
        for i in np.unique(ds.ca['Clusters']):
            cells = ds.ca["SampleID", "barcode"][ds.ca['Clusters'] == i]
            files = [os.path.join(self.config.paths.samples, x[0], 'fragments', f'{x[1]}.tsv.gz') for x in cells]
            chunks.append([i,files])

        logging.info('Start merging fragments by cluster')
        # mp.set_start_method('spawn')
        piles = []
        for ck in chunks:
            files = np.array(ck[1])
            ex = np.array([os.path.exists(x) for x in files])
            files = files[ex]

            fmerge = os.path.join(self.peakdir, f'fragments_{ck[0]}.tsv.gz')
            with open(fmerge, 'wb') as out:
                for f in files:
                    with open(f, 'rb') as file:
                        shutil.copyfileobj(file, out)
            piles.append([ck[0], fmerge])
            logging.info(f'Finished with cluster {ck[0]}')

        jobs = []
        for pile in piles:
            p = mp.Process(target=call_MACS, args=(pile, self.peakdir, self.config.paths.MACS,))
            jobs.append(p)
            p.start()
            
        ## Compound the peak lists
        peaks = [BedTool(x) for x in glob.glob(os.path.join(self.peakdir, '*.narrowPeak'))]
        logging.info('Identified on average {} peaks per cluster'.format(np.int(np.mean([len(x) for x in peaks]))))
        peaks_all = peaks[0].cat(*peaks[1:])

        f = os.path.join(self.peakdir, 'Compounded_peaks.bed')
        peaks_all = peaks_all.each(extend_fields, 6).each(add_ID).each(add_strand, '+').saveas(f)   ## Pad out the BED-file and save
        logging.info(f'Identified {peaks_all.count()} peaks after compounding list')

        ## Annotate peaks
        homer = os.path.join(self.config.paths.HOMER, 'annotatePeaks.pl')
        genes = os.path.join(self.config.paths.ref, 'genes', 'genes.gtf')
        motifs = os.path.join(self.config.paths.ref, 'regions', 'motifs.pfm')
        f_annot = os.path.join(self.peakdir, 'annotated_peaks.txt')
        cmd = f'{homer} {f} hg38 -gtf {genes} -m {motifs} > {f_annot}'  ## Command to call HOMER
        os.system(cmd)  ## Actually call HOMER

        ## Load and reorder HOMER output
        table = read_HOMER_annotation(f_annot)
        peak_IDs = np.array([x[3] for x in peaks_all])
        table = reorder_by_IDs(table, peak_IDs)
        annot = {cols[i]: table[:,i] for i in range(table.shape[1])}

        ## Count peaks and make Peak by Cell matrix
        chunks = np.array_split(ds.ca['CellID'], 10)
        jobs = []
        q = mp.Queue()
        dicts = []
        for cells in chunks:
            p = mp.Process(target=Count_peaks, args=(cells, self.config.paths.samples, f, q, ))
            jobs.append(p)
            p.start()
            dicts.append(q.get())
        
        ## Unpack results
        Counts = {k: v for d in dicts for k, v in d.items()}
        r_dict = {k: v for v,k in enumerate(annot['ID'])} # Order dict for rows

        logging.info("Generating Sparse matrix")
        col = []
        row = []
        v = []

        cix = 0
        for cell in ds.ca['CellID'][:30]:

            for key in (Counts[cell]):
                col.append(cix)
                row.append(r_dict[key])
                v.append(Counts[cell][key])
            cix+=1
        matrix = sparse.coo_matrix((v, (row,col)), shape=(len(r_dict.keys()), len(ds.ca['CellID'])))

        ## Create loomfile
        logging.info("Constructing loomfile")
        self.loom = f'{ds.filename.split(".")[0]}_peaks.loom'

        loompy.create(filename=self.loom, 
                    layers=matrix, 
                    row_attrs=annot, 
                    col_attrs=dict(ds.ca),
                    file_attrs=dict(ds.attrs))
        logging.info("Loom peaks file saved as {}".format(floom))

        return self.loom
            
# peak_caller = Peak_caller()
# r = peak_caller.fit(ds)
# logging.info('Saved peaks as {r}')

