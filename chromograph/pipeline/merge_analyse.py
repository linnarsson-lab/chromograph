import loompy
import os
import sys
import numpy as np
from datetime import datetime
import logging
sys.path.append('/home/camiel/chromograph/')
from chromograph.pipeline.Bin_analysis import *
from chromograph.peak_calling.peak_caller import *
from chromograph.features.gene_smooth import GeneSmooth
from chromograph.pipeline import config

config = config.load_config()

if sys.argv[1] == 'Cerebellum':
    samples = ['232_1', '232_2']
    tissue = 'Cerebellum'
elif sys.argv[1] == 'Midbrain':
    samples = ['232_3', '232_4']
    tissue = 'Midbrain'

bsize = '5kb'

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

import gzip
import glob
import pybedtools
from pybedtools import BedTool
import MACS2
import shutil
import community
import networkx as nx
from scipy import sparse
from typing import *
import multiprocessing as mp

sys.path.append('/home/camiel/chromograph/')
import chromograph
from chromograph.peak_calling.utils import *
from  chromograph.peak_calling.call_MACS import call_MACS

class Peak_caller:
    def __init__(self) -> None:
        """
        Generate fragments piles based on cluster identities and use MACS to call peaks
        
        Args:
            ds                    Loom connection

        Remarks:
        
        """
        self.config = chromograph.pipeline.config.load_config()
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

        for p in jobs:
            p.join()
            
        ## Compound the peak lists
        peaks = [BedTool(x) for x in glob.glob(os.path.join(self.peakdir, '*.narrowPeak'))]
        logging.info('Identified on average {} peaks per cluster'.format(np.int(np.mean([len(x) for x in peaks]))))
        peaks_all = peaks[0].cat(*peaks[1:])

        f = os.path.join(self.peakdir, 'Compounded_peaks.bed')
        peaks_all = peaks_all.each(extend_fields, 6).each(add_ID).each(add_strand, '+').saveas(f)   ## Pad out the BED-file and save
        logging.info(f'Identified {peaks_all.count()} peaks after compounding list')

        ## Clean up
        for file in glob.glob(os.path.join(self.peakdir, '*.tsv.gz')):
            os.system(f'rm {file}')

        ## Annotate peaks
        logging.info(f'Annotating peaks')
        homer = os.path.join(self.config.paths.HOMER, 'annotatePeaks.pl')
        genes = os.path.join(self.config.paths.ref, 'genes', 'genes.gtf')
        motifs = os.path.join(self.config.paths.ref, 'regions', 'motifs.pfm')
        f_annot = os.path.join(self.peakdir, 'annotated_peaks.txt')
        cmd = f'{homer} {f} hg38 -gtf {genes} -m {motifs} > {f_annot}'  ## Command to call HOMER
        os.system(cmd)  ## Actually call HOMER

        ## Load and reorder HOMER output
        logging.info(f'Reordering annotation file')
        table = read_HOMER_annotation(f_annot)
        peak_IDs = np.array([x[3] for x in peaks_all])
        table = reorder_by_IDs(table, peak_IDs)
        annot = {cols[i]: table[:,i] for i in range(table.shape[1])}

        ## Count peaks and make Peak by Cell matrix
        ## Counting peaks
        chunks = np.array_split(ds.ca['CellID'], 10)
        jobs = []
        q = mp.Queue()
        dicts = []
        for cells in chunks:
            p = mp.Process(target=Count_peaks, args=(cells, self.config.paths.samples, f, q, ))
            jobs.append(p)
            p.start()
            dicts.append(q.get())
            
        for p in jobs:
            p.join()
        
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

if __name__ == '__main__':

    ## Check if directory exists
    if not os.path.isdir(config.paths.build):
        os.mkdir(config.paths.build)

    logging.info(f'The build folder is {config.paths.build}, now analyzing {tissue}')

    ## Merge Bin files
    outfile = os.path.join(config.paths.build, tissue + '.loom')
    inputfiles = [os.path.join(config.paths.samples, '10X' + sample, '10X' + sample + f"_{bsize}.loom") for sample in samples]

    logging.info(f'Input files {inputfiles}')

    # loompy.combine_faster(inputfiles, outfile, key = 'loc')
    # loompy.combine(inputfiles, outfile, key = 'loc')       ## Use if running into memory errors
    logging.info('Finished combining loom-files')

    # ## Run primary Clustering and embedding
    # with loompy.connect(outfile) as ds:
    #     ds.attrs['tissue'] = tissue
    #     bin_analysis = bin_analysis()
    #     bin_analysis.fit(ds)
        
    # ## Merge GA files
    # GA_file = os.path.join(config.paths.build, tissue + '_GA.loom')
    # inputfiles = [os.path.join(config.paths.samples, '10X' + sample, f'10X{sample}_GA.loom') for sample in samples]
    # for x in inputfiles:
    #     with loompy.connect(x) as ds:
    #         logging.info(f"{x} has shape{ds.shape}")
    # loompy.combine_faster(inputfiles, GA_file, key = 'Accession')
    # with loompy.connect(GA_file) as ds:
    #     logging.info(f'Transferring column attributes and column graphs to GA file')
    #     with loompy.connect(outfile) as dsb:
    #         for x in dsb.ca:
    #             if x not in ds.ca:
    #                 ds.ca[x] = dsb.ca[x]
    #         for x in dsb.col_graphs:
    #             if x not in ds.col_graphs:
    #                 ds.col_graphs[x] = dsb.col_graphs[x]
    #     logging.info(f"GA file has shape{ds.shape}")
    #     Smooth = GeneSmooth()
    #     Smooth.fit(ds)

    ## Call peaks
    with loompy.connect(outfile) as ds:
        peak_caller = Peak_caller()
        peak_caller.fit(ds)