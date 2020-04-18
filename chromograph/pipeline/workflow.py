import loompy
import os
import sys
import numpy as np
from datetime import datetime
import logging
sys.path.append('/home/camiel/chromograph/')
import chromograph
from chromograph.pipeline.Bin_analysis import *
from chromograph.pipeline import config
from chromograph.pipeline.peak_analysis import Peak_analysis
from chromograph.features.gene_smooth import GeneSmooth
from chromograph.peak_calling.peak_caller import *
from chromograph.peak_calling.utils import *
from chromograph.peak_calling.call_MACS import call_MACS
from chromograph.plotting.peak_annotation_plot import *
from chromograph.motifs.motif_compounder import motif_compounder

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

config = config.load_config()

if sys.argv[1] == 'Cerebellum':
    samples = ['232_1', '232_2', '250_1', '250_2']
    tissue = 'Cerebellum'
elif sys.argv[1] == 'Midbrain':
    samples = ['232_3', '232_4', '242_3', '242_4']
    tissue = 'Midbrain'
elif sys.argv[1] == 'Hindbrain':
    samples = ['242_1', '242_2', '250_3', '250_4']
    tissue = 'Hindbrain'

bsize = '5kb'

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

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
        '''
        Generate fragments piles based on cluster identities and use MACS to call peaks
        
        Args:

        Returns:
        
        Remarks:
        
        '''
        ## Check if location for peaks and compounded fragments exists
        if not os.path.isdir(self.peakdir):
            os.mkdir(self.peakdir)   

        if not os.path.exists(os.path.join(self.peakdir, 'Compounded_peaks.bed')):
            logging.info('Calling peaks')
            logging.info(f'Saving peaks to folder {self.peakdir}')

            chunks = []
            for i in np.unique(ds.ca['Clusters']):
                cells = [x.split(':') for x in ds.ca['CellID'][ds.ca['Clusters'] == i]]
                files = [os.path.join(self.config.paths.samples, x[0], 'fragments', f'{x[1]}.tsv.gz') for x in cells]
                if len(cells) > self.config.params.peak_min_cells:
                    chunks.append([i,files])

            logging.info('Start merging fragments by cluster')
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

            logging.info(f'Downsample pile-ups to {self.config.params.peak_depth / 1e6} million fragments')
            pool = mp.Pool(10) 
            for pile in piles:
                pool.apply_async(bed_downsample, args=(pile, self.config.params.peak_depth,))
            pool.close()
            pool.join()

            logging.info(f'Finished downsampling')

            logging.info(f'Start calling peaks')
            pool = mp.Pool(10) 
            for pile in piles:
                pool.apply_async(call_MACS, args=(pile, self.peakdir, self.config.paths.MACS,))
            pool.close()
            pool.join()
                
            ## Compound the peak lists
            peaks = [BedTool(x) for x in glob.glob(os.path.join(self.peakdir, '*.narrowPeak'))]
            logging.info('Identified on average {} peaks per cluster'.format(np.int(np.mean([len(x) for x in peaks]))))
            peaks_all = peaks[0].cat(*peaks[1:])

            f = os.path.join(self.peakdir, 'Compounded_peaks.bed')
            peaks_all.merge()
            peaks_all = peaks_all.each(extend_fields, 6).each(add_ID).each(add_strand, '+').saveas(f)   ## Pad out the BED-file and save
            logging.info(f'Identified {peaks_all.count()} peaks after compounding list')

            ## Clean up
            for file in glob.glob(os.path.join(self.peakdir, '*.tsv.gz')):
                os.system(f'rm {file}')

        else:
            logging.info('Compounded peak file already present, loading now')
            f = os.path.join(self.peakdir, 'Compounded_peaks.bed')
            peaks_all = BedTool(f)
        
        f_annot = os.path.join(self.peakdir, 'annotated_peaks.txt')
        if not os.path.exists(f_annot):
            ## Annotate peaks
            logging.info(f'Annotating peaks')
            homer = os.path.join(self.config.paths.HOMER, 'annotatePeaks.pl')
            genes = os.path.join(self.config.paths.ref, 'genes', 'genes.gtf')
            motifs = os.path.join(self.config.paths.ref, 'regions', 'motifs.pfm')
            cmd = f'{homer} {f} hg38 -gtf {genes} -m {motifs} > {f_annot}'  ## Command to call HOMER
            os.system(cmd)  ## Actually call HOMER

        ## Load and reorder HOMER output
        logging.info(f'Reordering annotation file')
        cols, table, TF_cols, TFs = read_HOMER_annotation(f_annot)
        peak_IDs = np.array([x[3] for x in peaks_all])
        table = reorder_by_IDs(table, peak_IDs)
        annot = {cols[i]: table[:,i] for i in range(table.shape[1])}
        logging.info('Plotting peak annotation wheel')
        plot_peak_annotation_wheel(annot, os.path.join(self.config.paths.build, 'exported', 'peak_annotation_wheel.png'))

        # Count peaks and make Peak by Cell matrix
        # Counting peaks
        logging.info(f'Start counting peaks')
        chunks = np.array_split(ds.ca['CellID'], 10)
        jobs = []
        q = mp.Queue()
        dicts = []
        for cells in chunks:
            p = mp.Process(target=Count_peaks, args=(cells, self.config.paths.samples, f, q, ))
            jobs.append(p)
            p.start()
            
        for i in range(len(chunks)):
            dicts.append(q.get())
            
        for p in jobs:
            p.join()

        dicts = []
        def log_result(result):
            # This is called whenever pool returns a result.
            dicts.append(result)

        pool = mp.Pool(10)
        chunks = np.array_split(ds.ca['CellID'], 10)
        for cells in chunks:
            pool.apply_async(Count_peaks2, args=(cells, self.config.paths.samples, f, ), callback = log_result)
        pool.close()
        pool.join()
        
        ## Unpack results
        Counts = {k: v for d in dicts for k, v in d.items()}
        r_dict = {k: v for v,k in enumerate(annot['ID'])} # Order dict for rows

        logging.info("Generating Sparse matrix")
        col = []
        row = []
        v = []

        cix = 0
        for cell in ds.ca['CellID']:
            if len(Counts[cell]) > 0:
                for key in (Counts[cell]):
                    col.append(cix)
                    row.append(r_dict[key])
                    v.append(np.int8(Counts[cell][key]))
                cix+=1
        matrix = sparse.coo_matrix((v, (row,col)), shape=(len(r_dict.keys()), len(ds.ca['CellID'])))
        logging.info(f'Matrix has shape {matrix.shape} with {matrix.nnz} elements')

        ## Create loomfile
        logging.info("Constructing loomfile")
        self.loom = f'{ds.filename.split(".")[0]}_peaks.loom'

        loompy.create(filename=self.loom, 
                    layers=matrix, 
                    row_attrs=annot, 
                    col_attrs=dict(ds.ca),
                    file_attrs=dict(ds.attrs))
        logging.info(f'Loom peaks file saved as {self.loom}')

        return self.loom

if __name__ == '__main__':

    logging.info(f'Performing the following steps: {config.steps}')
    logging.info(f'The build folder is {config.paths.build}, now analyzing {tissue}')
    outfile = os.path.join(config.paths.build, tissue + '.loom')

    if 'bin_analysis' in config.steps:
        ## Check if directory exists
        if not os.path.isdir(config.paths.build):
            os.mkdir(config.paths.build)

        # ## Merge Bin files
        inputfiles = [os.path.join(config.paths.samples, '10X' + sample, '10X' + sample + f"_{bsize}.loom") for sample in samples]
        if not os.path.exists(outfile):
            logging.info(f'Input files {inputfiles}')
            loompy.combine_faster(inputfiles, outfile, key = 'loc')
            # loompy.combine(inputfiles, outfile, key = 'loc')       ## Use if running into memory errors
            logging.info('Finished combining loom-files')
        else:
            logging.info('Combined bin file already exists, usingt his for analysis')

        ## Run primary Clustering and embedding
        with loompy.connect(outfile) as ds:
            ds.attrs['tissue'] = tissue
            bin_analysis = bin_analysis()
            bin_analysis.fit(ds)
    
    if 'GA' in config.steps:
        ## Merge GA files
        GA_file = os.path.join(config.paths.build, tissue + '_GA.loom')
        inputfiles = [os.path.join(config.paths.samples, '10X' + sample, f'10X{sample}_GA.loom') for sample in samples]
        for x in inputfiles:
            with loompy.connect(x) as ds:
                logging.info(f"{x} has shape{ds.shape}")
        loompy.combine_faster(inputfiles, GA_file, key = 'Accession')

        ## Transer column attributes
        with loompy.connect(GA_file) as ds:
            logging.info(f'Transferring column attributes and column graphs to GA file')
            with loompy.connect(outfile) as dsb:
                ds.permute(ds.ca['CellID'].argsort(), axis=1)
                dsb.permute(dsb.ca['CellID'].argsort(), axis=1)
                for x in dsb.ca:
                    if x not in ds.ca:
                        ds.ca[x] = dsb.ca[x]
                ## Transfer column graphs
                for x in dsb.col_graphs:
                    if x not in ds.col_graphs:
                        ds.col_graphs[x] = dsb.col_graphs[x]
            logging.info(f"GA file has shape{ds.shape}")
            Smooth = GeneSmooth()
            Smooth.fit(ds)

    if 'peak_calling' in config.steps:
        ## Call peaks
        with loompy.connect(outfile, 'r') as ds:
            peak_caller = Peak_caller()
            peak_caller.fit(ds)

    if 'peak_analysis' in config.steps:
        ## Analyse peak-file
        peak_file = os.path.join(config.paths.build, tissue + '_peaks.loom')
        with loompy.connect(peak_file, 'r+') as ds:
            Peak_analysis = Peak_analysis()
            Peak_analysis.fit(ds)

    if 'motifs' in config.steps:
        if 'peak_file' not in locals():
            peak_file = os.path.join(config.paths.build, tissue + '_peaks.loom')

        with loompy.connect(peak_file, 'r') as ds:
            motif_compounder = motif_compounder()
            motif_compounder.fit(ds)
