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

import gzip
import glob
import pybedtools
from pybedtools import BedTool
import MACS2
import shutil
import multiprocessing as mp
from joblib import parallel_backend
from joblib import Parallel, delayed
import scipy.sparse as sparse

## Import chromograph
import chromograph
from chromograph.pipeline.utils import transfer_ca
from chromograph.preprocessing.utils import get_blacklist
from chromograph.peak_calling.peak_caller import *
from chromograph.peak_calling.utils import *
from chromograph.peak_calling.call_MACS import call_MACS
from chromograph.plotting.peak_annotation_plot import *
from chromograph.motifs.motif_compounder import Motif_compounder

## Setup logger and load config
logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

class Peak_caller:
    def __init__(self, outdir) -> None:
        """
        Generate fragments piles based on cluster identities and use MACS to call peaks
        
        Args:
            ds                    Loom connection

        Remarks:
        
        """
        self.config = chromograph.pipeline.config.load_config()
        self.outdir = outdir
        self.peakdir = os.path.join(outdir, 'peaks')
        self.loom = ''
        self.precomp = 'None'
        logging.info("Peak Caller initialised")
    
    def fit(self, ds: loompy.LoomConnection) -> None:
        '''
        Generate fragments piles based on cluster identities and use MACS to call peaks
        
        Args:

        Returns:
        
        Remarks:
        
        '''
        ## Get sample name from loom-file
        name = ds.filename.split(".")[0]

        ## Check if location for peaks and compounded fragments exists
        if not os.path.isdir(self.peakdir):
            os.mkdir(self.peakdir)   

        ## Check if Compounded peaks already exists
        if not os.path.exists(os.path.join(self.peakdir, 'Compounded_peaks.bed')):
            path_precomp = os.path.join(self.config.paths.build, 'All', 'peaks', 'Compounded_peaks.bed')
            if os.path.exists(path_precomp):
                logging.info(f'Use peaks computed for full dataset')
                self.precomp = 'All'
                shutil.copyfile(path_precomp, os.path.join(self.peakdir, 'Compounded_peaks.bed'))
                f = os.path.join(self.peakdir, 'Compounded_peaks.bed')
            
                path_pre_annot = os.path.join(self.config.paths.build, 'All', 'peaks', 'annotated_peaks.txt')
                if os.path.exists(path_pre_annot):
                    logging.info(f'Use annotation of precomputed peaks')
                    shutil.copyfile(path_pre_annot, os.path.join(self.peakdir, 'annotated_peaks.txt'))
                    shutil.copyfile(os.path.join(self.config.paths.build, 'All', 'peaks', 'motif_annotation.txt'), os.path.join(self.peakdir, 'motif_annotation.txt'))
            
            else:
                logging.info('No precomputed peak list. Calling peaks')
                logging.info(f'Saving peaks to folder {self.peakdir}')

                chunks = []
                for i in np.unique(ds.ca['preClusters']):
                    cells = [x.split(':') for x in ds.ca['CellID'][ds.ca['preClusters'] == i]]
                    files = np.array([os.path.join(self.config.paths.samples, x[0], 'fragments', f'{x[1]}.tsv.gz') for x in cells])
                    X = np.random.choice(len(files), size=int(len(files)/2), replace=False)
                    r1 = np.zeros(len(files)).astype(bool)
                    r1[X] = True

                    if len(cells) > self.config.params.peak_min_cells:
                        chunks.append([f'{i}A', files[r1]])
                        chunks.append([f'{i}B', files[~r1]])

                logging.info('Start generating pseudobulk samples by precluster')
                logging.info(f'Total chunks: {len(chunks)}')
                with mp.get_context().Pool() as pool:
                    for ck in chunks:
                        pool.apply_async(merge_fragments, (ck, self.peakdir,))
                    pool.close()
                    pool.join()

                logging.info(f'Downsample large pile-ups to {self.config.params.peak_depth / 1e6} million fragments')
                piles = [[ck[0], os.path.join(self.peakdir, f'cluster_{ck[0]}.tsv.gz')] for ck in chunks]
                with mp.get_context().Pool(min(mp.cpu_count(), len(piles)), maxtasksperchild=1) as pool:
                    for pile in piles:
                        pool.apply_async(bed_downsample, args=(pile, self.config.params.peak_depth,))
                    pool.close()
                    pool.join()

                ## Manually call garbage collection to prevent memory leaks from mp
                gc.collect()

                logging.info(f'Start calling peaks')
                with mp.get_context().Pool(min(mp.cpu_count(), len(piles)), maxtasksperchild=1) as pool:
                    for pile in piles:
                        pool.apply_async(call_MACS, args=(pile, self.peakdir, self.config.paths.MACS,))
                    pool.close()
                    pool.join()

                ## Manually call garbage collection to prevent memory leaks from mp
                gc.collect()
                    
                ## Unify pseudobulk peaks
                peaks = []
                for i in np.unique(ds.ca['preClusters']):
                    files = glob.glob(os.path.join(self.peakdir, f'cluster_{i}*.narrowPeak'))
                    bd = [BedTool(f) for f in files]
                    peaks.append(bd[0].intersect(bd[1], wo=True).cluster().merge())
                    dp = len(peaks[-1])
                    sp = np.sum([len(x) for x in bd]) - (2*dp)

                ## Compound the peak lists
                logging.info(f'Identified on average {np.int(np.mean([len(x) for x in peaks]))} peaks per cluster')
                peaks_all = peaks[0].cat(*peaks[1:])
                peaks_all.cluster().merge()

                ## Substract blacklist
                black_list = BedTool(get_blacklist(self.config.params.reference_assembly))
                peaks_all.subtract(black_list, A=True)

                ## Pad out and save peaks
                f = os.path.join(self.peakdir, 'Compounded_peaks.bed')
                peaks_all = peaks_all.each(extend_fields, 6).each(add_ID).each(add_strand, '+').saveas(f)   ## Pad out the BED-file and save
                logging.info(f'Identified {peaks_all.count()} peaks after compounding list')

                ## Clean up
                for file in glob.glob(os.path.join(self.peakdir, '*.tsv.gz')):
                    os.system(f'rm {file}')
                for file in glob.glob(os.path.join(self.peakdir, '*.narrowPeak')):
                    os.system(f'rm {file}')
        else:
            logging.info('Compounded peak file already present, loading now')
            f = os.path.join(self.peakdir, 'Compounded_peaks.bed')
            peaks_all = BedTool(f)

        ## Check All_peaks.loom exists, get subset
        all_peaks_loom = os.path.join(self.config.paths.build, 'All', 'All_peaks.loom')
        if os.path.exists(all_peaks_loom):
            logging.info(f'Main peak matrix already exists')
            with loompy.connect(all_peaks_loom) as dsp:
                selection = np.array([x in ds.ca.CellID for x in dsp.ca.CellID])
                
            self.loom = os.path.join(self.outdir, f'{name}_peaks.loom')
            loompy.combine_faster([all_peaks_loom], self.loom, selections=[selection])
            logging.info(f'Finished creating peak file')

            with loompy.connect(self.loom, 'r+') as ds_out:
                logging.info(f'Transferring attributes')
                transfer_ca(ds, ds_out, 'CellID')
                return
        
        f_annot = os.path.join(self.peakdir, 'annotated_peaks.txt')
        if not os.path.exists(f_annot):
            ## Annotate peaks
            logging.info(f'Annotating peaks')
            homer = os.path.join(self.config.paths.HOMER, 'annotatePeaks.pl')
            logging.info(f'Found Homer at {homer}')
            genes = os.path.join(self.config.paths.ref, 'genes', 'genes.gtf')
            motifs = os.path.join(chromograph.__path__[0], 'references/human_TFs.motifs') ## Read the motif file from chromograph reference
            cmd = f'{homer} {f} hg38 -gtf {genes} -CpG > {f_annot}'  ## Command to call HOMER
            os.system(cmd)  ## Actually call HOMER

            ## Annotate motifs
            split_dir = os.path.join(self.peakdir, 'compound_split')
            out_motifs = os.path.join(self.peakdir, 'out_motifs')
            [os.mkdir(x) for x in [split_dir, out_motifs] if not os.path.exists(x)]
            subprocess.run(['awk', '{print $0 >>' + f'"{split_dir}/"' + '$1".bed"}', os.path.join(self.peakdir, 'Compounded_peaks.bed')]) ## Split by chromosome since motif annotation takes a lot of RAM

            logging.info('Annotating motifs')
            with mp.get_context().Pool(10, maxtasksperchild=1) as pool:
                for file in tqdm(os.listdir(split_dir)):
                    chrom_file = os.path.join(split_dir, file)
                    out_file = os.path.join(out_motifs, f'{file.split("/")[-1].split(".")[0]}.txt')
                    pool.apply_async(homer_motif_call, args=(homer,chrom_file, motifs, out_file,))
                pool.close()
                pool.join()

            ## Merge motif outputs
            motif_outputs = [os.path.join(out_motifs, x) for x in os.listdir(out_motifs)]
            motif_annotation = open(os.path.join(self.peakdir, 'motif_annotation.txt'), 'a')
            subprocess.call(['head', '-1', motif_outputs[0]], stdout = motif_annotation)
            for x in motif_outputs:
                subprocess.call(['tail', '-n', '+2', '-q',x], stdout = motif_annotation)

        ## Load and reorder HOMER output
        logging.info(f'Reordering annotation file')
        cols, table = read_HOMER_annotation(f_annot)
        peak_IDs = np.array([x[3] for x in peaks_all])
        table = reorder_by_IDs(table, peak_IDs)
        annot = {cols[i]: table[:,i] for i in range(table.shape[1])}
        logging.info('Plotting peak annotation wheel')
        plot_peak_annotation_wheel(annot, os.path.join(self.outdir, 'exported', 'peak_annotation_wheel.png'))

        ## Clean up stranded pybedtools tmp files
        pybedtools.helpers.cleanup(verbose=True, remove_all=True)

        logging.info(f'Start counting peaks')
        chunks = np.array_split(ds.ca['CellID'], np.int(np.ceil(ds.shape[1]/500)))
        logging.info(f'Total of {len(chunks)} chunks')

        if len(chunks) > len(glob.glob(os.path.join(self.peakdir, '*.loom'))):
            with mp.get_context().Pool(min(mp.cpu_count(),len(chunks)), maxtasksperchild=1) as pool:
                for i, cells in enumerate(chunks):
                    pool.apply_async(generate_peak_matrix, args=(i, cells, self.config.paths.samples, self.peakdir, annot, False, ))
                pool.close()
                pool.join()

        logging.info(f'Combining files')
        self.loom = os.path.join(self.outdir, f'{name}_peaks.loom')
        inputfiles = [os.path.join(self.peakdir, f) for f in sorted(glob.glob(os.path.join(self.peakdir, '*.loom')))]
        loompy.combine_faster(inputfiles, self.loom, key = 'ID')
        logging.info(f'Transferring column attributes')
        with loompy.connect(self.loom) as ds2:
            ds2.attrs['peak_file'] = self.precomp
            transfer_ca(ds, ds2, 'CellID')
        logging.info(f'Loom peaks file saved as {self.loom}')

        ## Cleanup
        for file in glob.glob(os.path.join(self.peakdir, '*.loom')):
            os.system(f'rm {file}')

        ## Clean up stranded pybedtools tmp files
        pybedtools.helpers.cleanup(verbose=True, remove_all=True)
        return self.loom
