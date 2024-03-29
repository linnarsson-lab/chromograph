# Description etc to be added

import numpy as np
import os
import yaml
import pybedtools
from pybedtools import BedTool
import collections
import csv
import pandas as pd
import matplotlib.pyplot as plt
import loompy
import scipy.sparse as sparse
import json
import logging
import pickle
import glob
import gc
import multiprocessing as mp

from chromograph.pipeline import config
from chromograph.plotting.sample_QC_plot import sample_QC_plots
from chromograph.preprocessing.utils import *
from chromograph.features.feature_count import *
from chromograph.preprocessing.doublet_finder import doublet_finder
from chromograph.RNA.utils import rna_barcodes_to_atac

class Chromgen:
    def __init__(self) -> None:
        """
        Generate a binned loom file from scATAC-seq data
        
        Args:
            steps                    Which steps to include in the analysis

        Remarks:
            --- PUNCHCARD SUPPORT NEEDS TO INTEGRATED, FOR NOW PASS PARAMETER TO FIT FUNCTION ---
            # All parameters are obtained from the config object, which comes from the default config
            # and can be overridden by the config in the current punchcard
        """
        self.config = config.load_config()
        self.rnaXatac = False
        self.RNA_file = ''
        pybedtools.helpers.set_bedtools_path(self.config.paths.bedtools)
        logging.info("Chromgen initialised")

    def fit(self, indir: str, bsize: int = 5000, outdir: str = None, genome_size: str = None, blacklist: str = None, min_fragments: bool = False, path_meta: str = None) -> None:
        ''''
        Create a .loom file from 10X Genomics cellranger output with reads binned
        Args:
            indir (str):	path to the cellranger output folder (the one that contains 'outs')
            bsize (int):	size of the bins (defaults to 5000/5kb)
            outdir (str):	output folder wher the new loom file should be saved (default to indir)
            genome_size (str):	path to file containing chromosome sizes, usually derived from encode (i.e. 'hg19.chrom.sizes.tsv')
            blacklist (str):	path to bedfile containing blacklisted region (i.e. 'blacklist_hg19.bed')
            path_meta (str):    If a defined meta data file (.yaml format) should be used instead of the default database define path here (None for sangerDB), 
                                at minimum use a file defining the sample name
        Returns:
            path (str):		Full path to the created loom file.
        Remarks:
            The resulting file will be named ``{sampleID}.loom``, where the sampleID is the one given by cellranger.
        '''
        logging.info(f"Binning reads from {indir.split('/')[-1]} into {bsize/1000} kb bins")
        logging.info(f"Reading from {indir}")
        logging.info(f"Saving to {outdir}")
        logging.info(f"Available CPUs: {mp.cpu_count()}")

        sample = indir.split('/')[-1]
        if len(sample.split('_')) > 2:
            sample = '_'.join(sample.split('_')[:2])

        ## Check if rnaXatac
        if os.path.isfile(indir + '/outs/per_barcode_metrics.csv'):
            self.rnaXatac = True
            self.RNA_file = os.path.join(self.config.paths.RNA, f'{sample}.loom')
            logging.info(f'Checking if {self.RNA_file} exists')
            if not os.path.exists(self.RNA_file):
                logging.info(f'RUN RNA QC FIRST!')
                return
        if self.rnaXatac:
            logging.info(f'Multiome sample')
            fb = indir + '/outs/per_barcode_metrics.csv'
            ff = indir + '/outs/atac_fragments.tsv.gz'
            fs = indir + '/outs/summary.csv'
        else:
            logging.info('scATAC-seq sample')
            fb = indir + '/outs/singlecell.csv'
            ff = indir + '/outs/fragments.tsv.gz'
            fs = indir + '/outs/summary.json'

        if not os.path.isdir(outdir):
            os.mkdir(outdir)
   
        logging.info(f'Reading metadata and summary for {sample} from Cellranger output {fs}')
        if self.rnaXatac:
            summary = np.genfromtxt(fs, dtype=str, delimiter=',')
            summary = {str(k): str(v) for k, v in zip(summary[0,:], summary[1,:])}
            summary['reference_assembly'] = summary['Genome']

            if summary['reference_assembly'] == "hg38-final3":
                summary['reference_assembly'] = 'GRCh38'
        else:
            with open(fs, "r") as f:
                summary = json.load(f)
                summary = {k: str(v) for k,v in summary.items()}

            if summary['reference_genomes'] in ["['GRCh38']", "hg38-final3"]:
                summary['reference_assembly'] = 'GRCh38'
        summary['bin_size'] = bsize
        summary['level'] = self.config.params.level
        barcodes = pd.read_csv(fb)
        
        if self.rnaXatac:    
            barcodes = barcodes.rename(columns = {'barcode': 'barcode',
                                                    'is_cell': 'is__cell_barcode',
                                                    'atac_raw_reads': 'total',
                                                    'atac_unmapped_reads': 'unmapped',
                                                    'atac_lowmapq': 'lowmapq',
                                                    'atac_dup_reads': 'duplicate',
                                                    'atac_chimeric_reads': 'chimeric',
                                                    'atac_mitochondrial_reads': 'mitochondrial',
                                                    'atac_fragments': 'passed_filters',
                                                    'atac_TSS_fragments': 'TSS_fragments',
                                                    'atac_peak_region_fragments': 'peak_region_fragments',
                                                    'atac_peak_region_cutsites': 'peak_region_cutsites'}, 
                                    inplace = False)
        ## Transfer metadata to dict format
        meta = {}
        # passed = (barcodes['is__cell_barcode'] == 1) & (barcodes['passed_filters'] > self.config.params.level) & (barcodes['passed_filters'] < 100000)
        if min_fragments:
            passed = (barcodes['is__cell_barcode'] == 1) & (barcodes['passed_filters'] > self.config.params.level)
            logging.info(f'Only preserving cells with more than {self.config.params.level} fragments')
        else:
            passed = (barcodes['is__cell_barcode'] == 1)
        for key in barcodes:
            meta[key] = np.array(barcodes[key][passed])

        meta['CellID'] = np.array([f'{sample}:{x}' for x in meta['barcode']])

        ## Remove trailing -1 if present
        if len(meta['CellID'][0].split('-')) > 1:
            meta['CellID'] = np.array([x.split('-')[0] for x in meta['CellID']])
    
        ## Retrieve sample metadata from SangerDB
        if not path_meta:
            logging.info(f'Retrieve metadata from {[self.config.paths.metadata, sample]}')
            m = load_sample_metadata(self.config.paths.metadata, sample)
        else:
            logging.info(f'Retrieve metadata from {path_meta}')
            with open(path_meta, 'r') as file:
                m = yaml.load(file, Loader=yaml.FullLoader)
        for k,v in m.items():
            meta[k] = np.array([v] * len(meta['barcode']))

        logging.info(f"Total of {len(meta['barcode'])} valid cells")
        logging.info(f"Ref. assembly {summary['reference_assembly']}")

        # Get Chromosome sizes
        if genome_size == None:
            chrom_size = get_chrom_sizes(summary['reference_assembly'])    
        else:
            chrom_size = {}
            with open(genome_size) as f:
                for line in f:
                    x = line.split()
                    chrom_size[x[0]] = int(x[1])

        ## Generate size bins
        logging.info(f"Generate {str(int(bsize/1000)) + ' kb'} bins based on provided chromosome sizes")
        chrom_bins = generate_bins(chrom_size, bsize)

        ## Split fragments by cell
        split_fragments2(ff, outdir, meta, chrom_size)

        chunks = np.array_split(meta['barcode'], np.int(np.ceil(len(meta['barcode'])/100)))
        chunks = [[i,ck] for i, ck in enumerate(chunks)]

        logging.info(f'Counting bins')
        with mp.get_context().Pool() as pool:
            for ck in chunks:
                pool.apply_async(Count_bins, (ck[0], ck[1], outdir, chrom_bins,))
            pool.close()
            pool.join()

        mats = [pkl.load(open(os.path.join(outdir, f'{id}.pkl'), 'rb')) for id, cells in chunks]
        matrix = sparse.hstack(mats)

        logging.info("Loading blacklist")
        # Load Blacklist
        if blacklist == None:
            blacklist = get_blacklist(summary['reference_assembly'])    

        logging.info("Remove bins that overlap with the ENCODE blacklist")
        black_list = BedTool(blacklist)
        bins = [(k[0], str(k[1]), str(k[2])) for k in chrom_bins.keys()]
        intervals = BedTool(bins)
        cleaned = intervals.subtract(black_list, A=True)

        keep = [(row['chrom'], int(row['start']), int(row['end'])) for row in cleaned.sort()] # Sort the bins to make downstream alignment with features easier
        retain = [chrom_bins[x] for x in keep]
        clean_bin = [bins[x] for x in retain]
        
        logging.info(f"Number of bins after cleaning: {len(clean_bin)}")

        #####
        # Construct loom file
        #####
        
        ## Save a smaller section of the summary
        if self.rnaXatac:
            keys = ['Pipeline version', 'reference_assembly', 'bin_size']
        else:
            keys = ['cellranger-atac_version', 'reference_assembly', 'reference_organism', 'reference_version', 'bin_size']
        small_summary = {k: summary[k] for k in keys}
        
        ## We retain only the bins that have no overlap with the ENCODE blacklist
        matrix = matrix.tocsc()[retain,:]
        logging.info(f"Identified {len(matrix.nonzero()[0])} positive bins in {len(meta['barcode'])} cells after filtering blacklist")

        ## Create row attributes
        chrom = [x[0] for x in clean_bin]
        start = [x[1] for x in clean_bin]
        end = [x[2] for x in clean_bin]

        row_attrs = {'loc': np.array([f'{c}:{s}-{e}' for c,s,e in zip(chrom, start, end)]), 
                     'chrom': np.array(chrom), 'start': np.array(start), 'end': np.array(end)}

        ## Create loomfile
        logging.info("Constructing loomfile")
        sampleid = sample + '_' + str(int(bsize/1000)) + 'kb'
        floom = os.path.join(outdir, sampleid + '.loom')

        gc.collect()  ## Clean up

        loompy.create(filename=floom, 
                      layers=matrix, 
                      row_attrs=row_attrs, 
                      col_attrs=meta,
                      file_attrs=small_summary)
        self.loom = floom
        logging.info(f"Loom bin file saved as {floom}")
        
        ## Add Y-chromosomal percentage as column attribute
        with loompy.connect(self.loom, 'r+') as ds:
            ds.ca.FRtss = ds.ca.TSS_fragments / ds.ca.passed_filters
            ds.ca.Y = np.sum(ds[np.where(ds.ra.chrom == 'chrY')[0],:], axis=0) / ds.ca.passed_filters
            SEX = 'F' if np.median(ds.ca.Y) < .0005 else 'M'
            ds.ca.SEX = np.repeat(SEX, ds.shape[1])

        ## Doublet detection
        if not self.rnaXatac:
            with loompy.connect(self.loom, 'r+') as ds:
                if ds.shape[1] > 1000:
                    logging.info(f'Detecting doublets in loom file')
                    ds.ca['DoubletFinderScore'], ds.ca['DoubletFinderFlag'] = doublet_finder(ds, proportion_artificial=.2, qc_dir = self.config.paths.qc, name = sample, max_th=self.config.params.max_doubletFinder_TH)
                else:
                    logging.info(f'Too few cells for doublet detections')
                    ds.ca['DoubletFinderScore'], ds.ca['DoubletFinderFlag'] = np.zeros((ds.shape[1],)).astype(np.float64), np.zeros((ds.shape[1],)).astype(np.int64)
        else:
            logging.info(f'Transferring doublet score from RNA file')
            with loompy.connect(self.loom) as ds:
                with loompy.connect(self.RNA_file) as dsr:
                    logging.info(f"Shape of RNA file: {dsr.shape}")

                    if not 'DoubletFinderScore' in dsr.ca:
                        logging.info(f'ERROR: NO DOUBLETFINDERSCORE IN RNA!')
                        return
                    RNA_bar = rna_barcodes_to_atac(dsr)
                    match = {k:v for v, k in enumerate(ds.ca.CellID)}

                    ## Initialize empty attributes
                    dbs, dbf = np.zeros((ds.shape[1],)).astype(np.float64), np.zeros((ds.shape[1],)).astype(np.int64)
                    TotalUMI, NGenes = np.zeros((ds.shape[1],)).astype(np.int64), np.zeros((ds.shape[1],)).astype(np.int64)
                    MT_ratio, unspliced_ratio = np.zeros((ds.shape[1],)).astype(np.float64), np.zeros((ds.shape[1],)).astype(np.float64)
                    tsne = np.zeros((ds.shape[1],2)).astype(np.float64)

                    for i, x in enumerate(RNA_bar):
                        k = match[x]
                        dbs[k] = dsr.ca['DoubletFinderScore'][i]
                        dbf[k] = dsr.ca['DoubletFinderFlag'][i]
                        TotalUMI[k] = dsr.ca['TotalUMI'][i]
                        unspliced_ratio[k] = dsr.ca['unspliced_ratio'][i]
                        NGenes[k] = dsr.ca['NGenes'][i]
                        MT_ratio[k] = dsr.ca['MT_ratio'][i]
                        tsne[k] = dsr.ca['TSNE'][i]
                        
                    ds.ca['DoubletFinderScore'] = dbs
                    ds.ca['DoubletFinderFlag'] = dbf
                    ds.ca['TotalUMI'] = TotalUMI
                    ds.ca['NGenes'] = NGenes
                    ds.ca['unspliced_ratio'] = unspliced_ratio
                    ds.ca['MT_ratio'] = MT_ratio
                    ds.ca['TSNE'] = tsne

        ## Plot QC
        with loompy.connect(self.loom) as ds:
            logging.info(f'Generating ATAC QC plots')
            if not 'TSNE' in ds.ca:
                add_TSNE(ds)
            sample_QC_plots(ds, ff, os.path.join(self.config.paths.qc, f'{sample}_QC_atac.png'))

        logging.info(f'Finished processing {sample}')

        ## Cleanup
        for file in glob.glob(os.path.join(outdir, '*.pkl')):
            os.system(f'rm {file}')

        return