# Description etc to be added

import numpy as np
import os
import sys
import pybedtools
from pybedtools import BedTool
import collections
import csv
import matplotlib.pyplot as plt
import gzip
import loompy
import scipy.sparse as sparse
import json
import urllib.request
import logging
import pickle
import importlib
import multiprocessing as mp
import tqdm
import gc

from chromograph.pipeline import config
from chromograph.preprocessing.utils import *
from chromograph.features.feature_count import *
from chromograph.preprocessing.doublet_finder import doublet_finder

class Chromgen:
    def __init__(self, rnaXatac=False) -> None:
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
        self.rnaXatac = rnaXatac
        pybedtools.helpers.set_bedtools_path(self.config.paths.bedtools)
        logging.info("Chromgen initialised")
    
    def fragments_to_count(self, ff, outdir, meta, bsize):
        '''
        '''
        ## Read Fragments and generate size bins
        logging.info("Read fragments into dict")
        frag_dict = read_fragments(ff)

        ## Split fragments to seperate files for fast indexing
        logging.info(f"Saving fragments to separate folder for fast indexing")
        fdir = os.path.join(outdir, 'fragments')
        if not os.path.isdir(fdir):
            os.mkdir(fdir)
        if  len(os.listdir(fdir)) < len(meta['barcode']):
            i = 0
            for x in meta['barcode']:
                f = os.path.join(fdir, f'{x}.tsv.gz')
                if not os.path.exists(f):
                    frags = BedTool(frag_dict[x]).saveas(f)
                i += 1
                if i%1000 == 0:
                    logging.info(f'Finished separating fragments for {i} cells')

        ## Count fragments inside bins
        logging.info("Count fragments overlapping with bins")
        Count_dict = count_bins(frag_dict, meta['barcode'], bsize)
        logging.info("Finished counting fragments")

        return Count_dict

    def fit(self, indir: str, bsize: int = 5000, outdir: str = None, genome_size: str = None, blacklist: str = None) -> None:
        ''''
        Create a .loom file from 10X Genomics cellranger output with reads binned
        Args:
            indir (str):	path to the cellranger output folder (the one that contains 'outs')
            bsize (int):	size of the bins (defaults to 5000/5kb)
            outdir (str):	output folder wher the new loom file should be saved (default to indir)
            genome_size (str):	path to file containing chromosome sizes, usually derived from encode (i.e. 'hg19.chrom.sizes.tsv')
            blacklist (str):	path to bedfile containing blacklisted region (i.e. 'blacklist_hg19.bed')
        Returns:
            path (str):		Full path to the created loom file.
        Remarks:
            The resulting file will be named ``{sampleID}.loom``, where the sampleID is the one given by cellranger.
        '''
        logging.info(f"Binning reads from {indir.split('/')[-1]} into {bsize/1000} kb bins")
        logging.info(f"Reading from {indir}")
        logging.info(f"Saving to {outdir}")
        if self.rnaXatac:
            logging.info(f'Multiome sample')
            fb = indir + '/outs/per_barcode_metrics.csv'
            ff = indir + '/outs/atac_fragments.tsv.gz'
            fs = indir + '/outs/summary.csv'
        else:
            fb = indir + '/outs/singlecell.csv'
            ff = indir + '/outs/fragments.tsv.gz'
            fs = indir + '/outs/summary.json'
        sample = indir.split('/')[-1]

        if len(sample.split('_')) > 2:
            sample = '_'.join(sample.split('_')[:2])

        if not os.path.isdir(outdir):
            os.mkdir(outdir)
   
        logging.info(f'Reading metadata and summary for {sample} from Cellranger output {fs}')
        if self.rnaXatac:
            summary = np.genfromtxt(fs, dtype=str, delimiter=',')
            summary = {str(k): str(v) for k, v in zip(summary[0,:], summary[1,:])}
        else:
            with open(fs, "r") as f:
                summary = json.load(f)

                for k,v in summary.items():
                    summary[k] = str(v)
        summary['bin_size'] = bsize
        summary['level'] = self.config.params.level
        
        if self.rnaXatac:
            barcodes = np.genfromtxt(fb, delimiter=',', skip_header=1,
                                    dtype={'names':('barcode','gex_barcode','atac_barcode','is__cell_barcode','excluded_reason','gex_raw_reads','gex_mapped_reads','gex_conf_intergenic_reads',
                                                    'gex_conf_exonic_reads','gex_conf_intronic_reads','gex_conf_exonic_unique_reads','gex_conf_exonic_antisense_reads',
                                                    'gex_conf_exonic_dup_reads','gex_exonic_umis','gex_conf_intronic_unique_reads','gex_conf_intronic_antisense_reads',
                                                    'gex_conf_intronic_dup_reads','gex_intronic_umis','gex_conf_txomic_unique_reads','gex_umis_count','gex_genes_count',
                                                    'total','unmapped','lowmapq','duplicate','chimeric','mitochondrial',
                                                    'passed_filters	','TSS_fragments','peak_region_fragments','peak_region_cutsites'),
                                            'formats':('U18', 'U18', 'U18', 'i8', 'i8', 'i8', 'i8', 'i8', 
                                                        'i8', 'i8', 'i8', 'i8', 
                                                        'i8', 'i8', 'i8', 'i8', 
                                                        'i8', 'i8', 'i8', 'i8', 'i8',
                                                        'i8', 'i8', 'i8', 'i8', 'i8', 'i8', 
                                                        'i8', 'i8', 'i8', 'i8')})
            barcodes['barcode'] = barcodes['atac_barcode'] ## We set the downstream barcode to the ATAC barcode instead of the defaults RNA barcode in cellranger
        else:
            barcodes = np.genfromtxt(fb, delimiter=',', skip_header=2,
                                    dtype={'names':('barcode','total','duplicate','chimeric','unmapped','lowmapq','mitochondrial','passed_filters','cell_id','is__cell_barcode',
                                                    'TSS_fragments','DNase_sensitive_region_fragments','enhancer_region_fragments','promoter_region_fragments','on_target_fragments',
                                                    'blacklist_region_fragments','peak_region_fragments','peak_region_cutsites'),
                                            'formats':('U18', 'i8', 'i8', 'i8', 'i8', 'i8', 'i8', 'i8', 'U18', 'i8', 'i8', 'i8', 'i8', 'i8', 'i8', 'i8', 'i8', 'i8')})

        ## Transfer metadata to dict format
        meta = {}
        passed = (barcodes['is__cell_barcode'] == 1) & (barcodes['passed_filters'] > self.config.params.level) & (barcodes['passed_filters'] < 100000)
        for key in barcodes.dtype.names:
            meta[key] = barcodes[key][passed]
        meta['CellID'] = [f'{sample}:{x}' for x in meta['barcode']]
    
        ## Retrieve sample metadata from SangerDB
        logging.info(f'Retrieve metadata from {[self.config.paths.metadata, sample]}')
        m =  load_sample_metadata(self.config.paths.metadata, sample)
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

        Count_dict = self.fragments_to_count(ff, outdir, meta, bsize)

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
        
        # Create sparse matrix
        
        logging.info("Generating Sparse matrix")
        col = []
        row = []
        v = []

        cix = 0
        for cell in meta['barcode']:

            for key in (Count_dict[cell]):
                col.append(cix)
                row.append(chrom_bins[key])
                v.append(int(Count_dict[cell][key]))
            cix+=1
        matrix = sparse.coo_matrix((v, (row,col)), shape=(len(chrom_bins.keys()), len(meta['barcode'])), dtype='int8')

        ## Save a smaller section of the summary
        keys = ['cellranger-atac_version', 'reference_assembly', 'reference_assembly_accession', 'reference_assembly_fasta_url', 'reference_organism', 'reference_version', 'bin_size']
        small_summary = {k: summary[k] for k in keys}
        
        ## We retain only the bins that have no overlap with the ENCODE blacklist
        matrix = matrix.tocsc()[retain,:]
        logging.info(f"Identified {len(v)} positive bins in {len(meta['barcode'])} cells before filtering blacklist")
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
        
        ## Doublet detection
        with loompy.connect(self.loom, 'r+') as ds:
            if ds.shape[1] > 1000:
                logging.info(f'Detecting doublets in loom file')
                ds.ca['DoubletFinderScore'], ds.ca['DoubletFinderFlag'] = doublet_finder(ds, proportion_artificial=.2, qc_dir = outdir, max_th=0.6)
            else:
                logging.info(f'Too few cells for doublet detections')
                ds.ca['DoubletFinderScore'], ds.ca['DoubletFinderFlag'] = np.zeros((ds.shape[1],)).astype(np.float64), np.zeros((ds.shape[1],)).astype(np.int64)
            meta['DoubletFinderScore'] = ds.ca['DoubletFinderScore']
            meta['DoubletFinderFlag'] = ds.ca['DoubletFinderFlag']

        logging.info(f'Finished processing {sample}')

        return