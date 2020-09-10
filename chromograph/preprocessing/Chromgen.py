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

sys.path.append('/home/camiel/chromograph/')
from chromograph.pipeline import config
from chromograph.preprocessing.utils import *
from chromograph.features.feature_count import *
from chromograph.preprocessing.doublet_finder import doublet_finder

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
        pybedtools.helpers.set_bedtools_path(self.config.paths.bedtools)
        logging.info("Chromgen initialised")
    
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
        logging.info("Binning reads from {} into {} kb bins".format(indir.split('/')[-1], (bsize/1000)))
        logging.info("Reading from {}".format(indir))
        fb = indir + '/outs/singlecell.csv'
        ff = indir + '/outs/fragments.tsv.gz'
        fs = indir + '/outs/summary.json'
        sample = indir.split('/')[-1]

        if len(sample.split('_')) > 2:
            sample = '_'.join(sample.split('_')[:2])

        if not os.path.isdir(outdir):
            os.mkdir(outdir)
   
        logging.info(f'Reading metadata and summary for {sample} from Cellranger output {fs}')
        with open(fs, "r") as f:
            summary = json.load(f)

            for k,v in summary.items():
                summary[k] = str(v)
        summary['bin_size'] = bsize
        summary['level'] = self.config.params.level
        
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

        logging.info("Total of {} valid cells".format(len(meta['barcode'])))
        logging.info("Ref. assembly {}".format(summary['reference_assembly']))

        # Get Chromosome sizes
        if genome_size == None:
            chrom_size = get_chrom_sizes(summary['reference_assembly'])    
        else:
            chrom_size = {}
            with open(genome_size) as f:
                for line in f:
                    x = line.split()
                    chrom_size[x[0]] = int(x[1])

        ## Read Fragments and generate size bins
        logging.info("Read fragments into dict")
        frag_dict = read_fragments(ff)
        
        ## Split fragments to seperate files for fast indexing
        logging.info(f"Saving fragments to separate folder for fast indexing")
        fdir = os.path.join(outdir, 'fragments')
        if not os.path.isdir(fdir):
            os.mkdir(fdir)

        ## Save fragments to folder
        i = 0
        for x in meta['barcode']:
            f = os.path.join(fdir, f'{x}.tsv.gz')
            frags = BedTool(frag_dict[x]).saveas(f)
            i += 1
            if i%1000 == 0:
                logging.info(f'Finished separating fragments for {i} cells')
                
        del frags
        
        logging.info("Generate {} bins based on provided chromosome sizes".format(str(int(bsize/1000)) + ' kb'))
        chrom_bins = generate_bins(chrom_size, bsize)

        ## Count fragments inside bins
        logging.info("Count fragments overlapping with bins")
        Count_dict = count_bins(frag_dict, meta['barcode'], bsize)
        logging.info("Finished counting fragments")

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
        
        logging.info("Number of bins after cleaning: {}".format(len(clean_bin)))

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
        matrix = sparse.coo_matrix((v, (row,col)), shape=(len(chrom_bins.keys()), len(meta['barcode'])))

        ## Save a smaller section of the summary
        keys = ['cellranger-atac_version', 'reference_assembly', 'reference_assembly_accession', 'reference_assembly_fasta_url', 'reference_organism', 'reference_version', 'bin_size']
        small_summary = {k: summary[k] for k in keys}
        
        ## We retain only the bins that have no overlap with the ENCODE blacklist
        cleaned_matrix = matrix.tocsc()[retain,:]
        logging.info("Identified {} positive bins in {} cells before filtering blacklist".format(len(v), len(meta['barcode'])))
        logging.info("Identified {} positive bins in {} cells after filtering blacklist".format(len(cleaned_matrix.nonzero()[0]), len(meta['barcode'])))

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

        loompy.create(filename=floom, 
                      layers=cleaned_matrix, 
                      row_attrs=row_attrs, 
                      col_attrs=meta,
                      file_attrs=small_summary)
        self.loom = floom
        logging.info("Loom bin file saved as {}".format(floom))
        
        ## Cleanup
        del black_list, Count_dict, chrom_bins, chrom_size, intervals, cleaned, keep, retain, clean_bin, matrix, cleaned_matrix, col, row, v
        
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